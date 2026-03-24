import os
import sys
import logging
from typing import Dict, Any, List

# Add project root and src to path to import legacy scripts
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)


from app.data_models.schemas import AnalysisResult, PaperMetadata
from app.services.optimized_model_loader import optimized_model_engine
from scripts.parse_pdf_optimized import parse_pdf_with_grobid_optimized as parse_pdf_with_grobid
from scripts.preprocess_text import clean_text_comprehensive, chunk_text_for_models

logger = logging.getLogger(__name__)

class AnalysisService:
    def __init__(self):
        self.model_engine = optimized_model_engine
        self._paper_data_cache = {}  # Store paper data for chat indexing
        self.summarizer = None
        self.device = None
        self.tokenizer = None
        self.model_name = None

    def get_paper_data(self, job_id: str):
        """Get stored paper data for a job"""
        return self._paper_data_cache.get(job_id)
        
    def _store_paper_data(self, job_id: str, paper_data: Dict):
        """Store paper data for later use by chat service"""
        self._paper_data_cache[job_id] = paper_data
    async def analyze_paper(self, file_path: str, job_id: str) -> AnalysisResult:
        """
        Main entry point for analyzing a paper with robust section-wise summarization.
        """
        logger.info(f"Starting analysis for job {job_id} on file {file_path}")
        try:
            # 1. Extract Structure (GROBID)
            paper_data = parse_pdf_with_grobid(file_path, "output")
            if "paper_id" not in paper_data:
                paper_data["paper_id"] = os.path.basename(file_path).replace('.pdf', '')
            self._store_paper_data(job_id, paper_data)

            # DEBUG: Log all parsed section names
            parsed_sections = list((paper_data.get("sections", {}) or {}).keys())
            logger.info(f"[DEBUG] Parsed section names: {parsed_sections}")

            # 2. Normalize and clean sections
            normalized_sections = self.normalize_and_clean_sections(paper_data)
            logger.info(f"[DEBUG] Normalized section names: {list(normalized_sections.keys())}")

            # 3. Load fastest available summarizer and tokenizer
            if not self.summarizer:
                self.summarizer = await self.model_engine.get_lightweight_summarizer()
                if self.summarizer is None:
                    raise RuntimeError("No summarization model available!")
                # Get tokenizer and device from pipeline
                self.tokenizer = self.summarizer.tokenizer
                self.model_name = self.summarizer.model.config.name_or_path
                self.device = self.summarizer.device
                logger.info(f"[MODEL] Using summarizer: {self.model_name} on device {self.device}")

            # 4. Section-wise summarization (parallelized)
            section_summaries = await self.summarize_sections_parallel(normalized_sections)
            logger.info(f"[DEBUG] Section summaries generated: {list(section_summaries.keys())}")

            # 5. Output formatting
            summary_text = self.format_section_summaries(section_summaries)

            # 6. Metadata
            metadata = PaperMetadata(
                title=paper_data.get("title", "Unknown"),
                authors=paper_data.get("authors", []),
                paper_id=os.path.basename(file_path).replace('.pdf', ''),
                num_sections=len(normalized_sections),
                processing_method=f"{self.model_name}-Sectionwise"
            )

            return AnalysisResult(
                metadata=metadata,
                quick_summary=section_summaries.get("Abstract", ""),
                detailed_summary=section_summaries,
                comprehensive_analysis={
                    "html_summary": summary_text,
                    "summary": summary_text,
                    "processing_mode": f"{self.model_name}-sectionwise",
                    "details": f"Section-wise summarization using {self.model_name} pipeline"
                },
                original_abstract=paper_data.get("abstract", "")
            )
        except Exception as e:
            logger.error(f"Analysis failed for {job_id}: {e}")
            raise e
    # --- Section Normalization and Cleaning ---
    def normalize_and_clean_sections(self, paper_data: Dict) -> Dict[str, str]:
        section_map = {
            "Abstract": ["abstract"],
            "Introduction": ["introduction", "background", "motivation", "overview"],
            "Method": ["method", "methodology", "proposed method", "model", "architecture", "approach", "system"],
            "Experiments": ["experiment", "evaluation", "results", "analysis", "performance", "experimental setup"],
            "Conclusion": ["conclusion", "discussion", "future work", "summary"],
        }
        sections = []
        if paper_data.get("abstract"):
            sections.append(("abstract", paper_data["abstract"]))
        for k, v in (paper_data.get("sections", {}) or {}).items():
            sections.append((k, v))

        normalized = {}
        for raw_name, text in sections:
            norm_name = self.match_section_name(raw_name, section_map)
            if norm_name:
                cleaned = self.clean_section_text(text)
                if cleaned and len(cleaned) > 40:
                    if norm_name in normalized:
                        normalized[norm_name] += "\n" + cleaned
                    else:
                        normalized[norm_name] = cleaned
        return normalized

    def match_section_name(self, name: str, section_map: Dict[str, List[str]]) -> str:
        name_lc = name.lower()
        for norm, keywords in section_map.items():
            for kw in keywords:
                if kw in name_lc:
                    return norm
        return None

    def clean_section_text(self, text: str) -> str:
        import re
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\(([^)]*\d{4}[^)]*)\)", "", text)
        noisy_patterns = [
            r"author contributions?", r"affiliations?", r"acknowledg(e)?ments?", r"references?", r"footnotes?",
            r"conflict of interest", r"funding", r"ethics statement", r"supplementary", r"appendix"
        ]
        lines = text.splitlines()
        cleaned_lines = []
        for line in lines:
            l = line.strip()
            if not l:
                continue
            if any(re.search(pat, l, re.IGNORECASE) for pat in noisy_patterns):
                continue
            if len(l) < 5 or l.isdigit():
                continue
            cleaned_lines.append(l)
        cleaned = " ".join(cleaned_lines)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    # --- Section-wise Summarization ---
    async def summarize_sections_parallel(self, normalized_sections: Dict[str, str]) -> Dict[str, str]:
        import asyncio
        gen_settings = {
            "Abstract": dict(max_length=600, min_length=300),
            "Introduction": dict(max_length=1600, min_length=480),
            "Method": dict(max_length=1400, min_length=400),
            "Experiments": dict(max_length=1400, min_length=400),
            "Conclusion": dict(max_length=800, min_length=240),
        }
        summaries = {}
        tasks = []
        # Summarize Abstract using only the original abstract text if available
        if "Abstract" in normalized_sections:
            try:
                paper_data = next(iter(self._paper_data_cache.values()), None)
                orig_abstract = None
                if paper_data and "abstract" in paper_data:
                    orig_abstract = paper_data["abstract"]
                if orig_abstract and len(orig_abstract) > 40:
                    abstract_text = orig_abstract
                else:
                    abstract_text = normalized_sections["Abstract"]
            except Exception:
                abstract_text = normalized_sections["Abstract"]
            prompt = self.build_instruction_prompt("Abstract", abstract_text)
            settings = gen_settings["Abstract"]
            tasks.append(self.summarize_long_text(prompt, settings, section_name="Abstract"))
        # Summarize other sections in parallel
        for section in ["Introduction", "Method", "Experiments", "Conclusion"]:
            if section in normalized_sections:
                text = normalized_sections[section]
                prompt = self.build_instruction_prompt(section, text)
                settings = gen_settings.get(section, dict(max_length=360, min_length=120))
                tasks.append(self.summarize_long_text(prompt, settings, section_name=section))
        results = await asyncio.gather(*tasks)
        for i, section in enumerate([s for s in ["Abstract", "Introduction", "Method", "Experiments", "Conclusion"] if s in normalized_sections]):
            summaries[section] = self.postprocess_summary(results[i])
        return summaries

    def build_instruction_prompt(self, section: str, text: str) -> str:
        instruction = (
            "Summarize and explain this research section in simple terms, "
            "highlighting the main idea, methodology, and key results. "
            "Make it easy to understand for students."
        )
        return f"{instruction}\nSection: {section}\n{text}"

    async def summarize_long_text(self, prompt: str, gen_kwargs: Dict, section_name: str = "") -> str:
        # Always chunk to fit model's max input size
        chunks = self.split_into_chunks(prompt, self.tokenizer, max_tokens=480, stride=128)
        logger.info(f"[CHUNKING] Section '{section_name}' split into {len(chunks)} chunks.")
        chunk_summaries = await self.summarize_chunks_batch(chunks, gen_kwargs)
        merged = await self.merge_summaries(chunk_summaries, gen_kwargs)
        return merged

    def format_section_summaries(self, section_summaries: Dict[str, str]) -> str:
        order = ["Abstract", "Introduction", "Method", "Experiments", "Conclusion"]
        out = ["Paper Summary\n"]
        for sec in order:
            if sec in section_summaries:
                out.append(f"{sec}:\n{section_summaries[sec]}\n")
        return "\n".join(out).strip()

    # --- Modular Summarization Utilities ---
    def split_into_chunks(self, text, tokenizer, max_tokens=480, stride=128):
        # max_tokens=480 to fit within Flan-T5-large 512 token limit (with margin for special tokens)
        input_ids = tokenizer.encode(text, truncation=False)
        chunks = []
        start = 0
        while start < len(input_ids):
            end = min(start + max_tokens, len(input_ids))
            chunk_ids = input_ids[start:end]
            chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            if end < len(input_ids):
                last_period = chunk.rfind('.')
                if last_period > 0:
                    chunk = chunk[:last_period+1]
                    end = start + tokenizer.encode(chunk, truncation=False).__len__()
            chunks.append(chunk.strip())
            if end == len(input_ids):
                break
            start += max_tokens - stride
        return [c for c in chunks if c]

    async def summarize_chunks_batch(self, chunks, gen_kwargs):
        import torch
        # Use larger batch size if on GPU
        batch_size = 8 if self.device != -1 and str(self.device) != "cpu" else 4
        all_summaries = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                summary_ids = self.summarizer.model.generate(
                    **inputs,
                    num_beams=gen_kwargs.get('num_beams', 4),
                    no_repeat_ngram_size=gen_kwargs.get('no_repeat_ngram_size', 3),
                    length_penalty=gen_kwargs.get('length_penalty', 2.0),
                    min_length=gen_kwargs.get('min_length', 60),
                    max_length=gen_kwargs.get('max_length', 180),
                    early_stopping=gen_kwargs.get('early_stopping', False)
                )
            summaries = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
            all_summaries.extend(summaries)
        return all_summaries

    async def merge_summaries(self, summaries, gen_kwargs):
        merged = " ".join(summaries)
        if len(self.tokenizer.encode(merged)) > gen_kwargs.get('max_length', 180):
            merge_kwargs = gen_kwargs.copy()
            merge_kwargs['max_length'] = min(1024, merge_kwargs.get('max_length', 4000) * 2)
            merged_chunks = self.split_into_chunks(merged, self.tokenizer, max_tokens=1024)
            merged_summaries = await self.summarize_chunks_batch(merged_chunks, merge_kwargs)
            merged = " ".join(merged_summaries)
        return merged.strip()

    def postprocess_summary(self, summary):
        import re
        cleaned = summary
        # Remove only the 'Paper Summary Abstract:' section and its content, up to the next section header or end
        cleaned = re.sub(r'Paper Summary Abstract:[\s\S]*?(?=(\n[A-Z][a-z]+\n|Introduction:|Method:|Conclusion:|Results:|Experiment:|$))', '', cleaned, flags=re.IGNORECASE)
        # Remove extra whitespace left by removal
        cleaned = re.sub(r'\n{2,}', '\n\n', cleaned)
        # Remove URLs, emails, and numbers
        cleaned = re.sub(r'https?://\S+|www\.\S+', '', cleaned)
        cleaned = re.sub(r'\S+@\S+', '', cleaned)
        cleaned = re.sub(r'\b\d+\b', '', cleaned)
        # Remove bracketed numbers like [1], [2]
        cleaned = re.sub(r'\[\d+\]', '', cleaned)
        # Remove long garbage tokens
        cleaned = re.sub(r'\b\w{30,}\b', '', cleaned)
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def get_section_settings(self, section_name):
        """Return generation settings for each section type."""
        # These can be tuned further
        section_map = {
            'abstract': dict(min_length=30, max_length=120),
            'introduction': dict(min_length=120, max_length=300),
            'methods': dict(min_length=80, max_length=200),
            'results': dict(min_length=80, max_length=200),
            'conclusion': dict(min_length=30, max_length=120),
        }
        key = section_name.lower()
        for k in section_map:
            if k in key:
                return section_map[k]
        return dict(min_length=60, max_length=180)

    async def _generate_quick_summary(self, paper_data: Dict, summarizer) -> str:
        section_label = "Abstract"
        abstract = paper_data.get("abstract", "")
        if not abstract:
            return f"{section_label}: No abstract available."
        clean_abs = clean_text_comprehensive(abstract)
        if not summarizer or len(clean_abs) < 50:
            return f"{section_label}: " + clean_abs[:200] + ("..." if len(clean_abs) > 200 else "")
        settings = self.get_section_settings("abstract")
        gen_kwargs = dict(
            num_beams=4,
            no_repeat_ngram_size=3,
            length_penalty=2.0,
            early_stopping=False,
            min_length=settings['min_length'],
            max_length=settings['max_length']
        )
        chunks = self.split_into_chunks(clean_abs, self.tokenizer, max_tokens=16000, stride=500)
        chunk_summaries = await self.summarize_chunks_batch(chunks, gen_kwargs)
        merged = await self.merge_summaries(chunk_summaries, gen_kwargs)
        return f"{section_label}: " + self.postprocess_summary(merged)

    async def _generate_detailed_summary(self, paper_data: Dict, summarizer) -> Dict[str, str]:
        detailed = {}
        sections = paper_data.get("sections", {})
        if not summarizer:
            for name, text in sections.items():
                if len(text) < 100:
                    continue
                detailed[name] = f"{name}: " + clean_text_comprehensive(text)
            return detailed
        for name, text in sections.items():
            if len(text) < 100:
                continue
            clean_txt = clean_text_comprehensive(text)
            settings = self.get_section_settings(name)
            gen_kwargs = dict(
                num_beams=4,
                no_repeat_ngram_size=3,
                length_penalty=2.0,
                early_stopping=False,
                min_length=settings['min_length'],
                max_length=settings['max_length']
            )
            chunks = self.split_into_chunks(clean_txt, self.tokenizer, max_tokens=16000, stride=500)
            chunk_summaries = await self.summarize_chunks_batch(chunks, gen_kwargs)
            merged = await self.merge_summaries(chunk_summaries, gen_kwargs)
            detailed[name] = f"{name}: " + self.postprocess_summary(merged)
        return detailed

    async def _generate_comprehensive_analysis(self, paper_data: Dict, summarizer) -> Dict[str, Any]:
        analysis = {}
        sections = paper_data.get("sections", {})
        if not summarizer:
            return analysis
        if "Introduction" in sections:
            intro = clean_text_comprehensive(sections["Introduction"])
            settings = self.get_section_settings("introduction")
            gen_kwargs = dict(
                num_beams=4,
                no_repeat_ngram_size=3,
                length_penalty=2.0,
                early_stopping=False,
                min_length=settings['min_length'],
                max_length=settings['max_length']
            )
            chunks = self.split_into_chunks(intro, self.tokenizer, max_tokens=16000, stride=500)
            chunk_summaries = await self.summarize_chunks_batch(chunks, gen_kwargs)
            merged = await self.merge_summaries(chunk_summaries, gen_kwargs)
            analysis["main_contribution"] = self.postprocess_summary(merged)
        return analysis
    

analysis_service = AnalysisService()
