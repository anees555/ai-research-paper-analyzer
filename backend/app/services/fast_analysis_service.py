import os
import sys
import logging
import asyncio
from typing import Dict, Any
from functools import lru_cache
import concurrent.futures

# Add project root and src to path to import legacy scripts
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from app.data_models.schemas import AnalysisResult, PaperMetadata
from app.services.optimized_model_loader import optimized_model_engine
from app.services.enhanced_analysis_service import enhanced_analysis_service
from scripts.parse_pdf_optimized import parse_pdf_with_grobid_optimized as parse_pdf_with_grobid
from scripts.preprocess_text import clean_text_comprehensive, chunk_text_for_models

logger = logging.getLogger(__name__)


class FastAnalysisService:

    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.model_engine = optimized_model_engine

    # ---------------------------------------------------
    # MAIN ENTRY
    # ---------------------------------------------------
    async def analyze_fast(self, file_path: str, job_id: str) -> AnalysisResult:
        try:
            paper_data = await self._extract_structure_fast(file_path)
            result = await self._generate_fast_analysis(paper_data)
            return result
        except Exception as e:
            logger.error(f"[FAST] Analysis failed for job {job_id}: {str(e)}")
            raise

    # ---------------------------------------------------
    # PDF STRUCTURE EXTRACTION
    # ---------------------------------------------------
    async def _extract_structure_fast(self, file_path: str) -> Dict[str, Any]:
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        timeout = min(60, max(30, int(file_size * 10)))

        logger.info(f"GROBID timeout: {timeout}s for {file_size:.1f}MB file")

        loop = asyncio.get_event_loop()
        paper_data = await loop.run_in_executor(
            self.executor,
            self._run_grobid_with_timeout,
            file_path,
            timeout
        )

        if "paper_id" not in paper_data:
            paper_data["paper_id"] = os.path.basename(file_path).replace(".pdf", "")

        return paper_data

    def _run_grobid_with_timeout(self, file_path: str, timeout: int) -> Dict[str, Any]:
        try:
            result = parse_pdf_with_grobid(file_path, "output")
            return result
        except Exception as e:
            raise e

    # ---------------------------------------------------
    # FAST SUMMARY
    # ---------------------------------------------------
    async def _generate_fast_analysis(self, paper_data: Dict) -> AnalysisResult:

        abstract = paper_data.get("abstract", "")
        sections = paper_data.get("sections", {})

        summarizer = await self._get_fast_summarizer()
        if summarizer is None:
            return self._generate_fallback_summary(paper_data)

        async def summarize(text, section_name):
            import torch
            import re

            # Limit input to first 2 paragraphs for clarity
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
            limited_text = '\n'.join(paragraphs[:2])

            # Use a high-level, non-technical prompt
            prompt = (
                f"Write a simple, clear summary for a general audience. "
                f"Focus on what the paper is about, the problem it addresses, the method used, and the main conclusion. "
                f"Avoid technical jargon. Section: {section_name}. "
            )
            inputs = summarizer.tokenizer(
                [prompt + limited_text],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(summarizer.device)

            with torch.no_grad():
                summary_ids = summarizer.model.generate(
                    **inputs,
                    num_beams=4,
                    min_length=30,
                    max_length=100
                )

            summary = summarizer.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            # Remove any prompt echoes or leading/trailing whitespace
            summary = summary.replace(prompt, "").replace(f"Summarize {section_name.lower()}: ", "").strip()

            # Post-process: keep only first 2-3 sentences, remove incomplete or repeated lines
            sentences = re.split(r'(?<=[.!?]) +', summary)
            seen = set()
            clean_sentences = []
            for s in sentences:
                s = s.strip()
                if len(s) < 8 or s.lower() in seen:
                    continue
                seen.add(s.lower())
                clean_sentences.append(s)
                if len(clean_sentences) >= 3:
                    break
            summary_clean = ' '.join(clean_sentences).strip()
            return summary_clean


        # Build summary with each section as a separate paragraph (no prefix)
        quick_parts = []
        if abstract:
            quick_parts.append(abstract.strip())

        intro = sections.get("Introduction", "")
        if intro:
            s = await summarize(intro, "Introduction")
            quick_parts.append(s)

        methods = sections.get("Method", "") or sections.get("Model Architecture", "")
        if methods:
            s = await summarize(methods, "Methods")
            quick_parts.append(s)

        conclusion = sections.get("Conclusion", "")
        if conclusion:
            s = await summarize(conclusion, "Conclusion")
            quick_parts.append(s)

        quick_summary = "\n\n".join(quick_parts)

        return AnalysisResult(
            metadata=PaperMetadata(
                title=paper_data.get("title", "Unknown Paper"),
                authors=paper_data.get("authors", []),
                paper_id=paper_data.get("paper_id", ""),
                num_sections=len(sections),
                processing_method="Fast Mode"
            ),
            quick_summary=quick_summary,
            detailed_summary={},
            comprehensive_analysis={"summary": quick_summary},
            original_abstract=abstract
        )

    # ---------------------------------------------------
    # FALLBACK SUMMARY
    # ---------------------------------------------------
    def _generate_fallback_summary(self, paper_data: Dict) -> AnalysisResult:
        abstract = paper_data.get("abstract", "")
        intro = paper_data.get("sections", {}).get("Introduction", "")

        text = abstract if abstract else intro
        sentences = text.split(".")[:3]
        summary = ". ".join(sentences)

        return AnalysisResult(
            metadata=PaperMetadata(
                title=paper_data.get("title", "Unknown Paper"),
                authors=paper_data.get("authors", []),
                paper_id=paper_data.get("paper_id", ""),
                num_sections=len(paper_data.get("sections", {})),
                processing_method="Fallback"
            ),
            quick_summary=summary,
            detailed_summary={},
            comprehensive_analysis={},
            original_abstract=abstract
        )

    # ---------------------------------------------------
    # MODEL LOADING
    # ---------------------------------------------------

    async def _get_fast_summarizer(self):
        if not hasattr(self, "_fast_summarizer") or self._fast_summarizer is None:
            self._fast_summarizer = await self.model_engine.get_lightweight_summarizer()
        return self._fast_summarizer

    async def _get_full_summarizer(self):
        if not hasattr(self, "_full_summarizer") or self._full_summarizer is None:
            self._full_summarizer = await self.model_engine.get_full_summarizer()
        return self._full_summarizer


fast_analysis_service = FastAnalysisService()