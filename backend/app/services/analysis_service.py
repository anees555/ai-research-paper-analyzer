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
from app.services.model_loader import model_engine
from scripts.parse_pdf_optimized import parse_pdf_with_grobid_optimized as parse_pdf_with_grobid
from scripts.preprocess_text import clean_text_comprehensive, chunk_text_for_models

logger = logging.getLogger(__name__)

class AnalysisService:
    def __init__(self):
        self.model_engine = model_engine
        self._paper_data_cache = {}  # Store paper data for chat indexing

    def get_paper_data(self, job_id: str):
        """Get stored paper data for a job"""
        return self._paper_data_cache.get(job_id)
        
    def _store_paper_data(self, job_id: str, paper_data: Dict):
        """Store paper data for later use by chat service"""
        self._paper_data_cache[job_id] = paper_data
    async def analyze_paper(self, file_path: str, job_id: str) -> AnalysisResult:
        """
        Main entry point for analyzing a paper.
        """
        logger.info(f"Starting analysis for job {job_id} on file {file_path}")
        
        try:
            # 1. Extract Structure (GROBID)
            # using 'output' dir relative to root or temp? 
            # We'll use a temp dir or standard output dir.
            paper_data = parse_pdf_with_grobid(file_path, "output")
            
            # Store paper data for chat indexing
            self._store_paper_data(job_id, paper_data)
            
            # 2. Generate Enhanced Summaries
            summarizer = self.model_engine.get_summarizer()
            
            # Generate traditional summaries
            quick_summary = self._generate_quick_summary(paper_data, summarizer)
            detailed = self._generate_detailed_summary(paper_data, summarizer)
            comprehensive = self._generate_comprehensive_analysis(paper_data, summarizer)
            
            # 3. Construct Result
            metadata = PaperMetadata(
                title=paper_data.get("title", "Unknown"),
                authors=paper_data.get("authors", []),
                paper_id=os.path.basename(file_path).replace('.pdf', ''),
                num_sections=len(paper_data.get("sections", {})),
                processing_method="AI-Enhanced" if summarizer else "Heuristic"
            )
            
            return AnalysisResult(
                metadata=metadata,
                quick_summary=quick_summary,
                detailed_summary=detailed,
                comprehensive_analysis=comprehensive,
                original_abstract=paper_data.get("abstract", "")
            )

        except Exception as e:
            logger.error(f"Analysis failed for {job_id}: {e}")
            raise e

    def _generate_quick_summary(self, paper_data: Dict, summarizer) -> str:
        abstract = paper_data.get("abstract", "")
        if not abstract:
            return "No abstract available."
        
        clean_abs = clean_text_comprehensive(abstract)
        
        if summarizer and len(clean_abs) > 100:
            try:
                res = summarizer(clean_abs, max_length=100, min_length=30, do_sample=False)
                return res[0]['summary_text']
            except Exception as e:
                logger.warning(f"Summarization failed: {e}")
                return clean_abs[:200] + "..."
        return clean_abs[:200] + "..."

    def _generate_detailed_summary(self, paper_data: Dict, summarizer) -> Dict[str, str]:
        detailed = {}
        sections = paper_data.get("sections", {})
        key_sections = ["Introduction", "Methods", "Results", "Conclusion"]
        
        for name, text in sections.items():
            # Filter for key sections or process all? The original code did key sections.
            # But let's be flexible.
            if len(text) < 200:
                continue
                
            clean_txt = clean_text_comprehensive(text)
            if summarizer:
                try:
                    # chunking logic simplified
                    chunks = chunk_text_for_models(clean_txt, "bart")
                    parts = []
                    for chunk in chunks[:2]:
                        res = summarizer(chunk["text"], max_length=80, min_length=20, do_sample=False)
                        parts.append(res[0]['summary_text'])
                    detailed[name] = " ".join(parts)
                except Exception:
                    detailed[name] = clean_txt[:300] + "..."
            else:
                detailed[name] = clean_txt[:300] + "..."
                
        return detailed

    def _generate_comprehensive_analysis(self, paper_data: Dict, summarizer) -> Dict[str, Any]:
        # Similar to original logic
        analysis = {}
        sections = paper_data.get("sections", {})
        
        if "Introduction" in sections:
            intro = clean_text_comprehensive(sections["Introduction"])
            if summarizer:
                try:
                     res = summarizer(intro[:1024], max_length=100, min_length=30, do_sample=False)
                     analysis["main_contribution"] = res[0]['summary_text']
                except:
                    analysis["main_contribution"] = intro[:300]
        
        return analysis

analysis_service = AnalysisService()
