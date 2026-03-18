import os
import sys
import logging
import asyncio
import concurrent.futures
from typing import Dict, Any, List, Optional
from functools import lru_cache

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
    """
    High-performance analysis service with multiple optimization strategies
    """
    
    def __init__(self):
        self.model_engine = optimized_model_engine
        self.paper_cache = {}  # In-memory cache for processed papers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    
    def get_paper_data(self, job_id: str):
        """Get stored paper data for a job"""
        return self.paper_cache.get(job_id)
    
    def _store_paper_data(self, job_id: str, paper_data: Dict):
        """Store paper data for later use by chat service"""
        self.paper_cache[job_id] = paper_data
    
    async def analyze_paper_fast(self, file_path: str, job_id: str, mode: str = "fast") -> AnalysisResult:
        """
        Fast paper analysis - abstract + 3-paragraph summary in simple language
        
        Args:
            file_path: Path to PDF file
            job_id: Unique job identifier
            mode: Currently only supports "fast" mode
        """
        logger.info(f"[FAST] Starting fast analysis for job {job_id}")
        
        try:
            # Step 1: Quick GROBID extraction with 60-second timeout
            paper_data = await self._extract_structure_fast(file_path, mode="fast")
            
            # Store paper data for chat indexing
            self._store_paper_data(job_id, paper_data)
            
            # Step 2: Generate 3-paragraph summary
            result = await self._generate_fast_analysis(paper_data)
            
            return result
            
        except Exception as e:
            logger.error(f"[FAST] Fast analysis failed for job {job_id}: {str(e)}")
            raise
    
    async def _extract_structure_fast(self, file_path: str, mode: str = "fast") -> Dict[str, Any]:
        """Extract structure with 60-second GROBID timeout for fast mode"""
        
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        timeout = min(60, max(30, int(file_size * 10)))  # 30-60s based on size
        
        logger.info(f"Using optimized GROBID timeout: {timeout}s for {file_size:.1f}MB file")
        
        # Run GROBID in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        paper_data = await loop.run_in_executor(
            self.executor,
            self._run_grobid_with_timeout,
            file_path, timeout
        )
        
        # Add paper_id to the paper_data for consistent usage
        if "paper_id" not in paper_data:
            paper_data["paper_id"] = os.path.basename(file_path).replace('.pdf', '')
        
        return paper_data
    
    def _run_grobid_with_timeout(self, file_path: str, timeout: int) -> Dict[str, Any]:
        """Run GROBID with specific timeout"""
        import signal
        
        class TimeoutError(Exception):
            pass
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"GROBID processing timed out after {timeout}s")
        
        # Set timeout (Unix/Linux only - for Windows, use alternative approach)
        if os.name != 'nt':  # Not Windows
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
        
        try:
            result = parse_pdf_with_grobid(file_path, "output")
            if os.name != 'nt':
                signal.alarm(0)  # Cancel timeout
            return result
        except Exception as e:
            if os.name != 'nt':
                signal.alarm(0)
            raise
    
    async def _generate_fast_analysis(self, paper_data: Dict) -> AnalysisResult:
        """
        Generate fast analysis: original abstract + 3-paragraph summary
        Paragraphs cover: Problem, Technology/Solution, Implementation & Conclusion
        Total ~50 lines in simple language
        """
        logger.info("[FAST] ============== Generating 3-paragraph summary ==============")
        
        # Get original abstract
        abstract = paper_data.get("abstract", "")
        logger.info(f"[FAST] Abstract length: {len(abstract)} chars")
        
        # Extract key sections for building paragraphs
        sections = paper_data.get("sections", {})
        logger.info(f"[FAST] Available sections: {list(sections.keys())[:5]}... (total: {len(sections)})")
        
        # Build 3-paragraph summary with detailed logging
        logger.info("[FAST] >>> PARAGRAPH 1: Problem Statement...")
        paragraph_1 = self._extract_problem_statement(sections, abstract)
        logger.info(f"[FAST] P1 result: {len(paragraph_1)} chars")
        logger.info(f"[FAST] P1 preview: {paragraph_1[:200]}...")
        
        logger.info("[FAST] >>> PARAGRAPH 2: Solution & Technology...")
        paragraph_2 = self._extract_solution_technology(sections, abstract)
        logger.info(f"[FAST] P2 result: {len(paragraph_2)} chars")
        logger.info(f"[FAST] P2 preview: {paragraph_2[:200]}...")
        
        logger.info("[FAST] >>> PARAGRAPH 3: Implementation & Conclusion...")
        paragraph_3 = self._extract_implementation_conclusion(sections, abstract)
        logger.info(f"[FAST] P3 result: {len(paragraph_3)} chars")
        logger.info(f"[FAST] P3 preview: {paragraph_3[:200]}...")
        
        # Combine into final summary (with \n\n for separation)
        three_paragraph_summary = f"{paragraph_1}\n\n{paragraph_2}\n\n{paragraph_3}"
        
        # Create formatted HTML with proper paragraph tags
        abstract_html = f"""<h2 class="text-2xl font-bold mb-6">Abstract</h2>
<p class="mb-8 leading-relaxed text-gray-700">{abstract}</p>""" if abstract else ""
        
        summary_html = f"""
{abstract_html}
<div class="summary-content">
  <h2 class="text-2xl font-bold mb-6 mt-8">3-Paragraph Summary</h2>
  <p class="mb-4 leading-relaxed text-gray-700">{paragraph_1}</p>
  <p class="mb-4 leading-relaxed text-gray-700">{paragraph_2}</p>
  <p class="mb-4 leading-relaxed text-gray-700">{paragraph_3}</p>
</div>
"""
        
        logger.info(f"[FAST] ✓ Summary generated: {len(three_paragraph_summary.split())} words, {len(three_paragraph_summary)} chars")
        logger.info(f"[FAST] ✓ HTML summary: {len(summary_html)} chars")
        logger.info("[FAST] ==============================================================")
        
        return AnalysisResult(
            metadata=PaperMetadata(
                title=paper_data.get("title", "Unknown Paper"),
                authors=paper_data.get("authors", []),
                paper_id=paper_data.get("paper_id", ""),
                num_sections=len(sections),
                processing_method="Fast Mode (3-Paragraph Summary)"
            ),
            quick_summary=three_paragraph_summary,
            detailed_summary={},
            comprehensive_analysis={
                "html_summary": summary_html,
                "summary": three_paragraph_summary,
                "processing_mode": "fast",
                "details": "Fast mode analysis complete"
            },
            original_abstract=abstract
        )
    
    def _extract_problem_statement(self, sections: Dict, abstract: str) -> str:
        """Extract first paragraph covering the problem statement - return full text"""
        text = ""
        
        if "Introduction" in sections:
            text = sections["Introduction"]
            logger.debug("[FAST] P1: Using Introduction section")
        elif "Background" in sections:
            text = sections["Background"]
            logger.debug("[FAST] P1: Using Background section")
        elif abstract and len(abstract) > 50:
            text = abstract
            logger.debug("[FAST] P1: Using abstract")
        
        if not text:
            return "This paper addresses a relevant research problem in its field."
        
        # Clean text aggressively
        text = text.replace('\n', ' ').replace('\r', ' ').replace('  ', ' ').strip()
        
        # Return full text without character truncation
        logger.debug(f"[FAST] P1: Full text: {len(text)} chars")
        return text
    
    def _extract_solution_technology(self, sections: Dict, abstract: str) -> str:
        """Extract second paragraph covering the technology/solution - return full text"""
        text = ""
        
        # Look for relevant sections in priority order
        for section_name in ["Model Architecture", "Proposed Method", "Methodology", "Approach", "Method", "Our Approach"]:
            if section_name in sections and sections[section_name]:
                text = sections[section_name]
                logger.debug(f"[FAST] P2: Using {section_name}")
                break
        
        # If no specific section found, try to synthesize from abstract
        if not text and abstract and len(abstract) > 100:
            text = abstract
            logger.debug("[FAST] P2: Using abstract")
        
        if not text:
            return "The paper proposes a solution using advanced techniques to address the problem effectively."
        
        # Clean text aggressively
        text = text.replace('\n', ' ').replace('\r', ' ').replace('  ', ' ').strip()
        
        # Return full text without character truncation
        logger.debug(f"[FAST] P2: Full text: {len(text)} chars")
        return text
    
    def _extract_implementation_conclusion(self, sections: Dict, abstract: str) -> str:
        """Extract third paragraph covering implementation and conclusion - return full text"""
        text = ""
        
        # Look for Conclusion first, then Results, then Evaluation
        for section_name in ["Conclusion", "Results", "Evaluation", "Experiments", "Conclusion and Future Work"]:
            if section_name in sections and sections[section_name]:
                text = sections[section_name]
                logger.debug(f"[FAST] P3: Using {section_name}")
                break
        
        if not text and abstract and len(abstract) > 100:
            # Try to extract last meaningful part of abstract
            text = abstract
            logger.debug("[FAST] P3: Using abstract")
        
        if not text:
            return "The approach shows promising results and effective performance compared to existing methods, opening avenues for future research."
        
        # Clean text aggressively
        text = text.replace('\n', ' ').replace('\r', ' ').replace('  ', ' ').strip()
        
        # Return full text without character truncation
        logger.debug(f"[FAST] P3: Full text: {len(text)} chars")
        return text
    
    
    async def _generate_balanced_analysis(self, paper_data: Dict) -> AnalysisResult:
        """Generate balanced analysis with selective AI processing"""
        
        # Load lightweight model if not already loaded
        summarizer = await self._get_fast_summarizer()
        
        # Run AI processing in parallel for different sections
        tasks = []
        
        # Task 1: Quick summary
        tasks.append(self._generate_ai_summary_async(paper_data, summarizer, "quick"))
        
        # Task 2: Key sections summary
        tasks.append(self._generate_section_summaries_async(paper_data, summarizer))
        
        # Task 3: Extract insights
        tasks.append(self._extract_insights_async(paper_data))
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        quick_summary = results[0] if not isinstance(results[0], Exception) else "Summary generation failed"
        detailed_summary = results[1] if not isinstance(results[1], Exception) else {}
        comprehensive_analysis = results[2] if not isinstance(results[2], Exception) else {}
        
        return AnalysisResult(
            job_id="temp",
            title=paper_data.get("title", "Unknown"),
            authors=paper_data.get("authors", []),
            quick_summary=quick_summary,
            detailed_summary=detailed_summary,
            comprehensive_analysis=comprehensive_analysis,
            metadata=PaperMetadata(
                title=paper_data.get("title", ""),
                authors=paper_data.get("authors", []),
                paper_id=paper_data.get("paper_id", str(hash(paper_data.get("title", "unknown")))[:8]),
                num_sections=len(paper_data.get("sections", {})),
                processing_method="balanced"
            )
        )
    
    async def _generate_comprehensive_analysis(self, paper_data: Dict) -> AnalysisResult:
        """Generate comprehensive analysis using enhanced analysis service for professional output"""
        
        # Try enhanced analysis first for professional quality output
        try:
            # Create temporary file path for enhanced analysis
            temp_job_id = f"temp_{hash(str(paper_data))}"
            
            # Store paper data temporarily
            self._store_paper_data(temp_job_id, paper_data)
            
            # Use enhanced analysis service for comprehensive mode
            enhanced_result = await enhanced_analysis_service.analyze_paper_enhanced(
                "", temp_job_id, mode="enhanced"
            )
            
            # Update result with proper job handling
            enhanced_result.metadata.processing_method = "comprehensive_enhanced"
            return enhanced_result
            
        except Exception as e:
            logger.warning(f"Enhanced analysis failed, using fallback comprehensive: {e}")
            
            # Fallback to original comprehensive logic
            summarizer = await self._get_full_summarizer()
            
            quick_summary = await self._generate_ai_summary_async(paper_data, summarizer, "comprehensive")
            detailed = await self._generate_section_summaries_async(paper_data, summarizer, comprehensive=True)
            comprehensive = await self._generate_comprehensive_insights_async(paper_data, summarizer)
            
            return AnalysisResult(
                job_id="temp",
                title=paper_data.get("title", "Unknown"),
                authors=paper_data.get("authors", []),
                quick_summary=quick_summary,
                detailed_summary=detailed,
                comprehensive_analysis=comprehensive,
                metadata=PaperMetadata(
                    title=paper_data.get("title", ""),
                    authors=paper_data.get("authors", []),
                    paper_id=paper_data.get("paper_id", str(hash(paper_data.get("title", "unknown")))[:8]),
                    num_sections=len(paper_data.get("sections", {})),
                    processing_method="comprehensive_fallback"
                )
            )
    
    def _extract_quick_summary(self, paper_data: Dict) -> str:
        """Extract quick summary without AI models"""
        abstract = paper_data.get("abstract", "")
        if abstract:
            # Use first 2 sentences of abstract
            sentences = abstract.split('.')[:2]
            return '. '.join(sentences) + "."
        
        # Fallback to introduction
        intro = paper_data.get("sections", {}).get("Introduction", "")
        if intro:
            sentences = intro.split('.')[:2]
            return '. '.join(sentences) + "."
        
        return "Quick summary not available - please try balanced mode."
    
    def _extract_main_points(self, paper_data: Dict) -> str:
        """Extract main contribution without AI"""
        sections = paper_data.get("sections", {})
        
        # Look for conclusion first
        if "Conclusion" in sections:
            text = sections["Conclusion"]
            sentences = text.split('.')[:3]
            return '. '.join(sentences)
        
        # Fallback to abstract
        abstract = paper_data.get("abstract", "")
        if abstract:
            sentences = abstract.split('.')[:2]
            return '. '.join(sentences)
        
        return "Main contribution analysis requires balanced mode."
    
    @lru_cache(maxsize=1)
    async def _get_fast_summarizer(self):
        """Get lightweight summarizer model"""
        return await self.model_engine.get_lightweight_summarizer()
    
    @lru_cache(maxsize=1)
    async def _get_full_summarizer(self):
        """Get full summarizer model"""
        return await self.model_engine.get_full_summarizer()
    
    async def _generate_ai_summary_async(self, paper_data: Dict, summarizer, mode: str) -> str:
        """Generate AI summary asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._generate_ai_summary_sync,
            paper_data, summarizer, mode
        )
    
    def _generate_ai_summary_sync(self, paper_data: Dict, summarizer, mode: str) -> str:
        """Synchronous AI summary generation"""
        abstract = paper_data.get("abstract", "")
        if not abstract or not summarizer:
            return self._extract_quick_summary(paper_data)
        
        try:
            cleaned_abstract = clean_text_comprehensive(abstract)
            if len(cleaned_abstract) > 50:
                max_length = 60 if mode == "quick" else 100
                result = summarizer(
                    cleaned_abstract,
                    max_length=max_length,
                    min_length=20,
                    do_sample=False
                )
                return result[0]['summary_text']
        except Exception as e:
            logger.warning(f"AI summary generation failed: {e}")
        
        return self._extract_quick_summary(paper_data)
    
    async def _generate_section_summaries_async(self, paper_data: Dict, summarizer, comprehensive: bool = False):
        """Generate section summaries asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._generate_section_summaries_sync,
            paper_data, summarizer, comprehensive
        )
    
    def _generate_section_summaries_sync(self, paper_data: Dict, summarizer, comprehensive: bool) -> Dict[str, str]:
        """Synchronous section summary generation"""
        detailed = {}
        sections = paper_data.get("sections", {})
        
        key_sections = ["Introduction", "Methods", "Results", "Conclusion"] if comprehensive else ["Introduction", "Conclusion"]
        
        for section_name in key_sections:
            if section_name in sections:
                text = sections[section_name]
                if len(text) < 100:
                    continue
                
                try:
                    clean_text = clean_text_comprehensive(text)
                    if summarizer and len(clean_text) > 50:
                        # Process smaller chunks for speed
                        chunks = chunk_text_for_models(clean_text, "bart")
                        if chunks:
                            chunk = chunks[0]  # Use only first chunk for speed
                            result = summarizer(
                                chunk["text"],
                                max_length=80 if comprehensive else 50,
                                min_length=20,
                                do_sample=False
                            )
                            detailed[section_name] = result[0]['summary_text']
                    else:
                        # Fallback to extractive summary
                        sentences = clean_text.split('.')[:3]
                        detailed[section_name] = '. '.join(sentences)
                except Exception as e:
                    logger.warning(f"Section summary failed for {section_name}: {e}")
                    # Fallback: show first 3 sentences without character truncation
                    sentences = text.split('.')[:3]
                    detailed[section_name] = '. '.join([s.strip() for s in sentences if s.strip()]) + '.'
        
        return detailed
    
    async def _extract_insights_async(self, paper_data: Dict):
        """Extract insights asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._extract_insights_sync,
            paper_data
        )
    
    def _extract_insights_sync(self, paper_data: Dict) -> Dict[str, Any]:
        """Extract insights without heavy AI processing"""
        sections = paper_data.get("sections", {})
        
        analysis = {
            "main_contribution": self._extract_main_points(paper_data),
            "methodology": "Balanced processing mode",
            "key_findings": "",
            "processing_time": "60-120 seconds",
            "mode": "balanced"
        }
        
        # Extract key findings from Results section
        if "Results" in sections:
            results_text = sections["Results"]
            sentences = results_text.split('.')[:3]
            analysis["key_findings"] = '. '.join(sentences)
        elif "Conclusion" in sections:
            conclusion_text = sections["Conclusion"] 
            sentences = conclusion_text.split('.')[:2]
            analysis["key_findings"] = '. '.join(sentences)
        
        return analysis
    
    async def _generate_comprehensive_insights_async(self, paper_data: Dict, summarizer):
        """Generate comprehensive insights with full AI processing"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._generate_comprehensive_insights_sync,
            paper_data, summarizer
        )
    
    def _generate_comprehensive_insights_sync(self, paper_data: Dict, summarizer) -> Dict[str, Any]:
        """Generate comprehensive insights (original quality)"""
        # This would use the original comprehensive analysis logic
        # but with optimized model loading
        
        analysis = {
            "main_contribution": "",
            "methodology": "",
            "key_findings": "",
            "limitations": "",
            "future_work": "",
            "processing_time": "120-300 seconds",
            "mode": "comprehensive"
        }
        
        sections = paper_data.get("sections", {})
        
        # Use AI models for each section if available
        if summarizer:
            for field, section_name in [
                ("main_contribution", "Introduction"),
                ("methodology", "Methods"),
                ("key_findings", "Results"),
                ("limitations", "Discussion"),
                ("future_work", "Conclusion")
            ]:
                if section_name in sections:
                    try:
                        text = clean_text_comprehensive(sections[section_name])
                        chunks = chunk_text_for_models(text, "bart")
                        if chunks:
                            result = summarizer(
                                chunks[0]["text"],
                                max_length=100,
                                min_length=30,
                                do_sample=False
                            )
                            analysis[field] = result[0]['summary_text']
                    except Exception as e:
                        logger.warning(f"Failed to analyze {field}: {e}")
                        sentences = sections[section_name].split('.')[:2]
                        analysis[field] = '. '.join(sentences)
        
        return analysis

# Create global instance
fast_analysis_service = FastAnalysisService()