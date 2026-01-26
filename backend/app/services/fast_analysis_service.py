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
    
    async def analyze_paper_fast(self, file_path: str, job_id: str, mode: str = "balanced") -> AnalysisResult:
        """
        Fast paper analysis with configurable processing modes
        
        Args:
            file_path: Path to PDF file
            job_id: Unique job identifier
            mode: Processing mode - "fast", "balanced", or "comprehensive"
        """
        logger.info(f"Starting fast analysis for job {job_id} in {mode} mode")
        
        try:
            # Step 1: Quick GROBID extraction with optimized timeout
            paper_data = await self._extract_structure_fast(file_path, mode)
            
            # Store paper data for chat indexing
            self._store_paper_data(job_id, paper_data)
            
            # Step 2: Generate summaries based on mode
            if mode == "fast":
                result = await self._generate_fast_analysis(paper_data)
            elif mode == "balanced":
                result = await self._generate_balanced_analysis(paper_data)
            else:  # comprehensive
                result = await self._generate_comprehensive_analysis(paper_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Fast analysis failed for job {job_id}: {str(e)}")
            raise
    
    async def _extract_structure_fast(self, file_path: str, mode: str) -> Dict[str, Any]:
        """Extract structure with optimized GROBID settings"""
        
        # Set timeout based on mode and file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        
        if mode == "fast":
            timeout = min(60, max(30, int(file_size * 10)))  # 30-60s based on size
        elif mode == "balanced":
            timeout = min(120, max(45, int(file_size * 15)))  # 45-120s
        else:
            timeout = min(300, max(90, int(file_size * 30)))  # 90-300s
        
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
        """Generate fast analysis with minimal AI processing"""
        
        # Use lightweight summarization or extractive methods
        quick_summary = self._extract_quick_summary(paper_data)
        
        # Generate minimal detailed analysis
        detailed = {}
        sections = paper_data.get("sections", {})
        key_sections = ["Introduction", "Conclusion", "Abstract"]
        
        for section_name in key_sections:
            if section_name in sections:
                text = sections[section_name]
                if len(text) > 100:
                    # Use extractive summary (first 2 sentences + last sentence)
                    sentences = text.split('.')[:3]
                    detailed[section_name] = '. '.join(sentences)[:300] + "..."
        
        return AnalysisResult(
            job_id="temp",
            title=paper_data.get("title", "Unknown"),
            authors=paper_data.get("authors", []),
            quick_summary=quick_summary,
            detailed_summary=detailed,
            comprehensive_analysis={
                "main_contribution": self._extract_main_points(paper_data),
                "methodology": "Fast processing mode - detailed analysis available in balanced mode",
                "key_findings": "Use balanced mode for AI-powered analysis",
                "processing_time": "< 60 seconds",
                "mode": "fast"
            },
            metadata=PaperMetadata(
                title=paper_data.get("title", ""),
                authors=paper_data.get("authors", []),
                paper_id=paper_data.get("paper_id", str(hash(paper_data.get("title", "unknown")))[:8]),
                num_sections=len(paper_data.get("sections", {})),
                processing_method="fast"
            )
        )
    
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
                    sentences = text.split('.')[:2]
                    detailed[section_name] = '. '.join(sentences)[:200] + "..."
        
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