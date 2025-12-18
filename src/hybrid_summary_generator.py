#!/usr/bin/env python3
"""
Hybrid AI Summary Generator
Uses GROBID when possible, falls back to PyPDF2 + AI models when GROBID times out
"""

import os
import json
import warnings
from typing import Dict, List, Any, Optional
import time

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from scripts.parse_pdf_optimized import OptimizedGROBIDProcessor
    GROBID_AVAILABLE = True
except ImportError:
    GROBID_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from scripts.preprocess_text import clean_text_comprehensive, chunk_text_for_models

class HybridSummaryGenerator:
    """
    Hybrid summary generator with multiple fallback strategies
    """
    
    def __init__(self, output_dir: str = "outputs/hybrid_summaries"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.grobid_processor = None
        self.bart_summarizer = None
        self.longformer_summarizer = None
        self.processing_stats = {
            "grobid_success": 0,
            "grobid_timeout": 0,
            "pypdf2_fallback": 0,
            "total_processed": 0
        }
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all available components"""
        
        # Initialize GROBID processor

        if GROBID_AVAILABLE:
            try:
                self.grobid_processor = OptimizedGROBIDProcessor()
                if self.grobid_processor.check_grobid_health():
                    print(" GROBID processor initialized and healthy")
                else:
                    print(" GROBID processor initialized but server not healthy")
                    self.grobid_processor = None
            except Exception as e:
                print(f" GROBID processor initialization failed: {e}")
                self.grobid_processor = None
        else:
            print(" GROBID processor not available")
        
        # Initialize AI models
        if TRANSFORMERS_AVAILABLE:
            try:
                print(" Loading BART model for quick summaries...")
                self.bart_summarizer = pipeline(
                    "summarization", 
                    model="facebook/bart-large-cnn",
                    device=-1
                )
                print(" BART model loaded successfully")
                
                print(" Loading Longformer model for deep summaries...")
                self.longformer_summarizer = pipeline(
                    "summarization",
                    model="allenai/led-large-16384-arxiv",  # Longformer for long documents
                    device=-1
                )
                print(" Longformer model loaded successfully")
                
            except Exception as e:
                print(f" AI model loading failed: {e}")
                self.bart_summarizer = None
                self.longformer_summarizer = None
        else:
            print(" Transformers not available")
    
    def extract_with_grobid(self, pdf_path: str, timeout_limit: int = 300) -> Dict[str, Any]:
        """
        Try GROBID extraction with timeout limit
        """
        if not self.grobid_processor:
            raise Exception("GROBID processor not available")
        
        print(f" Attempting GROBID extraction (max {timeout_limit}s)...")
        
        try:
            # Override timeout strategies for faster processing
            self.grobid_processor.timeout_strategies = {
                'quick': min(60, timeout_limit),
                'normal': min(180, timeout_limit), 
                'extended': min(300, timeout_limit),
                'maximum': timeout_limit
            }
            
            result = self.grobid_processor.parse_pdf_optimized(pdf_path, "output")
            self.processing_stats["grobid_success"] += 1
            return result
            
        except Exception as e:
            if "timeout" in str(e).lower():
                print(f" GROBID timed out after {timeout_limit}s")
                self.processing_stats["grobid_timeout"] += 1
            else:
                print(f" GROBID failed: {e}")
            raise
    
    def extract_with_pypdf2(self, pdf_path: str) -> Dict[str, Any]:
        """
        Fallback extraction using PyPDF2
        """
        if not PYPDF2_AVAILABLE:
            raise Exception("PyPDF2 not available")
        
        print(" Using PyPDF2 fallback extraction...")
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from all pages
                full_text = ""
                for page in pdf_reader.pages:
                    full_text += page.extract_text() + "\n"
                
                # Clean the text
                full_text = clean_text_comprehensive(full_text)
                
                # Simple parsing
                paragraphs = [p.strip() for p in full_text.split('\n\n') if len(p.strip()) > 50]
                
                # Try to identify sections heuristically
                sections = {}
                current_section = "Content"
                current_content = []
                
                section_keywords = [
                    "abstract", "introduction", "methodology", "methods", 
                    "results", "discussion", "conclusion", "references"
                ]
                
                for paragraph in paragraphs[:20]:  # Check first 20 paragraphs
                    paragraph_lower = paragraph.lower()
                    
                    # Check if this paragraph is a section header
                    is_section_header = any(
                        keyword in paragraph_lower and len(paragraph) < 100
                        for keyword in section_keywords
                    )
                    
                    if is_section_header:
                        # Save previous section
                        if current_content:
                            sections[current_section] = " ".join(current_content)
                        
                        # Start new section
                        current_section = paragraph.strip()
                        current_content = []
                    else:
                        current_content.append(paragraph)
                
                # Save last section
                if current_content:
                    sections[current_section] = " ".join(current_content)
                
                # Try to extract title (first substantial line)
                title = "Unknown Title"
                lines = full_text.split('\n')
                for line in lines:
                    if 10 < len(line.strip()) < 200:
                        title = line.strip()
                        break
                
                # Try to extract abstract
                abstract = ""
                if "abstract" in full_text.lower():
                    # Simple heuristic to find abstract
                    for key, content in sections.items():
                        if "abstract" in key.lower():
                            abstract = content[:500]  # First 500 chars
                            break
                
                self.processing_stats["pypdf2_fallback"] += 1
                
                return {
                    "title": title,
                    "authors": [],  # Can't extract reliably from PyPDF2
                    "abstract": abstract,
                    "sections": sections,
                    "processing_method": "PyPDF2_fallback",
                    "processing_stats": {
                        "total_pages": len(pdf_reader.pages),
                        "total_characters": len(full_text),
                        "sections_found": len(sections)
                    }
                }
                
        except Exception as e:
            print(f" PyPDF2 extraction failed: {e}")
            raise
    
    def generate_ai_summary(self, paper_data: Dict[str, Any]) -> str:
        """
        Generate quick AI summary using BART
        """
        if not self.bart_summarizer:
            # Fallback to simple summary
            if paper_data.get("abstract"):
                return f" Summary: {paper_data['abstract'][:200]}..."
            return " No summary available"

        try:
            # Use abstract if available, otherwise use first section
            input_text = ""
            if paper_data.get("abstract"):
                input_text = paper_data["abstract"]
            elif paper_data.get("sections"):
                first_section = list(paper_data["sections"].values())[0]
                input_text = first_section[:1000]  # First 1000 chars
            
            if not input_text or len(input_text) < 50:
                return " Insufficient content for AI summary"
            
            # Clean input text
            input_text = clean_text_comprehensive(input_text)
            
            # Generate AI summary
            result = self.bart_summarizer(
                input_text,
                max_length=100,
                min_length=30,
                do_sample=False
            )
            
            return f" Quick Summary: {result[0]['summary_text']}"
            
        except Exception as e:
            print(f" AI summarization failed: {e}")
            # Fallback
            if paper_data.get("abstract"):
                sentences = paper_data["abstract"].split(". ")
                return f" Summary: {sentences[0]}." if sentences else " Summary unavailable"
            return " Summary generation failed"

    def generate_deep_summary(self, paper_data: Dict[str, Any]) -> str:
        """
        Generate comprehensive 2-3 paragraph summary using Longformer
        """
        if not self.longformer_summarizer:
            # Fallback to extended BART summary
            if self.bart_summarizer and paper_data.get("sections"):
                try:
                    # Use more content for fallback
                    sections = paper_data.get("sections", {})
                    combined_text = ""
                    
                    # Combine all sections up to 2000 chars
                    for section_name, content in sections.items():
                        if len(combined_text) < 2000:
                            combined_text += f"{content} "
                    
                    combined_text = clean_text_comprehensive(combined_text[:2000])
                    
                    result = self.bart_summarizer(
                        combined_text,
                        max_length=200,
                        min_length=100,
                        do_sample=False
                    )
                    
                    return f" Extended Summary: {result[0]['summary_text']}"
                    
                except Exception as e:
                    print(f" Extended summary failed: {e}")
            
            return " Deep summary not available"

        try:
            # Prepare comprehensive input text for Longformer
            sections = paper_data.get("sections", {})
            if not sections:
                return " No content available for deep summary"
            
            # Build comprehensive text from all sections
            comprehensive_text = ""
            
            # Add abstract if available
            if paper_data.get("abstract"):
                comprehensive_text += f"Abstract: {paper_data['abstract']}\n\n"
            
            # Add all sections with context
            for section_name, content in sections.items():
                if section_name.lower() != "references":  # Skip references
                    comprehensive_text += f"{section_name}: {content}\n\n"
            
            # Limit to Longformer's capacity (approximately 16k tokens = ~64k chars)
            comprehensive_text = comprehensive_text[:60000]
            comprehensive_text = clean_text_comprehensive(comprehensive_text)
            
            if len(comprehensive_text) < 200:
                return " Insufficient content for deep summary"
            
            print(" Generating comprehensive summary with Longformer...")
            
            # Generate comprehensive summary (2-3 paragraphs)
            result = self.longformer_summarizer(
                comprehensive_text,
                max_length=400,  # ~2-3 paragraphs
                min_length=200,
                do_sample=False
            )
            
            return f" Deep Analysis: {result[0]['summary_text']}"
            
        except Exception as e:
            print(f" Deep summary generation failed: {e}")
            # Fallback to extended BART summary
            return " Deep summary failed"

    def process_single_paper(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a single PDF with hybrid approach
        """
        pdf_name = os.path.basename(pdf_path)
        print(f"\n{'='*60}")
        print(f" Processing: {pdf_name}")
        print(f"{'='*60}")
        
        self.processing_stats["total_processed"] += 1
        paper_data = None
        processing_method = "unknown"
        
        # Strategy 1: Try GROBID first (with shorter timeout)
        if self.grobid_processor:
            try:
                paper_data = self.extract_with_grobid(pdf_path, timeout_limit=120)  # 2 minute limit
                processing_method = "GROBID_optimized"
                print(" GROBID extraction successful!")
                
            except Exception as e:
                print(f" GROBID failed, trying fallback: {e}")
        
        # Strategy 2: Fallback to PyPDF2 if GROBID failed
        if paper_data is None and PYPDF2_AVAILABLE:
            try:
                paper_data = self.extract_with_pypdf2(pdf_path)
                processing_method = "PyPDF2_fallback"
                print(" PyPDF2 fallback successful!")
                
            except Exception as e:
                print(f" PyPDF2 fallback also failed: {e}")
        
        # If all extraction failed
        if paper_data is None:
            return {
                "paper_id": pdf_name.replace('.pdf', ''),
                "error": "All extraction methods failed",
                "status": "failed"
            }
        
        # Generate summaries
        print(" Generating multi-level AI summaries...")
        
        summaries = {
            "paper_id": pdf_name.replace('.pdf', ''),
            "title": paper_data.get("title", "Unknown Title"),
            "authors": paper_data.get("authors", []),
            "processing_method": processing_method,
            "quick_summary": self.generate_ai_summary(paper_data),
            "deep_summary": self.generate_deep_summary(paper_data),
            "sections_found": list(paper_data.get("sections", {}).keys()),
            "original_abstract": paper_data.get("abstract", ""),
            "extraction_stats": paper_data.get("processing_stats", {}),
            "ai_enhanced": self.bart_summarizer is not None,
            "deep_analysis_available": self.longformer_summarizer is not None
        }
        
        # Save results
        output_file = os.path.join(self.output_dir, f"{summaries['paper_id']}_hybrid_summary.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
        
        print(f" Hybrid summary saved: {output_file}")
        return summaries
    
    def process_directory(self, pdf_directory: str, max_papers: int = 5) -> List[Dict[str, Any]]:
        """
        Process directory with hybrid approach
        """
        print(" Hybrid AI Summary Generator")
        print("Tries GROBID first, falls back to PyPDF2 + AI")
        print("=" * 60)
        
        if not os.path.exists(pdf_directory):
            print(f" Directory not found: {pdf_directory}")
            return []
        
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        if not pdf_files:
            print(" No PDF files found")
            return []
        
        pdf_files = pdf_files[:max_papers]
        print(f" Processing {len(pdf_files)} PDFs with hybrid approach")
        
        all_summaries = []
        start_time = time.time()
        
        for i, pdf_file in enumerate(pdf_files):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            
            try:
                summary = self.process_single_paper(pdf_path)
                all_summaries.append(summary)
                
            except Exception as e:
                print(f" Failed to process {pdf_file}: {e}")
                all_summaries.append({
                    "paper_id": pdf_file.replace('.pdf', ''),
                    "error": str(e),
                    "status": "failed"
                })
        
        # Generate final report
        processing_time = time.time() - start_time
        self.generate_hybrid_report(all_summaries, processing_time)
        
        return all_summaries
    
    def generate_hybrid_report(self, summaries: List[Dict[str, Any]], processing_time: float):
        """
        Generate comprehensive hybrid processing report
        """
        successful = [s for s in summaries if "error" not in s]
        failed = [s for s in summaries if "error" in s]
        
        grobid_success = sum(1 for s in successful if s.get("processing_method") == "GROBID_optimized")
        pypdf2_success = sum(1 for s in successful if s.get("processing_method") == "PyPDF2_fallback")
        
        report = f""" HYBRID AI SUMMARY GENERATION REPORT
{"=" * 70}

 PROCESSING SUMMARY:
  • Total Processing Time: {processing_time:.1f} seconds
  • Average Time per Paper: {processing_time/len(summaries):.1f}s
  • Total Papers Attempted: {len(summaries)}

 SUCCESS BREAKDOWN:
  •  Total Successful: {len(successful)}
  •  GROBID Success: {grobid_success}
  •  PyPDF2 Fallback: {pypdf2_success} 
  •  Complete Failures: {len(failed)}

 AI ENHANCEMENT:
  • BART Model Available: {' Yes' if self.bart_summarizer else ' No'}
  • Longformer Model Available: {' Yes' if self.longformer_summarizer else ' No'}
  • Quick Summaries Generated: {sum(1 for s in successful if s.get('ai_enhanced', False))}
  • Deep Analyses Generated: {sum(1 for s in successful if s.get('deep_analysis_available', False))}

 OUTPUT DIRECTORY: {self.output_dir}

 SUCCESSFULLY PROCESSED:
"""
        
        for i, summary in enumerate(successful, 1):
            title = summary.get("title", "Unknown")[:40]
            method = summary.get("processing_method", "Unknown")
            sections = len(summary.get("sections_found", []))
            report += f"  {i}. {title}... ({method}, {sections} sections)\n"
        
        if failed:
            report += f"\n FAILED PAPERS:\n"
            for fail in failed:
                report += f"  • {fail['paper_id']}: {fail.get('error', 'Unknown error')}\n"
        
        report += f"""
 OPTIMIZATION RESULTS:
  • GROBID Timeout Strategy: Reduced to 2 minutes
  • Fallback Success Rate: {pypdf2_success}/{len(summaries)} papers
  • Overall Success Rate: {len(successful)}/{len(summaries)} ({len(successful)/len(summaries)*100:.1f}%)

 NEXT STEPS:
  1. Review hybrid summaries in: {self.output_dir}
  2. Fine-tune timeout settings based on results
  3. Consider GROBID server optimization
  4. Deploy web interface with hybrid processing

 CONCLUSION:
  The hybrid approach successfully processes papers even when GROBID times out.
  This provides a robust solution for the AI research paper analyzer.
"""
        
        print(f"\n{report}")
        
        # Save report
        report_file = os.path.join(self.output_dir, "hybrid_processing_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f" Detailed report saved: {report_file}")

def main():
    """
    Run hybrid summary generation
    """
    generator = HybridSummaryGenerator()
    
    pdf_dir = "datasets/arxiv/pdfs"
    if os.path.exists(pdf_dir):
        summaries = generator.process_directory(pdf_dir, max_papers=3)
        print(f"\n Hybrid processing complete! Processed {len(summaries)} papers")
    else:
        print(f" PDF directory not found: {pdf_dir}")

if __name__ == "__main__":
    main()
