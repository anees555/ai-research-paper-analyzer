#!/usr/bin/env python3
"""
Demo Summary Generator
Generate summaries without requiring GROBID (for demonstration)
"""

import os
import json
import PyPDF2
from typing import Dict, List, Any, Optional

class DemoSummaryGenerator:
    """
    Demo summary generator using simple PDF text extraction
    """
    
    def __init__(self, output_dir: str = "outputs/demo_summaries"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print("🚀 Demo Summary Generator initialized")
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Simple PDF text extraction using PyPDF2
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from all pages
                full_text = ""
                for page in pdf_reader.pages:
                    full_text += page.extract_text() + "\n"
                
                # Simple parsing - split into paragraphs
                paragraphs = [p.strip() for p in full_text.split('\n\n') if len(p.strip()) > 50]
                
                # Try to identify title (first substantial line)
                title = "Unknown Title"
                lines = full_text.split('\n')
                for line in lines:
                    if len(line.strip()) > 10 and len(line.strip()) < 200:
                        title = line.strip()
                        break
                
                return {
                    "title": title,
                    "full_text": full_text,
                    "paragraphs": paragraphs,
                    "page_count": len(pdf_reader.pages),
                    "word_count": len(full_text.split())
                }
                
        except Exception as e:
            print(f"❌ Error extracting from {pdf_path}: {e}")
            return {
                "title": f"Error: {os.path.basename(pdf_path)}",
                "full_text": "",
                "paragraphs": [],
                "page_count": 0,
                "word_count": 0,
                "error": str(e)
            }
    
    def generate_quick_summary(self, paper_data: Dict[str, Any]) -> str:
        """
        Generate quick summary from first few paragraphs
        """
        if "error" in paper_data:
            return f"❌ Could not generate summary: {paper_data['error']}"
        
        paragraphs = paper_data.get("paragraphs", [])
        
        if not paragraphs:
            return "❌ No substantial content found for summary"
        
        # Use first 2-3 paragraphs as quick summary
        summary_paragraphs = paragraphs[:min(3, len(paragraphs))]
        quick_summary = " ".join(summary_paragraphs)
        
        # Truncate if too long
        if len(quick_summary) > 500:
            quick_summary = quick_summary[:500] + "..."
        
        return quick_summary
    
    def generate_detailed_summary(self, paper_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate section-based detailed summary
        """
        if "error" in paper_data:
            return {"error": f"Could not process: {paper_data['error']}"}
        
        paragraphs = paper_data.get("paragraphs", [])
        
        if len(paragraphs) < 4:
            return {"summary": "Document too short for detailed analysis"}
        
        # Simple heuristic: divide paragraphs into sections
        total_paragraphs = len(paragraphs)
        
        detailed_summary = {
            "introduction": " ".join(paragraphs[:total_paragraphs//4]),
            "main_content": " ".join(paragraphs[total_paragraphs//4:3*total_paragraphs//4]),
            "conclusion": " ".join(paragraphs[3*total_paragraphs//4:])
        }
        
        # Truncate each section
        for key in detailed_summary:
            if len(detailed_summary[key]) > 300:
                detailed_summary[key] = detailed_summary[key][:300] + "..."
        
        return detailed_summary
    
    def generate_comprehensive_analysis(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive analysis
        """
        if "error" in paper_data:
            return {"error": f"Could not analyze: {paper_data['error']}"}
        
        analysis = {
            "document_stats": {
                "pages": paper_data.get("page_count", 0),
                "words": paper_data.get("word_count", 0),
                "paragraphs": len(paper_data.get("paragraphs", []))
            },
            "content_overview": "",
            "key_points": [],
            "estimated_reading_time": f"{paper_data.get('word_count', 0) // 200} minutes"
        }
        
        paragraphs = paper_data.get("paragraphs", [])
        
        if paragraphs:
            # Content overview from first paragraph
            analysis["content_overview"] = paragraphs[0][:400] + "..."
            
            # Extract key points (first sentence of each paragraph)
            key_points = []
            for para in paragraphs[:10]:  # First 10 paragraphs
                sentences = para.split('. ')
                if sentences and len(sentences[0]) > 20:
                    key_points.append(sentences[0] + ".")
            
            analysis["key_points"] = key_points
        
        return analysis
    
    def process_single_paper(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a single PDF and generate all summaries
        """
        print(f"📄 Processing: {os.path.basename(pdf_path)}")
        
        # Extract text
        paper_data = self.extract_text_from_pdf(pdf_path)
        
        # Generate summaries
        summaries = {
            "paper_id": os.path.basename(pdf_path).replace('.pdf', ''),
            "title": paper_data["title"],
            "file_path": pdf_path,
            "quick_summary": self.generate_quick_summary(paper_data),
            "detailed_summary": self.generate_detailed_summary(paper_data),
            "comprehensive_analysis": self.generate_comprehensive_analysis(paper_data),
            "processing_stats": {
                "method": "PyPDF2 text extraction",
                "pages_processed": paper_data.get("page_count", 0),
                "words_extracted": paper_data.get("word_count", 0)
            }
        }
        
        # Save results
        output_file = os.path.join(self.output_dir, f"{summaries['paper_id']}_demo_summary.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Demo summary saved: {output_file}")
        return summaries
    
    def process_directory(self, pdf_directory: str) -> List[Dict[str, Any]]:
        """
        Process all PDFs in directory
        """
        print(f"📁 Processing directory: {pdf_directory}")
        
        if not os.path.exists(pdf_directory):
            print(f"❌ Directory not found: {pdf_directory}")
            return []
        
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        
        if not pdf_files:
            print("❌ No PDF files found")
            return []
        
        print(f"📄 Found {len(pdf_files)} PDF files")
        
        all_summaries = []
        
        for i, pdf_file in enumerate(pdf_files):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            print(f"\n--- Processing {i+1}/{len(pdf_files)} ---")
            
            summary = self.process_single_paper(pdf_path)
            all_summaries.append(summary)
        
        # Generate report
        self.generate_report(all_summaries)
        
        return all_summaries
    
    def generate_report(self, summaries: List[Dict[str, Any]]):
        """
        Generate processing report
        """
        total_papers = len(summaries)
        total_pages = sum(s.get("processing_stats", {}).get("pages_processed", 0) for s in summaries)
        total_words = sum(s.get("processing_stats", {}).get("words_extracted", 0) for s in summaries)
        
        report = f"""📊 DEMO SUMMARY GENERATION REPORT
{"=" * 50}

📄 Papers Processed: {total_papers}
📖 Total Pages: {total_pages:,}
📝 Total Words: {total_words:,}
📂 Output Directory: {self.output_dir}

🔍 PROCESSED PAPERS:
"""
        
        for i, summary in enumerate(summaries[:10]):  # Show first 10
            title = summary.get("title", "Unknown")[:50]
            pages = summary.get("processing_stats", {}).get("pages_processed", 0)
            report += f"  {i+1}. {title} ({pages} pages)\n"
        
        if len(summaries) > 10:
            report += f"  ... and {len(summaries) - 10} more papers\n"
        
        report += f"""
📈 SUMMARY STATISTICS:
  • Average pages per paper: {total_pages/total_papers:.1f}
  • Average words per paper: {total_words/total_papers:,.0f}
  • Estimated reading time saved: {(total_words * 0.8) // 200:.0f} minutes

🎯 NEXT STEPS:
  1. Review generated summaries in: {self.output_dir}
  2. Start GROBID server for better structure extraction
  3. Implement actual AI models for improved summaries
  4. Set up web interface for user access

⚠️  NOTE: This is a DEMO using simple text extraction.
    For production quality, use GROBID + AI models.
"""
        
        print(f"\n{report}")
        
        # Save report
        report_file = os.path.join(self.output_dir, "demo_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📋 Report saved: {report_file}")

def main():
    """
    Demo summary generation
    """
    print("🎯 Demo Research Paper Summary Generator")
    print("(Works without GROBID - for demonstration)")
    print("=" * 60)
    
    generator = DemoSummaryGenerator()
    
    # Try ArXiv PDFs first
    pdf_dir = "datasets/arxiv/pdfs"
    
    if os.path.exists(pdf_dir):
        print(f"📁 Found ArXiv PDFs: {pdf_dir}")
        generator.process_directory(pdf_dir)
    else:
        # Try papers directory
        papers_dir = "papers"
        if os.path.exists(papers_dir):
            print(f"📁 Found papers directory: {papers_dir}")
            generator.process_directory(papers_dir)
        else:
            print("❌ No PDF directories found")
            print("Please ensure PDFs are in:")
            print("  - datasets/arxiv/pdfs/ (from ArXiv collection)")
            print("  - papers/ (manual uploads)")

if __name__ == "__main__":
    main()
