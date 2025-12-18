#!/usr/bin/env python3
"""
Summary Generation Module
Generate multi-level summaries from processed research papers
"""

import os
import json
from typing import Dict, List, Any, Optional
from scripts.parse_pdf import parse_pdf_with_grobid
from scripts.preprocess_text import clean_text_comprehensive, chunk_text_for_models

# For now, we'll use mock implementations since models aren't installed yet
# Later, these will be replaced with actual Hugging Face transformers

class SummaryGenerator:
    """
    Generate multi-level summaries from research papers
    """
    
    def __init__(self):
        self.models_loaded = False
        # In production, load actual models here:
        # self.bart_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        # self.longformer_model = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-large-16384-arxiv")
        print("🤖 Summary Generator initialized (using mock models for now)")
    
    def generate_quick_summary(self, paper_data: Dict[str, Any]) -> str:
        """
        Generate a quick 1-2 sentence summary using BART
        """
        if not paper_data.get("abstract"):
            return "No abstract available for quick summary."
        
        # For now, use the first sentence of abstract as mock summary
        # In production, this would use BART model
        abstract = clean_text_comprehensive(paper_data["abstract"])
        sentences = abstract.split(". ")
        
        if len(sentences) >= 2:
            mock_summary = f"{sentences[0]}. {sentences[1]}."
        else:
            mock_summary = sentences[0] + "."
        
        return f"🔸 Quick Summary: {mock_summary}"
    
    def generate_detailed_summary(self, paper_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate detailed section-wise summaries using BART
        """
        detailed_summary = {}
        
        # Key sections to summarize
        key_sections = ["Introduction", "Methods", "Results", "Conclusion"]
        
        for section_name in key_sections:
            if section_name in paper_data.get("sections", {}):
                section_text = paper_data["sections"][section_name]
                
                if len(section_text.strip()) > 100:
                    # Mock detailed summary (in production, use BART)
                    cleaned_text = clean_text_comprehensive(section_text)
                    sentences = cleaned_text.split(". ")
                    
                    # Take first 2-3 sentences as mock summary
                    summary_sentences = sentences[:min(3, len(sentences))]
                    section_summary = ". ".join(summary_sentences) + "."
                    
                    detailed_summary[section_name] = section_summary
        
        return detailed_summary
    
    def generate_comprehensive_analysis(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive analysis using Longformer
        """
        analysis = {
            "main_contribution": "",
            "methodology": "",
            "key_findings": "",
            "limitations": "",
            "future_work": ""
        }
        
        sections = paper_data.get("sections", {})
        
        # Extract main contribution (mock implementation)
        if "Introduction" in sections:
            intro_text = clean_text_comprehensive(sections["Introduction"])
            # Mock: Take last paragraph of introduction as contribution
            paragraphs = intro_text.split("\n\n")
            if paragraphs:
                analysis["main_contribution"] = paragraphs[-1][:300] + "..."
        
        # Extract methodology
        if "Methods" in sections or "Methodology" in sections:
            method_key = "Methods" if "Methods" in sections else "Methodology"
            method_text = clean_text_comprehensive(sections[method_key])
            analysis["methodology"] = method_text[:300] + "..."
        
        # Extract key findings
        if "Results" in sections:
            results_text = clean_text_comprehensive(sections["Results"])
            analysis["key_findings"] = results_text[:300] + "..."
        
        # Extract limitations and future work
        if "Conclusion" in sections:
            conclusion_text = clean_text_comprehensive(sections["Conclusion"])
            # Mock: Split conclusion into limitations and future work
            sentences = conclusion_text.split(". ")
            if len(sentences) > 2:
                analysis["limitations"] = sentences[-2] + "."
                analysis["future_work"] = sentences[-1] + "."
        
        return analysis
    
    def generate_all_summaries(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate all types of summaries for a paper
        """
        print(f"🔄 Generating summaries for: {paper_data.get('title', 'Unknown')[:50]}...")
        
        summaries = {
            "paper_id": paper_data.get("paper_id", "unknown"),
            "title": paper_data.get("title", ""),
            "authors": paper_data.get("authors", []),
            "quick_summary": self.generate_quick_summary(paper_data),
            "detailed_summary": self.generate_detailed_summary(paper_data),
            "comprehensive_analysis": self.generate_comprehensive_analysis(paper_data),
            "original_abstract": paper_data.get("abstract", ""),
            "processing_time": "< 1 second (mock)",
            "summary_stats": {
                "sections_processed": len(paper_data.get("sections", {})),
                "total_words": sum(len(text.split()) for text in paper_data.get("sections", {}).values()),
                "summary_reduction": "~80% reduction (estimated)"
            }
        }
        
        return summaries

class PaperProcessor:
    """
    Complete paper processing workflow: PDF → Summaries
    """
    
    def __init__(self, output_dir: str = "outputs/summaries"):
        self.output_dir = output_dir
        self.summary_generator = SummaryGenerator()
        os.makedirs(output_dir, exist_ok=True)
    
    def process_single_paper(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a single PDF and generate all summaries
        """
        print(f"📄 Processing paper: {os.path.basename(pdf_path)}")
        
        try:
            # Step 1: Extract structured data with GROBID
            print("🔄 Extracting text with GROBID...")
            paper_data = parse_pdf_with_grobid(pdf_path, "output")
            paper_data["paper_id"] = os.path.basename(pdf_path).replace('.pdf', '')
            paper_data["file_path"] = pdf_path
            
            # Step 2: Generate summaries
            print("🔄 Generating summaries...")
            summaries = self.summary_generator.generate_all_summaries(paper_data)
            
            # Step 3: Save results
            output_file = os.path.join(self.output_dir, f"{summaries['paper_id']}_summaries.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summaries, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Summaries saved to: {output_file}")
            return summaries
            
        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {e}")
            return {
                "paper_id": os.path.basename(pdf_path).replace('.pdf', ''),
                "error": str(e),
                "status": "failed"
            }
    
    def process_directory(self, pdf_directory: str) -> List[Dict[str, Any]]:
        """
        Process all PDFs in a directory
        """
        print(f"📁 Processing PDFs from: {pdf_directory}")
        
        if not os.path.exists(pdf_directory):
            print(f"❌ Directory not found: {pdf_directory}")
            return []
        
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        
        if not pdf_files:
            print("❌ No PDF files found in directory")
            return []
        
        print(f"📄 Found {len(pdf_files)} PDF files")
        
        all_summaries = []
        
        for i, pdf_file in enumerate(pdf_files):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            print(f"\n--- Processing {i+1}/{len(pdf_files)} ---")
            
            summaries = self.process_single_paper(pdf_path)
            all_summaries.append(summaries)
        
        # Generate batch report
        self.generate_batch_report(all_summaries)
        
        return all_summaries
    
    def generate_batch_report(self, all_summaries: List[Dict[str, Any]]):
        """
        Generate a report for batch processing
        """
        successful = [s for s in all_summaries if "error" not in s]
        failed = [s for s in all_summaries if "error" in s]
        
        report = f"""📊 BATCH PROCESSING REPORT
{"=" * 50}

📄 Total Papers: {len(all_summaries)}
✅ Successfully Processed: {len(successful)}
❌ Failed: {len(failed)}
📂 Output Directory: {self.output_dir}

📋 SUCCESSFUL PAPERS:
"""
        
        for summary in successful[:10]:  # Show first 10
            title = summary.get("title", "Unknown")[:60]
            report += f"  • {title}...\n"
        
        if len(successful) > 10:
            report += f"  ... and {len(successful) - 10} more\n"
        
        if failed:
            report += f"\n❌ FAILED PAPERS:\n"
            for fail in failed:
                report += f"  • {fail['paper_id']}: {fail['error']}\n"
        
        report += f"\n🎯 Next Steps:\n"
        report += f"1. Review generated summaries in: {self.output_dir}\n"
        report += f"2. Implement actual AI models for better summaries\n"
        report += f"3. Set up web interface for user access\n"
        
        # Save report
        report_file = os.path.join(self.output_dir, "batch_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n{report}")
        print(f"📋 Report saved to: {report_file}")

def main():
    """
    Main summary generation workflow
    """
    print("📝 Research Paper Summary Generator")
    print("=" * 50)
    
    # Check for PDFs from ArXiv collection
    pdf_dir = "datasets/arxiv/pdfs"
    
    if os.path.exists(pdf_dir):
        print(f"📁 Found PDF directory: {pdf_dir}")
        
        processor = PaperProcessor()
        summaries = processor.process_directory(pdf_dir)
        
        print(f"\n🎉 Summary generation complete!")
        print(f"📄 Processed {len(summaries)} papers")
        print(f"📂 Summaries saved in: outputs/summaries/")
        
    else:
        print(f"❌ PDF directory not found: {pdf_dir}")
        print("Please run arxiv_collector.py first to download papers")
        
        # Demo with single paper if available
        papers_dir = "papers"
        if os.path.exists(papers_dir):
            pdf_files = [f for f in os.listdir(papers_dir) if f.endswith('.pdf')]
            if pdf_files:
                print(f"\n📄 Found papers in {papers_dir}, processing first one as demo...")
                processor = PaperProcessor()
                demo_path = os.path.join(papers_dir, pdf_files[0])
                processor.process_single_paper(demo_path)

if __name__ == "__main__":
    main()
