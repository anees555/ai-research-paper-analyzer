#!/usr/bin/env python3
"""
Dataset Preparation Pipeline
Complete workflow from raw papers to training-ready datasets
"""

import os
import json
from typing import List, Dict, Any
from scripts.parse_pdf import parse_pdf_with_grobid
from scripts.preprocess_text import clean_text_comprehensive, chunk_text_for_models

class DatasetPipeline:
    """
    End-to-end dataset preparation pipeline
    """
    
    def __init__(self, input_dir: str = "datasets/arxiv", output_dir: str = "datasets/training"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for different model types
        os.makedirs(f"{output_dir}/bart", exist_ok=True)
        os.makedirs(f"{output_dir}/longformer", exist_ok=True)
        os.makedirs(f"{output_dir}/bert", exist_ok=True)
        os.makedirs(f"{output_dir}/visual", exist_ok=True)
    
    def process_pdfs_batch(self, pdf_directory: str) -> List[Dict[str, Any]]:
        """
        Process all PDFs in directory with GROBID
        """
        print("🔄 Processing PDFs with GROBID...")
        
        processed_papers = []
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        
        for i, pdf_file in enumerate(pdf_files):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            
            try:
                print(f"🔄 Processing {i+1}/{len(pdf_files)}: {pdf_file}")
                
                # Use your existing GROBID parser
                result = parse_pdf_with_grobid(pdf_path, "output")
                
                paper_data = {
                    "paper_id": pdf_file.replace('.pdf', ''),
                    "title": result["title"],
                    "authors": result["authors"],
                    "abstract": result["abstract"],
                    "sections": result["sections"],
                    "file_path": pdf_path,
                    "processing_status": "success"
                }
                
                processed_papers.append(paper_data)
                print(f"✅ Success: {result['title'][:50]}...")
                
            except Exception as e:
                print(f"❌ Error processing {pdf_file}: {e}")
                
                # Still add to list with error status
                processed_papers.append({
                    "paper_id": pdf_file.replace('.pdf', ''),
                    "file_path": pdf_path,
                    "processing_status": "error",
                    "error": str(e)
                })
        
        return processed_papers
    
    def create_bart_training_data(self, papers: List[Dict[str, Any]]) -> str:
        """
        Create BART training dataset for summarization tasks
        """
        print("📝 Creating BART training dataset...")
        
        training_examples = []
        
        for paper in papers:
            if paper["processing_status"] != "success":
                continue
                
            # Task 1: Abstract generation from key sections
            if paper.get("abstract") and paper.get("sections"):
                # Combine introduction + conclusion for input
                input_sections = []
                for section_name in ["Introduction", "Conclusion", "Results"]:
                    if section_name in paper["sections"]:
                        section_text = clean_text_comprehensive(paper["sections"][section_name])
                        if len(section_text) > 50:  # Minimum content
                            input_sections.append(f"{section_name}: {section_text}")
                
                if input_sections:
                    input_text = "\n\n".join(input_sections)
                    
                    # Chunk input if too long (BART max ~1024 tokens)
                    chunks = chunk_text_for_models(input_text, "bart")
                    
                    for chunk in chunks:
                        training_examples.append({
                            "input_text": chunk["text"],
                            "target_text": clean_text_comprehensive(paper["abstract"]),
                            "paper_id": paper["paper_id"],
                            "task": "abstract_generation"
                        })
            
            # Task 2: One-line summaries
            if paper.get("abstract"):
                abstract_sentences = paper["abstract"].split(". ")
                if len(abstract_sentences) > 0:
                    one_line = abstract_sentences[0] + "."
                    
                    # Use title + first paragraph as input
                    if paper.get("sections") and "Introduction" in paper["sections"]:
                        intro_text = paper["sections"]["Introduction"]
                        intro_chunks = chunk_text_for_models(intro_text, "bart")
                        
                        if intro_chunks:
                            training_examples.append({
                                "input_text": f"Title: {paper['title']}\n\n{intro_chunks[0]['text']}",
                                "target_text": one_line,
                                "paper_id": paper["paper_id"],
                                "task": "one_line_summary"
                            })
        
        # Save BART dataset
        bart_file = f"{self.output_dir}/bart/training_data.jsonl"
        self.save_jsonl(training_examples, bart_file)
        print(f"💾 BART dataset: {len(training_examples)} examples saved to {bart_file}")
        
        return bart_file
    
    def create_longformer_training_data(self, papers: List[Dict[str, Any]]) -> str:
        """
        Create Longformer training dataset for deep analysis
        """
        print("📚 Creating Longformer training dataset...")
        
        training_examples = []
        
        for paper in papers:
            if paper["processing_status"] != "success":
                continue
            
            # Combine full paper content
            full_text = f"Title: {paper['title']}\n\n"
            
            if paper.get("abstract"):
                full_text += f"Abstract: {paper['abstract']}\n\n"
            
            # Add all sections
            if paper.get("sections"):
                for section_name, section_content in paper["sections"].items():
                    if section_content and len(section_content.strip()) > 100:
                        clean_content = clean_text_comprehensive(section_content)
                        full_text += f"{section_name}:\n{clean_content}\n\n"
            
            # Create comprehensive analysis target
            analysis_target = ""
            if paper.get("abstract"):
                analysis_target += f"Summary: {paper['abstract']}\n\n"
            
            # Extract key insights from sections
            key_sections = ["Introduction", "Methods", "Results", "Conclusion"]
            for section in key_sections:
                if paper.get("sections") and section in paper["sections"]:
                    section_text = paper["sections"][section]
                    if len(section_text) > 100:
                        # Use first 200 chars as insight (in practice, use LLM to generate)
                        insight = section_text[:200] + "..."
                        analysis_target += f"{section} Key Point: {insight}\n\n"
            
            # Only include if substantial content
            if len(full_text) > 2000 and len(analysis_target) > 300:
                training_examples.append({
                    "input_text": full_text.strip(),
                    "target_text": analysis_target.strip(),
                    "paper_id": paper["paper_id"],
                    "task": "deep_analysis"
                })
        
        # Save Longformer dataset
        longformer_file = f"{self.output_dir}/longformer/training_data.jsonl"
        self.save_jsonl(training_examples, longformer_file)
        print(f"💾 Longformer dataset: {len(training_examples)} examples saved to {longformer_file}")
        
        return longformer_file
    
    def create_bert_classification_data(self, papers: List[Dict[str, Any]]) -> str:
        """
        Create BERT training dataset for classification tasks
        """
        print("🏷️  Creating BERT classification dataset...")
        
        training_examples = []
        
        for paper in papers:
            if paper["processing_status"] != "success":
                continue
            
            # Text for classification (title + abstract)
            classification_text = paper["title"]
            if paper.get("abstract"):
                classification_text += f" {paper['abstract']}"
            
            # Domain classification (simple heuristic - in practice, use expert labels)
            domain = self.classify_domain(classification_text)
            training_examples.append({
                "text": classification_text,
                "label": domain,
                "paper_id": paper["paper_id"],
                "task": "domain_classification"
            })
            
            # Quality assessment (based on structure)
            quality = self.assess_quality(paper)
            training_examples.append({
                "text": classification_text,
                "label": quality,
                "paper_id": paper["paper_id"],
                "task": "quality_assessment"
            })
        
        # Save BERT dataset
        bert_file = f"{self.output_dir}/bert/training_data.jsonl"
        self.save_jsonl(training_examples, bert_file)
        print(f"💾 BERT dataset: {len(training_examples)} examples saved to {bert_file}")
        
        return bert_file
    
    def classify_domain(self, text: str) -> str:
        """Simple domain classification based on keywords"""
        text_lower = text.lower()
        
        ai_keywords = ["neural", "deep learning", "machine learning", "artificial intelligence", "ai"]
        nlp_keywords = ["natural language", "nlp", "text", "language model", "linguistic"]
        cv_keywords = ["computer vision", "image", "visual", "cnn", "convolutional"]
        
        if any(keyword in text_lower for keyword in ai_keywords):
            return "artificial_intelligence"
        elif any(keyword in text_lower for keyword in nlp_keywords):
            return "natural_language_processing"
        elif any(keyword in text_lower for keyword in cv_keywords):
            return "computer_vision"
        else:
            return "other"
    
    def assess_quality(self, paper: Dict[str, Any]) -> str:
        """Simple quality assessment based on structure"""
        score = 0
        
        # Has abstract
        if paper.get("abstract") and len(paper["abstract"]) > 100:
            score += 1
        
        # Has multiple sections
        if paper.get("sections") and len(paper["sections"]) >= 4:
            score += 1
        
        # Has substantial content
        total_content = sum(len(section) for section in paper.get("sections", {}).values())
        if total_content > 5000:
            score += 1
        
        if score >= 3:
            return "high_quality"
        elif score >= 2:
            return "medium_quality"
        else:
            return "low_quality"
    
    def save_jsonl(self, data: List[Dict], filepath: str):
        """Save data in JSONL format"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def generate_dataset_report(self, datasets: Dict[str, str]) -> str:
        """Generate summary report of created datasets"""
        report = "📊 DATASET GENERATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        total_examples = 0
        
        for dataset_name, filepath in datasets.items():
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    count = sum(1 for line in f)
                
                report += f"📁 {dataset_name.upper()}:\n"
                report += f"   File: {filepath}\n"
                report += f"   Examples: {count:,}\n"
                report += f"   Size: {os.path.getsize(filepath) / 1024 / 1024:.1f} MB\n\n"
                
                total_examples += count
        
        report += f"🎯 TOTAL TRAINING EXAMPLES: {total_examples:,}\n"
        report += f"📂 Output directory: {self.output_dir}\n"
        
        # Save report
        report_file = f"{self.output_dir}/dataset_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        return report_file

def main():
    """
    Run complete dataset preparation pipeline
    """
    print("🏭 Dataset Preparation Pipeline")
    print("=" * 50)
    
    pipeline = DatasetPipeline()
    
    # Step 1: Check for processed papers or PDFs
    pdf_dir = "datasets/arxiv/pdfs"
    if not os.path.exists(pdf_dir):
        print(f"❌ PDF directory not found: {pdf_dir}")
        print("Please run arxiv_collector.py first to collect papers")
        return
    
    # Step 2: Process PDFs with GROBID
    processed_papers = pipeline.process_pdfs_batch(pdf_dir)
    print(f"\n✅ Processed {len(processed_papers)} papers")
    
    # Step 3: Create training datasets
    print("\n🔄 Creating training datasets...")
    datasets = {}
    
    datasets["bart"] = pipeline.create_bart_training_data(processed_papers)
    datasets["longformer"] = pipeline.create_longformer_training_data(processed_papers)
    datasets["bert"] = pipeline.create_bert_classification_data(processed_papers)
    
    # Step 4: Generate report
    print("\n📊 Generating dataset report...")
    report_file = pipeline.generate_dataset_report(datasets)
    
    print(f"\n🎉 Dataset preparation complete!")
    print(f"📋 Report saved to: {report_file}")
    print("\nNext steps:")
    print("1. Review dataset quality")
    print("2. Begin model fine-tuning")
    print("3. Set up training pipelines")

if __name__ == "__main__":
    main()
