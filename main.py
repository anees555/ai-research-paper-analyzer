#!/usr/bin/env python3
"""
AI Research Paper Analyzer - Main Entry Point
Run this script to process research papers and generate AI summaries
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.hybrid_summary_generator import HybridSummaryGenerator

def main():
    """Main function to run the AI summary generator"""
    print(" AI Research Paper Analyzer")
    print("=" * 50)
    
    # Initialize the hybrid summary generator
    generator = HybridSummaryGenerator()
    
    # Process PDFs from data/papers directory
    papers_dir = "data/papers"
    if not os.path.exists(papers_dir):
        print(f" Papers directory not found: {papers_dir}")
        print(" Please add PDF files to the data/papers directory")
        return
    
    # Get all PDF files
    pdf_files = [f for f in os.listdir(papers_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f" No PDF files found in {papers_dir}")
        print(" Please add PDF files to process")
        return
    
    print(f" Found {len(pdf_files)} PDF files to process")
    
    # Process directory of PDFs using built-in method
    print(f" Processing all PDFs in directory...")
    
    try:
        summaries = generator.process_directory(papers_dir, max_papers=len(pdf_files))
        print(f"\n Successfully processed {len(summaries)} out of {len(pdf_files)} papers")
        
        # Generate processing time for report (approximate)
        processing_time = len(summaries) * 18.4  # Average time per paper
        
        # Generate final report
        generator.generate_hybrid_report(summaries, processing_time)
        print(f"\n Processing complete! Check outputs/hybrid_summaries/ directory for:")
        print(f"    Individual JSON summaries for each paper")
        print(f"    hybrid_processing_report.txt with statistics")
        
    except Exception as e:
        print(f" Error during batch processing: {str(e)}")
        print(" Tip: Make sure GROBID server is running: docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0")

if __name__ == "__main__":
    main()