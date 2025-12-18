#!/usr/bin/env python3
"""
ArXiv Dataset Collection Script
Practical implementation for collecting research papers from ArXiv API
"""

import os
import requests
import xml.etree.ElementTree as ET
import json
import time
from datetime import datetime
from typing import List, Dict, Any

class ArXivCollector:
    """
    Collect papers from ArXiv API for training dataset
    """
    
    def __init__(self, output_dir: str = "datasets/arxiv"):
        self.base_url = "http://export.arxiv.org/api/query"
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def search_papers(self, 
                     categories: List[str] = ["cs.AI", "cs.CL", "cs.LG"],
                     max_results: int = 1000,
                     start: int = 0) -> List[Dict[str, Any]]:
        """
        Search ArXiv papers by categories
        
        Popular CS categories:
        - cs.AI: Artificial Intelligence
        - cs.CL: Computation and Language (NLP)
        - cs.LG: Machine Learning
        - cs.CV: Computer Vision
        - cs.IR: Information Retrieval
        """
        
        papers = []
        
        for category in categories:
            print(f"🔍 Searching category: {category}")
            
            # Build query
            query = f"cat:{category}"
            params = {
                "search_query": query,
                "start": start,
                "max_results": max_results // len(categories),
                "sortBy": "submittedDate",
                "sortOrder": "descending"
            }
            
            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                
                # Parse XML response
                root = ET.fromstring(response.content)
                
                # Extract paper information
                for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
                    paper = self.parse_entry(entry, category)
                    if paper:
                        papers.append(paper)
                
                print(f"✅ Found {len([p for p in papers if p['category'] == category])} papers in {category}")
                
                # Rate limiting - ArXiv requests to be polite
                time.sleep(3)
                
            except Exception as e:
                print(f"❌ Error fetching {category}: {e}")
                continue
        
        return papers
    
    def parse_entry(self, entry: ET.Element, category: str) -> Dict[str, Any]:
        """
        Parse individual ArXiv entry XML
        """
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        
        try:
            # Extract basic information
            arxiv_id = entry.find("atom:id", ns).text.split("/")[-1]
            title = entry.find("atom:title", ns).text.strip()
            summary = entry.find("atom:summary", ns).text.strip()
            
            # Extract authors
            authors = []
            for author in entry.findall("atom:author", ns):
                name = author.find("atom:name", ns).text
                authors.append(name)
            
            # Extract publication date
            published = entry.find("atom:published", ns).text
            
            # Extract PDF link
            pdf_url = None
            for link in entry.findall("atom:link", ns):
                if link.get("type") == "application/pdf":
                    pdf_url = link.get("href")
                    break
            
            paper = {
                "paper_id": arxiv_id,
                "title": title,
                "abstract": summary,
                "authors": authors,
                "published": published,
                "category": category,
                "pdf_url": pdf_url,
                "source": "arxiv"
            }
            
            return paper
            
        except Exception as e:
            print(f"❌ Error parsing entry: {e}")
            return None
    
    def download_pdfs(self, papers: List[Dict[str, Any]], max_downloads: int = 100) -> List[str]:
        """
        Download PDF files from ArXiv
        """
        downloaded_files = []
        pdf_dir = os.path.join(self.output_dir, "pdfs")
        os.makedirs(pdf_dir, exist_ok=True)
        
        for i, paper in enumerate(papers[:max_downloads]):
            if not paper["pdf_url"]:
                continue
                
            try:
                filename = f"{paper['paper_id'].replace('/', '_')}.pdf"
                filepath = os.path.join(pdf_dir, filename)
                
                # Skip if already downloaded
                if os.path.exists(filepath):
                    downloaded_files.append(filepath)
                    continue
                
                print(f"⬇️  Downloading {i+1}/{max_downloads}: {filename}")
                
                response = requests.get(paper["pdf_url"])
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                downloaded_files.append(filepath)
                
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Error downloading {paper['paper_id']}: {e}")
                continue
        
        return downloaded_files
    
    def save_metadata(self, papers: List[Dict[str, Any]], filename: str = "papers_metadata.json"):
        """
        Save paper metadata to JSON file
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved metadata for {len(papers)} papers to {filepath}")
        return filepath

def create_quick_start_dataset():
    """
    Quick start: Create a small dataset for immediate experimentation
    """
    print("🚀 Quick Start Dataset Collection")
    print("=" * 50)
    
    collector = ArXivCollector()
    
    # Collect recent papers (small batch for testing)
    papers = collector.search_papers(
        categories=["cs.AI", "cs.CL"],  # AI and NLP papers
        max_results=50,  # Small batch for testing
        start=0
    )
    
    print(f"\n📊 Collected {len(papers)} papers")
    
    # Save metadata
    metadata_file = collector.save_metadata(papers, "quick_start_metadata.json")
    
    # Download first 10 PDFs for testing
    print("\n⬇️  Downloading sample PDFs...")
    pdf_files = collector.download_pdfs(papers, max_downloads=10)
    
    print(f"\n✅ Quick start dataset ready!")
    print(f"📁 Metadata: {metadata_file}")
    print(f"📄 PDFs: {len(pdf_files)} files downloaded")
    print(f"📂 Location: {collector.output_dir}")
    
    return papers, pdf_files

def main():
    """
    Main dataset collection workflow
    """
    print("📚 ArXiv Dataset Collection for AI Research Paper Analyzer")
    print("=" * 60)
    
    choice = input("\nChoose collection mode:\n1. Quick start (50 papers)\n2. Full collection (1000+ papers)\nEnter choice (1/2): ")
    
    if choice == "1":
        create_quick_start_dataset()
    else:
        collector = ArXivCollector()
        
        # Full collection
        papers = collector.search_papers(
            categories=["cs.AI", "cs.CL", "cs.LG", "cs.CV"],
            max_results=1000,
            start=0
        )
        
        print(f"\n📊 Collected {len(papers)} papers")
        
        # Save metadata
        collector.save_metadata(papers)
        
        # Download PDFs (ask user how many)
        max_pdfs = int(input(f"\nHow many PDFs to download? (max {len(papers)}): "))
        pdf_files = collector.download_pdfs(papers, max_downloads=max_pdfs)
        
        print(f"\n✅ Dataset collection complete!")
        print(f"📄 Downloaded {len(pdf_files)} PDFs")

if __name__ == "__main__":
    main()
