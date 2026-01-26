#!/usr/bin/env python3
"""
Optimized PDF parser with advanced GROBID timeout handling
Multiple strategies to handle large PDF processing
"""

import os
import xml.etree.ElementTree as ET
import requests
import logging
import sys
import time
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedGROBIDProcessor:
    """
    Optimized GROBID processor with intelligent timeout handling
    """
    
    def __init__(self, base_url: str = "http://localhost:8070"):
        self.base_url = base_url
        self.timeout_strategies = {
            'quick': 60,      # 1 minute for small docs
            'normal': 300,    # 5 minutes for medium docs  
            'extended': 600,  # 10 minutes for large docs
            'maximum': 900    # 15 minutes for very large docs
        }
    
    def estimate_processing_time(self, pdf_path: str) -> str:
        """
        Estimate processing time based on PDF size
        """
        try:
            file_size = os.path.getsize(pdf_path)
            size_mb = file_size / (1024 * 1024)
            
            if size_mb < 1:
                return 'quick'      # < 1MB
            elif size_mb < 5:
                return 'normal'     # 1-5MB
            elif size_mb < 15:
                return 'extended'   # 5-15MB
            else:
                return 'maximum'    # > 15MB
                
        except Exception:
            return 'normal'  # Default fallback
    
    def check_grobid_health(self) -> bool:
        """
        Enhanced GROBID health check with retries
        """
        max_retries = 5
        retry_delays = [2, 5, 10, 15, 20]  # Progressive delays
        
        # Try both endpoint variations
        endpoints = ["/api/isalive", "/api/isAlive"]
        
        for attempt in range(max_retries):
            for endpoint in endpoints:
                try:
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                    if response.status_code == 200 and "true" in response.text.lower():
                        logger.info(f"[SUCCESS] GROBID server is healthy and responsive (endpoint: {endpoint})")
                        return True
                        
                except requests.exceptions.RequestException:
                    continue  # Try next endpoint
            
            # If no endpoint worked, wait and retry
            logger.warning(f"[WARNING] GROBID connection attempt {attempt + 1}/{max_retries} failed")
            
            if attempt < max_retries - 1:
                delay = retry_delays[attempt]
                logger.info(f"⏳ Waiting {delay} seconds before retry...")
                time.sleep(delay)
        
        logger.error("[ERROR] GROBID server is not reachable after all retries")
        return False
    
    def process_pdf_with_strategy(self, pdf_path: str, strategy: str = None) -> requests.Response:
        """
        Process PDF with specific timeout strategy
        """
        if strategy is None:
            strategy = self.estimate_processing_time(pdf_path)
        
        timeout = self.timeout_strategies[strategy]
        file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
        
        logger.info(f"[STATS] PDF Analysis: {file_size:.1f}MB → Using '{strategy}' strategy ({timeout}s timeout)")
        
        url = f"{self.base_url}/api/processFulltextDocument"
        
        with open(pdf_path, 'rb') as f:
            files = {'input': f}
            data = {
                'consolidateHeader': '1',
                'consolidateCitations': '1',
                'generateIDs': '1'
            }
            
            logger.info(f"[INIT] Sending PDF to GROBID (timeout: {timeout}s)...")
            start_time = time.time()
            
            try:
                response = requests.post(
                    url, 
                    files=files, 
                    data=data, 
                    timeout=timeout,
                    stream=False  # Don't stream for better reliability
                )
                
                processing_time = time.time() - start_time
                logger.info(f"[TIME] Processing completed in {processing_time:.1f}s")
                
                return response
                
            except requests.exceptions.Timeout:
                processing_time = time.time() - start_time
                logger.warning(f"[TIMEOUT] Timeout after {processing_time:.1f}s with '{strategy}' strategy")
                raise
    
    def process_with_fallback_strategies(self, pdf_path: str) -> requests.Response:
        """
        Try multiple timeout strategies with fallbacks
        """
        # Try strategies in order of estimated need
        base_strategy = self.estimate_processing_time(pdf_path)
        
        # Define fallback sequence
        if base_strategy == 'quick':
            strategies = ['quick', 'normal']
        elif base_strategy == 'normal':
            strategies = ['normal', 'extended']  
        elif base_strategy == 'extended':
            strategies = ['extended', 'maximum']
        else:
            strategies = ['maximum']
        
        last_error = None
        
        for i, strategy in enumerate(strategies):
            try:
                logger.info(f"[TARGET] Attempt {i+1}/{len(strategies)}: Trying '{strategy}' strategy")
                response = self.process_pdf_with_strategy(pdf_path, strategy)
                
                if response.status_code == 200:
                    logger.info(f"[SUCCESS] Success with '{strategy}' strategy!")
                    return response
                else:
                    logger.warning(f"[FAILED] Failed with status {response.status_code}")
                    
            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning(f"[TIMEOUT] '{strategy}' strategy timed out")
                continue
            except Exception as e:
                last_error = e
                logger.error(f"[ERROR] Error with '{strategy}' strategy: {e}")
                continue
        
        # All strategies failed
        logger.error("[ERROR] All timeout strategies failed")
        raise last_error or Exception("All processing strategies exhausted")
    
    def parse_pdf_optimized(self, pdf_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Main optimized PDF parsing function
        """
        logger.info(f"[INFO] Starting optimized processing: {os.path.basename(pdf_path)}")
        
        # Validate inputs
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError("Input file must be a PDF")
        
        # Check GROBID health
        if not self.check_grobid_health():
            raise ConnectionError("GROBID server is not available")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process with fallback strategies
        try:
            response = self.process_with_fallback_strategies(pdf_path)
            
            if response.status_code != 200:
                raise Exception(f"GROBID processing failed: HTTP {response.status_code}")
            
            # Save TEI XML output
            pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
            output_file = os.path.join(output_dir, f"{pdf_name}.tei.xml")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            xml_size = len(response.text)
            logger.info(f"[SAVED] Saved TEI XML: {output_file} ({xml_size:,} chars)")
            
            # Parse XML and extract data
            logger.info("[PARSING] Parsing TEI XML structure...")
            return self.parse_tei_xml(response.text)
            
        except Exception as e:
            logger.error(f"[ERROR] Optimized processing failed: {e}")
            raise
    
    def parse_tei_xml(self, xml_content: str) -> Dict[str, Any]:
        """
        Parse TEI XML with enhanced error handling
        """
        try:
            root = ET.fromstring(xml_content)
            ns = {"tei": "http://www.tei-c.org/ns/1.0"}
            
            # Extract metadata
            title_elem = root.find(".//tei:titleStmt/tei:title", ns)
            title = title_elem.text.strip() if title_elem is not None and title_elem.text else "Unknown Title"
            
            # Extract authors
            authors = []
            for author in root.findall(".//tei:author", ns):
                forename = author.find(".//tei:forename", ns)
                surname = author.find(".//tei:surname", ns)
                
                if forename is not None and surname is not None:
                    author_name = f"{forename.text.strip()} {surname.text.strip()}"
                    authors.append(author_name)
            
            # Extract abstract
            abstract_elem = root.find(".//tei:abstract", ns)
            abstract = ""
            if abstract_elem is not None:
                abstract_parts = []
                for p in abstract_elem.findall(".//tei:p", ns):
                    if p.text:
                        abstract_parts.append(p.text.strip())
                abstract = " ".join(abstract_parts)
            
            # Extract sections with improved traversal and filtering
            sections = {}
            body = root.find(".//tei:body", ns)
            if body is not None:
                # Get all div elements, including nested ones
                all_divs = body.findall(".//tei:div", ns)
                logger.info(f"[SEARCH] Found {len(all_divs)} div elements in document body")
                
                for div in all_divs:
                    head = div.find("tei:head", ns)
                    if head is not None and head.text:
                        section_title = head.text.strip()
                        
                        # Skip figure/table captions and very short headings that are likely not real sections
                        if len(section_title) < 3:  # Skip 1-2 char headers like "The", "A", etc.
                            logger.warning(f"[WARNING]  Skipping very short section title: '{section_title}'")
                            continue
                        
                        # Extract section content from paragraphs directly under this div
                        section_content = []
                        for p in div.findall("tei:p", ns):  # Direct children only
                            if p.text:
                                # Get full text content including nested elements
                                full_text = ''.join(p.itertext()).strip()
                                if full_text and len(full_text) > 10:  # Only meaningful content
                                    section_content.append(full_text)
                        
                        # Also check for nested paragraphs
                        for nested_div in div.findall("tei:div", ns):
                            for p in nested_div.findall(".//tei:p", ns):
                                if p.text:
                                    full_text = ''.join(p.itertext()).strip()
                                    if full_text and len(full_text) > 10:
                                        section_content.append(full_text)
                        
                        # Only add section if it has substantial content (at least 50 chars total)
                        combined_content = " ".join(section_content)
                        if len(combined_content) >= 50:
                            sections[section_title] = combined_content
                            logger.info(f"[SUCCESS] Extracted section: '{section_title}' ({len(section_content)} paragraphs, {len(combined_content)} chars)")
                        else:
                            logger.warning(f"[WARNING]  Skipping section with insufficient content: '{section_title}' ({len(combined_content)} chars)")
                
                logger.info(f"[STATS] Total valid sections extracted: {len(sections)}")
            
            result = {
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "sections": sections,
                "processing_stats": {
                    "xml_size": len(xml_content),
                    "sections_found": len(sections),
                    "authors_found": len(authors),
                    "has_abstract": bool(abstract)
                }
            }
            
            logger.info(f"[STATS] Extraction complete: {len(sections)} sections, {len(authors)} authors")
            return result
            
        except ET.ParseError as e:
            logger.error(f"[ERROR] TEI XML parsing failed: {e}")
            raise Exception(f"Invalid TEI XML structure: {e}")
        except Exception as e:
            logger.error(f"[ERROR] TEI processing error: {e}")
            raise

# Main function for backward compatibility
def parse_pdf_with_grobid_optimized(pdf_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Optimized version of the original parse_pdf_with_grobid function
    """
    processor = OptimizedGROBIDProcessor()
    return processor.parse_pdf_optimized(pdf_path, output_dir)

# Test function
def test_optimization():
    """
    Test the optimization with available PDFs
    """
    print("🧪 Testing GROBID Optimization")
    print("=" * 50)
    
    processor = OptimizedGROBIDProcessor()
    
    # Test health check
    if not processor.check_grobid_health():
        print("[ERROR] GROBID server not available for testing")
        return
    
    # Find test PDF
    test_dirs = ["datasets/arxiv/pdfs", "papers"]
    test_pdf = None
    
    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            pdf_files = [f for f in os.listdir(dir_path) if f.endswith('.pdf')]
            if pdf_files:
                test_pdf = os.path.join(dir_path, pdf_files[0])
                break
    
    if not test_pdf:
        print("[ERROR] No test PDFs found")
        return
    
    print(f"[INFO] Testing with: {os.path.basename(test_pdf)}")
    
    try:
        result = processor.parse_pdf_optimized(test_pdf, "output_test")
        print(f"[SUCCESS] Success! Extracted {len(result['sections'])} sections")
        print(f"[STATS] Title: {result['title'][:100]}...")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")

if __name__ == "__main__":
    test_optimization()
