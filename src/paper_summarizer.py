"""
paper_summarizer.py
A robust, modular pipeline for research paper summarization.
"""
import re
from typing import Dict, List
from lxml import etree

TARGET_SECTIONS = [
    'abstract', 'introduction', 'method', 'methodology', 'results', 'experiments', 'conclusion'
]

# 1. Text Cleaning

def clean_text(text: str) -> str:
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # Remove emails
    text = re.sub(r'\[\d+\]', '', text)  # Remove [1], [2]
    text = re.sub(r'\([A-Za-z\s,]+et al\.,?\s*\d{4}\)', '', text)  # Remove (Smith et al., 2020)
    lines = text.split('\n')
    seen = set()
    deduped = []
    for line in lines:
        l = line.strip()
        if l and l not in seen:
            deduped.append(l)
            seen.add(l)
    text = '\n'.join(deduped)
    text = re.sub(r'\b\w{30,}\b', '', text)  # Remove long garbage tokens
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

# 2. Section Filtering

def extract_sections_from_grobid(xml_str: str) -> Dict[str, str]:
    # DEBUG: Save raw XML for inspection
    with open('grobid_last_output.xml', 'w', encoding='utf-8') as f:
        f.write(xml_str)

    # Parse with namespace awareness
    try:
        tree = etree.fromstring(xml_str.encode('utf-8'))
    except Exception as e:
        print(f"[ERROR] Failed to parse XML: {e}")
        return {}

    nsmap = tree.nsmap.copy()
    nsmap[None] = nsmap.get(None, 'http://www.tei-c.org/ns/1.0')
    ns = {'tei': nsmap[None]}

    # Canonical section mapping (singular, plural, common variants)
    section_map = {
        r"abstract": "abstract",
        r"introduction|background": "introduction",
        r"methodology|methods|method": "method",
        r"conclusion[s]?|concluding remarks|summary and conclusion[s]?": "conclusion",
    }

    sections = {}
    unmatched_headers = []
    # Find all section divs (TEI format)
    for sec in tree.xpath('.//tei:div[@type="section"]', namespaces=ns):
        header = sec.findtext('.//tei:head', namespaces=ns)
        if header:
            header_lower = header.lower().strip()
            matched = False
            for pattern, canonical in section_map.items():
                if re.search(rf"\\b({pattern})\\b", header_lower):
                    # Only clean text, do not merge or reformat
                    text = ' '.join(sec.itertext())
                    # Only keep first occurrence of each canonical section
                    if canonical not in sections:
                        sections[canonical] = clean_text(text)
                    matched = True
                    break
            if not matched:
                unmatched_headers.append(header)
    # Abstract
    abstract = tree.find('.//tei:abstract', namespaces=ns)
    if abstract is not None:
        sections['abstract'] = clean_text(' '.join(abstract.itertext()))
    if unmatched_headers:
        print(f"[DEBUG] Unmatched section headers: {unmatched_headers}")
    return sections

# 3. Chunking

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap
    return chunks

# 4. Summarization

def summarize_chunks(chunks: List[str], model, tokenizer, max_length: int = 300) -> List[str]:
    summaries = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', truncation=True, max_length=1024)
        summary_ids = model.generate(
            inputs['input_ids'],
            max_length=max_length,
            num_beams=4,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries

# 5. Merge Summaries

def merge_summaries(section_summaries: List[str]) -> str:
    return ' '.join(section_summaries)

# 6. Orchestrator

def summarize_paper(pdf_path: str, model, tokenizer) -> Dict[str, str]:
    xml_str = parse_pdf_with_grobid(pdf_path)
    sections = extract_sections_from_grobid(xml_str)
    final_output = {}
    for section, text in sections.items():
        chunks = chunk_text(text)
        chunk_summaries = summarize_chunks(chunks, model, tokenizer)
        final_output[section] = merge_summaries(chunk_summaries)
    return final_output

# 7. Placeholder for GROBID parsing (to be implemented or imported)

import requests

def parse_pdf_with_grobid(pdf_path: str) -> str:
    """Call GROBID server and return XML as string."""
    with open(pdf_path, 'rb') as f:
        files = {'input': (pdf_path, f, 'application/pdf')}
        response = requests.post(
            'http://localhost:8070/api/processFulltextDocument',  # Change if your GROBID server is elsewhere
            files=files,
            data={'consolidateHeader': '1', 'consolidateCitations': '1'}
        )
    response.raise_for_status()
    return response.text

# 8. Output formatting

def format_output(section_summaries: Dict[str, str]) -> str:
    order = ['abstract', 'introduction', 'method', 'methodology', 'results', 'experiments', 'conclusion']
    output = []
    for section in order:
        if section in section_summaries:
            title = section.capitalize() + ' Summary:'
            output.append(f'{title}\n{section_summaries[section]}\n')
    return '\n'.join(output)
