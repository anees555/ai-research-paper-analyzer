import re
import sys
import os
import nltk
from typing import List, Dict, Tuple

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def clean_basic_text(text):
    """
    Basic text cleaning - removes extra whitespace and fixes simple formatting
    
    Args:
        text (str): Raw text from PDF extraction
    
    Returns:
        str: Cleaned text
    """
    # Step 1: Handle None or empty text
    if not text or not text.strip():
        return ""
    
    # Step 2: Remove extra whitespace and normalize spaces
    # Replace multiple spaces, tabs, newlines with single space
    cleaned_text = re.sub(r'\s+', ' ', text)
    
    # Step 3: Fix spacing around punctuation
    # Remove space before punctuation
    cleaned_text = re.sub(r'\s+([,.!?;:])', r'\1', cleaned_text)
    # Ensure single space after punctuation
    cleaned_text = re.sub(r'([,.!?;:])\s*', r'\1 ', cleaned_text)
    
    # Step 4: Basic cleanup
    # Remove extra spaces at start and end
    cleaned_text = cleaned_text.strip()
    
    # Step 5: Handle common PDF artifacts
    # Fix broken words (simple cases like "h t" -> "ht")
    cleaned_text = re.sub(r'\b([a-z])\s+([a-z])\b', r'\1\2', cleaned_text)
    
    return cleaned_text


def clean_pdf_artifacts(text):
    """
    Advanced cleaning for PDF extraction artifacts and academic paper issues
    
    Args:
        text (str): Text that has been through basic cleaning
    
    Returns:
        str: Text with PDF artifacts removed
    """
    if not text:
        return ""
    
    cleaned_text = text
    
    # Step 1: Fix broken references and citations
    # Fix spaced references like "[ 1 ]" -> "[1]"
    cleaned_text = re.sub(r'\[\s*(\d+)\s*\]', r'[\1]', cleaned_text)
    # Fix spaced citations like "( Smith et al. , 2020 )" -> "(Smith et al., 2020)"
    cleaned_text = re.sub(r'\(\s*([^)]+?)\s*,\s*(\d{4})\s*\)', r'(\1, \2)', cleaned_text)
    
    # Step 2: Fix mathematical notation and variables
    # Fix spaced variables like "x ∈ R d" -> "x ∈ Rd"
    cleaned_text = re.sub(r'(\w)\s*∈\s*(\w)\s+(\w)', r'\1 ∈ \2\3', cleaned_text)
    # Fix broken subscripts like "h t" -> "h_t" (common in academic papers)
    cleaned_text = re.sub(r'\b([a-zA-Z])\s+([a-zA-Z])\s*(?=[,\.\s])', r'\1_\2', cleaned_text)
    
    # Step 3: Fix broken words across line breaks
    # Fix hyphenated words like "atten-tion" -> "attention"
    cleaned_text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', cleaned_text)
    # Fix broken words with spaces like "trans former" -> "transformer"
    cleaned_text = re.sub(r'\b(trans)\s+(former|formation|mit)\b', r'\1\2', cleaned_text)
    cleaned_text = re.sub(r'\b(multi)\s+(head|layer|task)\b', r'\1-\2', cleaned_text)
    cleaned_text = re.sub(r'\b(self)\s+(attention)\b', r'\1-\2', cleaned_text)
    
    # Step 4: Clean up figure and table references
    # Fix spaced figure refs like "Figure 1" -> "Figure 1" (normalize spacing)
    cleaned_text = re.sub(r'(Figure|Fig|Table|Equation)\s*\.?\s*(\d+)', r'\1 \2', cleaned_text)
    # Remove standalone figure/table captions that got mixed in
    cleaned_text = re.sub(r'\b(Figure|Table)\s+\d+[:.].*?(?=\.|$)', '', cleaned_text)
    
    # Step 5: Fix common academic notation
    # Fix degree symbols and mathematical notation
    cleaned_text = re.sub(r'(\d+)\s*°\s*', r'\1° ', cleaned_text)
    # Fix percentage with spaces
    cleaned_text = re.sub(r'(\d+)\s*%', r'\1%', cleaned_text)
    
    # Step 6: Remove extra artifacts
    # Remove multiple periods that come from broken formatting
    cleaned_text = re.sub(r'\.{3,}', '...', cleaned_text)
    # Remove extra commas and fix spacing
    cleaned_text = re.sub(r',\s*,+', ',', cleaned_text)
    
    # Step 7: Final cleanup
    # Remove any leftover multiple spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text


def clean_academic_text(text):
    """
    Section-specific cleaning for academic papers
    
    Args:
        text (str): Text that has been through basic and artifact cleaning
    
    Returns:
        str: Text optimized for academic content
    """
    if not text:
        return ""
    
    cleaned_text = text
    
    # Step 1: Handle citations and references better
    # Standardize author citations format
    cleaned_text = re.sub(r'\(([A-Z][a-z]+)\s+et\s+al\.\s*,?\s*(\d{4})\)', r'(\1 et al., \2)', cleaned_text)
    
    # Step 2: Fix common academic abbreviations
    # Standardize common abbreviations
    abbreviations = {
        r'\be\.g\.\s*': 'e.g., ',
        r'\bi\.e\.\s*': 'i.e., ',
        r'\bvs\.\s*': 'vs. ',
        r'\betc\.\s*': 'etc. '
    }
    for pattern, replacement in abbreviations.items():
        cleaned_text = re.sub(pattern, replacement, cleaned_text)
    
    # Step 3: Handle section headers that got mixed in
    # Remove standalone section numbers
    cleaned_text = re.sub(r'^\s*\d+\.?\s*$', '', cleaned_text, flags=re.MULTILINE)
    
    # Step 4: Fix equation references
    # Standardize equation references
    cleaned_text = re.sub(r'(Equation|Eq)\s*\.?\s*\((\d+)\)', r'\1 (\2)', cleaned_text)
    
    return cleaned_text


def clean_text_comprehensive(text):
    """
    Apply all cleaning steps in the right order
    
    Args:
        text (str): Raw text from PDF extraction
    
    Returns:
        str: Fully cleaned text ready for processing
    """
    # Apply cleaning in order
    cleaned = clean_basic_text(text)
    cleaned = clean_pdf_artifacts(cleaned)
    cleaned = clean_academic_text(cleaned)
    
    return cleaned


def chunk_text_for_models(text: str, model_type: str = "bart") -> List[Dict[str, any]]:
    """
    Split text into chunks optimized for different summarization models
    
    Args:
        text (str): Clean text to be chunked
        model_type (str): "bart" for short chunks, "longformer" for long chunks
    
    Returns:
        List[Dict]: List of chunks with metadata
    """
    if not text:
        return []
    
    # Define chunk parameters based on model
    if model_type.lower() == "bart":
        # BART: ~512 tokens ≈ 350 words for one-line and bullet summaries
        target_words = 350
        max_words = 450
        min_words = 100
    elif model_type.lower() == "longformer":
        # Longformer: ~1024-4096 tokens ≈ 800-3000 words for deep summaries
        target_words = 1000
        max_words = 1200
        min_words = 300
    else:
        # Default to BART settings
        target_words = 350
        max_words = 450
        min_words = 100
    
    # Split text into sentences
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        # If adding this sentence would exceed max_words, finalize current chunk
        if current_word_count + sentence_words > max_words and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "word_count": current_word_count,
                "sentence_count": len(current_chunk),
                "model_type": model_type
            })
            current_chunk = []
            current_word_count = 0
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_word_count += sentence_words
        
        # If we've reached target words, consider finishing chunk at sentence boundary
        if current_word_count >= target_words:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "word_count": current_word_count,
                "sentence_count": len(current_chunk),
                "model_type": model_type
            })
            current_chunk = []
            current_word_count = 0
    
    # Add remaining sentences as final chunk (if above minimum)
    if current_chunk and current_word_count >= min_words:
        chunk_text = ' '.join(current_chunk)
        chunks.append({
            "text": chunk_text,
            "word_count": current_word_count,
            "sentence_count": len(current_chunk),
            "model_type": model_type
        })
    elif current_chunk and chunks:
        # If final chunk is too small, merge with previous chunk
        chunks[-1]["text"] += " " + ' '.join(current_chunk)
        chunks[-1]["word_count"] += current_word_count
        chunks[-1]["sentence_count"] += len(current_chunk)
    
    return chunks


def chunk_sections_for_summarization(sections: Dict[str, str], model_type: str = "bart") -> Dict[str, List[Dict]]:
    """
    Chunk all sections of a paper for summarization
    
    Args:
        sections (Dict[str, str]): Dictionary of section_name: section_text
        model_type (str): "bart" or "longformer"
    
    Returns:
        Dict[str, List[Dict]]: Dictionary of section_name: list_of_chunks
    """
    chunked_sections = {}
    
    for section_name, section_text in sections.items():
        # Clean the text first
        cleaned_text = clean_text_comprehensive(section_text)
        
        # Chunk the cleaned text
        chunks = chunk_text_for_models(cleaned_text, model_type)
        
        # Add section metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk["section_name"] = section_name
            chunk["chunk_index"] = i
            chunk["total_chunks_in_section"] = len(chunks)
        
        chunked_sections[section_name] = chunks
    
    return chunked_sections


def prepare_paper_for_summarization(paper_data: Dict, bart_sections: List[str] = None, longformer_sections: List[str] = None) -> Dict:
    """
    Prepare entire paper data for both BART and Longformer summarization
    
    Args:
        paper_data (Dict): Output from parse_pdf_with_grobid()
        bart_sections (List[str]): Sections to chunk for BART (one-line/bullet summaries)
        longformer_sections (List[str]): Sections to chunk for Longformer (deep summaries)
    
    Returns:
        Dict: Structured data ready for summarization
    """
    # Default sections for different models
    if bart_sections is None:
        bart_sections = ["Introduction", "Conclusion", "Abstract"]
    
    if longformer_sections is None:
        longformer_sections = ["Model Architecture", "Experiments", "Results", "Background"]
    
    # Clean metadata
    cleaned_metadata = {
        "title": clean_text_comprehensive(paper_data["metadata"]["title"]),
        "authors": paper_data["metadata"]["authors"],
        "abstract": clean_text_comprehensive(paper_data["metadata"]["abstract"])
    }
    
    # Prepare chunked sections
    bart_chunks = {}
    longformer_chunks = {}
    
    # Chunk sections for BART (short summaries)
    for section_name in bart_sections:
        if section_name in paper_data["sections"]:
            section_chunks = chunk_sections_for_summarization({section_name: paper_data["sections"][section_name]}, "bart")
            bart_chunks.update(section_chunks)
    
    # Chunk sections for Longformer (deep summaries)  
    for section_name in longformer_sections:
        if section_name in paper_data["sections"]:
            section_chunks = chunk_sections_for_summarization({section_name: paper_data["sections"][section_name]}, "longformer")
            longformer_chunks.update(section_chunks)
    
    return {
        "metadata": cleaned_metadata,
        "bart_chunks": bart_chunks,
        "longformer_chunks": longformer_chunks,
        "original_sections": paper_data["sections"]
    }


def test_chunking():
    """
    Test the chunking functions with sample data
    """
    # Sample long text for testing
    sample_text = """
    Recurrent neural networks, long short-term memory and gated recurrent units have been used for transformer architecture. The multi-head attention mechanism shows variables in mathematical notation. We see that input represents the mathematical foundation. This approach is similar to previous work in the field of natural language processing.
    
    The self-attention mechanism performs better than traditional approaches. These models achieve state-of-the-art results on various benchmarks. The computational complexity is reduced significantly compared to recurrent models. Training time is also improved due to parallelization capabilities.
    
    Experimental results demonstrate the effectiveness of the proposed method. We conducted extensive experiments on multiple datasets. The model achieves superior performance across different metrics. Statistical significance tests confirm the reliability of our findings.
    
    Future work will explore additional applications of this technology. We plan to investigate multi-modal extensions of the current framework. Potential improvements include better handling of long sequences and reduced memory requirements.
    """ * 3  # Repeat to make it longer
    
    print("=== TEXT CHUNKING TEST ===")
    print(f"Original text length: {len(sample_text.split())} words")
    
    # Test BART chunking
    print(f"\n--- BART Chunking (for one-line/bullet summaries) ---")
    bart_chunks = chunk_text_for_models(sample_text, "bart")
    print(f"Number of chunks: {len(bart_chunks)}")
    for i, chunk in enumerate(bart_chunks):
        print(f"Chunk {i+1}: {chunk['word_count']} words, {chunk['sentence_count']} sentences")
        print(f"Preview: {chunk['text'][:100]}...")
        print()
    
    # Test Longformer chunking
    print(f"\n--- Longformer Chunking (for deep summaries) ---")
    longformer_chunks = chunk_text_for_models(sample_text, "longformer")
    print(f"Number of chunks: {len(longformer_chunks)}")
    for i, chunk in enumerate(longformer_chunks):
        print(f"Chunk {i+1}: {chunk['word_count']} words, {chunk['sentence_count']} sentences")
        print(f"Preview: {chunk['text'][:100]}...")
        print()


def test_advanced_cleaning():
    """
    Test the advanced cleaning functions
    """
    # Sample problematic text with various issues
    sample_text = """Recurrent neural networks, long short-term memory  [ 1 ] and gated recurrent units ( Cho et al. , 2014 ) have been used for trans former architecture. The multi head attention mechanism in Figure 1 shows h t variables. We see that x ∈ R d represents the input. This is simi-lar to previous work... The self attention mechanism performs better than tra-ditional approaches."""
    
    print("=== ADVANCED CLEANING TEST ===")
    print(f"Original text:\n{sample_text}")
    print(f"\nLength: {len(sample_text)} characters")
    
    # Apply comprehensive cleaning
    cleaned = clean_text_comprehensive(sample_text)
    
    print(f"\nCleaned text:\n{cleaned}")
    print(f"\nLength: {len(cleaned)} characters")
    print(f"Difference: {len(sample_text) - len(cleaned)} characters removed")


def test_basic_cleaning():
    """
    Test function to see how the cleaning works with sample text
    """
    # Sample text from your actual extraction (with problems)
    sample_text = "Recurrent neural networks, long short-term memory  Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states h t , as a function of the previous hidden sta a..."
    
    print("=== BASIC TEXT CLEANING TEST ===")
    print(f"Original text:\n{sample_text}")
    print(f"\nLength: {len(sample_text)} characters")
    
    # Apply cleaning
    cleaned = clean_basic_text(sample_text)
    
    print(f"\nCleaned text:\n{cleaned}")
    print(f"\nLength: {len(cleaned)} characters")
    print(f"Difference: {len(sample_text) - len(cleaned)} characters removed")


def test_with_real_data():
    """
    Test with actual data from parse_pdf.py
    """
    # Import your parser
    sys.path.append(os.path.dirname(__file__))
    try:
        from parse_pdf import parse_pdf_with_grobid
        
        print("=== TESTING WITH REAL DATA ===")
        pdf_path = "papers/attention_is_all_you_need.pdf"
        output_dir = "output"
        
        # Get the parsed data
        result = parse_pdf_with_grobid(pdf_path, output_dir)
        
        # Test comprehensive pipeline with chunking
        section_name = "Introduction"
        if section_name in result['sections']:
            original_text = result['sections'][section_name]
            basic_cleaned = clean_basic_text(original_text)
            comprehensive_cleaned = clean_text_comprehensive(original_text)
            
            print(f"Section: {section_name}")
            print(f"Original length: {len(original_text)} characters ({len(original_text.split())} words)")
            print(f"Basic cleaned length: {len(basic_cleaned)} characters")
            print(f"Comprehensive cleaned length: {len(comprehensive_cleaned)} characters")
            print(f"Total reduction: {len(original_text) - len(comprehensive_cleaned)} characters")
            
            # Test chunking
            bart_chunks = chunk_text_for_models(comprehensive_cleaned, "bart")
            longformer_chunks = chunk_text_for_models(comprehensive_cleaned, "longformer")
            
            print(f"\nChunking Results:")
            print(f"BART chunks: {len(bart_chunks)} (avg {sum(c['word_count'] for c in bart_chunks)//len(bart_chunks) if bart_chunks else 0} words each)")
            print(f"Longformer chunks: {len(longformer_chunks)} (avg {sum(c['word_count'] for c in longformer_chunks)//len(longformer_chunks) if longformer_chunks else 0} words each)")
            
            print(f"\nFirst 150 characters of comprehensive cleaned text:")
            print(f"{comprehensive_cleaned[:150]}...")
            
            # Test full paper preparation
            print(f"\n--- Full Paper Preparation Test ---")
            prepared_data = prepare_paper_for_summarization(result)
            print(f"BART sections prepared: {list(prepared_data['bart_chunks'].keys())}")
            print(f"Longformer sections prepared: {list(prepared_data['longformer_chunks'].keys())}")
            
            total_bart_chunks = sum(len(chunks) for chunks in prepared_data['bart_chunks'].values())
            total_longformer_chunks = sum(len(chunks) for chunks in prepared_data['longformer_chunks'].values())
            print(f"Total BART chunks: {total_bart_chunks}")
            print(f"Total Longformer chunks: {total_longformer_chunks}")
        else:
            print(f"Section '{section_name}' not found")
            print(f"Available sections: {list(result['sections'].keys())}")
            
    except Exception as e:
        print(f"Error testing with real data: {e}")
        print("Make sure GROBID is running and parse_pdf.py works")


def clean_text_comprehensive(text: str) -> str:
    """
    Comprehensive text cleaning for research papers
    """
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common PDF artifacts
    text = re.sub(r'[0-9]+\s*$', '', text)  # Page numbers at end
    text = re.sub(r'^[0-9]+\s*', '', text)  # Page numbers at start
    text = re.sub(r'\f', ' ', text)  # Form feed characters
    text = re.sub(r'•', '-', text)  # Bullet points
    
    # Fix common spacing issues
    text = re.sub(r'\s+([.,:;!?])', r'\1', text)  # Space before punctuation
    text = re.sub(r'([.,:;!?])\s*([a-zA-Z])', r'\1 \2', text)  # Space after punctuation
    
    # Remove excessive punctuation
    text = re.sub(r'[.]{3,}', '...', text)  # Multiple dots
    text = re.sub(r'[-]{2,}', '--', text)  # Multiple dashes
    
    # Clean up citations and references
    text = re.sub(r'\[[\d,\s-]+\]', '', text)  # Remove citation numbers [1,2,3]
    text = re.sub(r'\([\d,\s-]+\)', '', text)  # Remove citation numbers (1,2,3)
    
    # Remove URLs and email addresses
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # Final cleanup
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text

def chunk_text_for_models(text: str, model_type: str) -> List[Dict[str, any]]:
    """
    Chunk text based on model token limits
    """
    # Approximate token limits (conservative estimates)
    max_tokens = {
        "bart": 900,      # BART can handle ~1024 tokens
        "longformer": 3500,  # Longformer can handle 4096 tokens
        "bert": 450       # BERT can handle 512 tokens
    }
    
    limit = max_tokens.get(model_type.lower(), 500)
    
    # Rough estimation: 1 token ≈ 4 characters for English text
    char_limit = limit * 4
    
    chunks = []
    
    if len(text) <= char_limit:
        chunks.append({
            "text": text,
            "chunk_id": 0,
            "tokens_estimate": len(text) // 4
        })
        return chunks
    
    # Split by sentences first
    sentences = split_into_sentences(text)
    
    current_chunk = ""
    chunk_id = 0
    
    for sentence in sentences:
        # If adding this sentence would exceed limit
        if len(current_chunk) + len(sentence) > char_limit:
            if current_chunk:  # Save current chunk
                chunks.append({
                    "text": current_chunk.strip(),
                    "chunk_id": chunk_id,
                    "tokens_estimate": len(current_chunk) // 4
                })
                chunk_id += 1
                current_chunk = sentence
            else:  # Sentence itself is too long, force split
                long_chunks = force_split_text(sentence, char_limit)
                for long_chunk in long_chunks:
                    chunks.append({
                        "text": long_chunk,
                        "chunk_id": chunk_id,
                        "tokens_estimate": len(long_chunk) // 4
                    })
                    chunk_id += 1
        else:
            current_chunk += (" " if current_chunk else "") + sentence
    
    # Add final chunk
    if current_chunk:
        chunks.append({
            "text": current_chunk.strip(),
            "chunk_id": chunk_id,
            "tokens_estimate": len(current_chunk) // 4
        })
    
    return chunks

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using simple rules
    """
    # Handle common abbreviations that shouldn't end sentences
    abbrevs = ['Dr.', 'Prof.', 'Mr.', 'Mrs.', 'Ms.', 'Ph.D.', 'M.D.', 'B.S.', 'M.S.', 
               'i.e.', 'e.g.', 'et al.', 'etc.', 'vs.', 'Fig.', 'Table']
    
    for abbrev in abbrevs:
        text = text.replace(abbrev, abbrev.replace('.', '<DOT>'))
    
    # Split on sentence ending punctuation
    sentences = re.split(r'[.!?]+\s+', text)
    
    # Restore dots in abbreviations
    sentences = [s.replace('<DOT>', '.') for s in sentences]
    
    # Filter out empty sentences and very short ones
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    return sentences

def force_split_text(text: str, char_limit: int) -> List[str]:
    """
    Force split text that's too long even for one chunk
    """
    chunks = []
    
    while len(text) > char_limit:
        # Find last space before limit
        split_point = text.rfind(' ', 0, char_limit)
        
        if split_point == -1:  # No space found, force split at limit
            split_point = char_limit
        
        chunks.append(text[:split_point])
        text = text[split_point:].lstrip()
    
    if text:  # Add remaining text
        chunks.append(text)
    
    return chunks


if __name__ == "__main__":
    # Run tests
    print("Step 1: Basic cleaning test")
    test_basic_cleaning()
    print("\n" + "="*50 + "\n")
    
    print("Step 2: Advanced cleaning test")
    test_advanced_cleaning()
    print("\n" + "="*50 + "\n")
    
    print("Step 3: Text chunking test")
    test_chunking()
    print("\n" + "="*50 + "\n")
    
    print("Step 4: Real data with comprehensive processing")
    test_with_real_data()
