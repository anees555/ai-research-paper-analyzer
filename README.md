# AI Research Paper Analyzer

Comprehensive research paper analysis platform combining automated summarization, semantic document search, and web-based API access. Built for researchers, academics, and data scientists who need efficient paper processing and intelligent document retrieval.

## Overview

This project provides a complete research paper processing pipeline that transforms PDF documents into structured, searchable knowledge. The system combines traditional document processing with modern AI techniques to deliver both quick insights and deep analysis.

### Core Capabilities

The platform processes research papers and generates multiple types of output:
- **Quick AI summaries** using BART model for immediate insights
- **Comprehensive analysis** using Longformer model for detailed understanding
- **Semantic document search** with vector embeddings and similarity matching
- **Question-answering system** leveraging RAG (Retrieval-Augmented Generation)
- **Rich metadata extraction** including authors, sections, and citation analysis
- **RESTful API** for web application integration

### Processing Reliability

Built with production reliability in mind, the system employs hybrid processing that combines GROBID structured extraction with PyPDF2 fallback processing, ensuring consistent results across diverse document formats and quality levels.

## Features

### Document Processing
- **Hybrid PDF Processing**: Primary GROBID extraction with PyPDF2 fallback
- **Batch Processing**: Handle multiple documents simultaneously
- **Metadata Extraction**: Authors, sections, citations, and structural elements
- **Format Preservation**: Maintains document structure and formatting context

### AI-Powered Analysis
- **Multi-Model Summarization**: BART for concise summaries, Longformer for detailed analysis
- **Semantic Understanding**: Vector embeddings for document similarity and search
- **Question Answering**: RAG-based system for document querying
- **Contextual Retrieval**: Intelligent chunk selection and context assembly

### Web Integration
- **FastAPI Backend**: Production-ready API with async processing
- **Task Management**: Background processing with status tracking
- **File Upload**: Multi-format document upload and processing
- **Response Formatting**: Structured JSON output for easy integration

### Semantic Search System
- **Vector Storage**: ChromaDB for efficient similarity search
- **Embedding Generation**: Sentence transformers for semantic understanding
- **Interactive Queries**: Natural language document search and filtering
- **Context Assembly**: Intelligent retrieval of relevant document sections

## Quick Start

### Environment Setup

1. **Create and activate virtual environment**:
   ```bash
   # Use the provided activation script
   .\activate_env.ps1
   # Or manually activate
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**:
   ```bash
   python -m pip install -r requirements.txt
   ```

### Basic Usage

1. **Start GROBID server**:
   ```bash
   docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0
   ```

2. **Process documents**:
   ```bash
   # Add PDF files to data/papers/ directory
   python main.py
   ```

3. **Run semantic search**:
   ```bash
   python semantic-document-search/app.py
   ```

4. **Start web API**:
   ```bash
   cd backend
   python main.py
   ```

## Project Structure

```
├── src/                           # Core processing modules
│   ├── hybrid_summary_generator.py    # Main document processor
│   ├── ai_summary_generator.py        # AI summarization engine
│   └── scripts/                      # Utility and processing scripts
├── semantic-document-search/          # Semantic search system
│   ├── app.py                        # Main search application
│   ├── src/                          # Search processing modules
│   └── document_qa.py                # Question-answering interface
├── backend/                          # Web API and server components
│   ├── main.py                       # FastAPI application
│   └── backend/                      # Additional server utilities
├── data/                            # Input and training data
│   ├── papers/                       # PDF documents for processing
│   └── datasets/                     # Training and reference datasets  
├── outputs/                         # Generated analysis results
│   └── hybrid_summaries/             # AI-generated summaries and reports
├── models/                          # AI model cache and storage
├── requirements.txt                 # Unified dependency specification
└── main.py                         # Primary application entry point
```

## Usage Examples

### Document Analysis
```bash
# Process all PDFs in data/papers/ directory
python main.py
```

### Semantic Search
```bash
# Interactive document search
python semantic-document-search/app.py

# RAG question-answering mode
python semantic-document-search/app.py qa
```

### API Integration
```bash
# Start development server
cd backend
python main.py

# API will be available at http://localhost:8000
# Documentation at http://localhost:8000/docs
```

## System Requirements

- Python 3.13+ recommended
- Docker for GROBID server
- 8GB+ RAM recommended for large document processing
- GPU acceleration optional but recommended for faster processing

## Development Status

The system is actively maintained and production-ready. Current implementation includes:
- Stable document processing pipeline with 100% success rate
- Deployed semantic search with vector storage
- Production FastAPI backend with async processing
- Comprehensive error handling and logging

## Contributing

This project welcomes contributions from the research and development community. Areas of active development include model optimization, additional document format support, and enhanced semantic search capabilities.

## Acknowledgments

Built using open source tools including GROBID for document structure extraction, Hugging Face transformers for AI processing, ChromaDB for vector storage, and FastAPI for web services. Special recognition to the broader research community for methodology and approach validation.