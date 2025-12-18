# 🎯 AI Research Paper Analyzer

AI-powered research paper summarization using BART and Longformer models with hybrid PDF processing.

## 📋 **Overview**

This project processes research papers (PDF format) and generates:
- **Quick AI summaries** using BART model (30-100 words)
- **Deep analysis** using Longformer model (200-400 words)  
- **Rich metadata extraction** (authors, sections, citations)
- **Hybrid processing** with 100% success rate (GROBID + PyPDF2 fallback)

**Current Status**: Backend fully functional with AI models loaded and tested.

## ✨ Features

- **AI Summaries**: BART (quick) + Longformer (deep analysis)
- **Hybrid Processing**: GROBID + PyPDF2 fallback (100% success rate)
- **Batch Processing**: Multiple papers simultaneously
- **Rich Metadata**: Authors, sections, citations extraction

## 🚀 Quick Start

1. **Setup Environment**:
   ```bash
   # Activate virtual environment
   majorenv\Scripts\activate

   # Install dependencies (if needed)
   pip install -r requirements.txt
   ```

2. **Start GROBID Server**:
   ```bash
   docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0
   ```

3. **Add PDFs & Run**:
   ```bash
   # Add PDF files to data/papers/ directory
   # Then run the main script
   python main.py
   ```

## 📁 Structure

```
├── src/                    # Source code
│   ├── hybrid_summary_generator.py
│   ├── ai_summary_generator.py  
│   └── scripts/           # Processing utilities
├── data/                   # Input data
│   ├── papers/            # PDF files to process
│   └── datasets/          # Training datasets
├── outputs/               # Generated summaries
├── models/                # AI model cache
└── main.py               # Run this to start
```
## 🧪 **Testing**

Current status: Backend fully functional with 100% success rate processing PDFs and generating AI summaries.

To test with sample papers:
```bash
# Add PDF files to data/papers/ directory
# Run the main script
python main.py
```

Check `outputs/hybrid_summaries/` for generated AI summaries in JSON format.

- **Anish Dahal** - *Initial work* - [GitHub](https://github.com/anees555)

## 🙏 **Acknowledgments**

- GROBID team for document processing
- Hugging Face for transformer models
- Open source community for tools and libraries