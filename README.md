# AI Research Paper Analyzer

A comprehensive AI-powered research paper analysis system that processes academic PDFs to generate intelligent summaries, extract metadata, and enable natural language conversations with research papers.

## Overview

This application leverages state-of-the-art machine learning models and natural language processing techniques to transform complex academic papers into accessible, structured summaries. The system combines GROBID-based PDF parsing, transformer-based summarization models, and retrieval-augmented generation (RAG) for interactive paper exploration.

## Key Features

### Intelligent PDF Processing
- Academic structure extraction using GROBID service
- Automatic section detection and hierarchy preservation
- Metadata extraction including authors, affiliations, and citations
- Figure and table caption analysis

### AI-Powered Summarization
- Multiple summarization models: DistilBART, BART-Large-CNN, and Longformer
- Executive summary generation with grammar correction
- Section-wise detailed breakdowns
- Key contribution and results extraction
- Technical term glossary generation

### Interactive Chat Interface
- RAG-powered conversations with uploaded papers
- Context-aware responses using ChromaDB vector storage
- Groq API integration for enhanced language understanding
- Chat history management and session persistence

### Professional Analysis Output
- Comprehensive HTML summaries with proper formatting
- Bold headings and hierarchical text sizing
- Research methodology and approach analysis
- Practical implications and findings
- Quality scoring and completeness metrics

### Performance Optimization
- Model caching and lazy loading
- Memory-efficient operation with DistilBART preloading
- Async processing for large documents
- Background model initialization

## Technology Stack

### Backend
- **Framework**: FastAPI (Python 3.13+)
- **Database**: PostgreSQL with SQLAlchemy ORM
- **PDF Processing**: GROBID (Docker service)
- **AI Models**: 
  - Transformers (HuggingFace)
  - DistilBART for fast summarization
  - BART-Large-CNN for high-quality summaries
  - Longformer/LED for long documents
- **Vector Database**: ChromaDB for semantic search
- **LLM Integration**: Groq API
- **Authentication**: JWT-based security

### Frontend
- **Framework**: Next.js 16.1.1 with App Router
- **Language**: TypeScript
- **Styling**: TailwindCSS
- **State Management**: React Query
- **HTTP Client**: Axios

## Installation

### Prerequisites
- Python 3.8 or higher (Python 3.13+ recommended)
- Node.js 18 or higher (Node.js 25+ recommended)
- Docker for running GROBID service
- Git for version control

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd research_summary_project
```

### Step 2: Backend Setup
```bash
# Create and activate virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Environment Configuration
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
DATABASE_URL=postgresql://user:password@localhost:5432/research_db
JWT_SECRET=your_secret_key_here
```

### Step 4: Start GROBID Service
```bash
docker pull lfoppiano/grobid:0.8.0
docker run -d -p 8070:8070 --name grobid-research lfoppiano/grobid:0.8.0
```

### Step 5: Frontend Setup
```bash
cd frontend
npm install
```

## Running the Application

### Development Mode

**Backend**:
```bash
# From project root
cd backend
python main.py
# API runs on http://localhost:8003
```

**Frontend**:
```bash
# From project root
cd frontend
npm run dev
# Frontend runs on http://localhost:3000
```

### Quick Start (After Initial Setup)

**Windows PowerShell**:
```powershell
.\.venv\Scripts\Activate.ps1
docker start grobid-research
cd backend; python main.py
# In new terminal
cd frontend; npm run dev
```

**Linux/Mac**:
```bash
source .venv/bin/activate && docker start grobid-research
cd backend && python main.py &
cd frontend && npm run dev
```

## API Endpoints

### Analysis Endpoints
- `POST /api/analysis/upload` - Upload and analyze PDF
- `GET /api/analysis/{job_id}` - Get analysis results
- `GET /api/analysis/list` - List all analyses
- `DELETE /api/analysis/{job_id}` - Delete analysis

### Chat Endpoints
- `POST /api/chat/message` - Send chat message
- `GET /api/chat/history/{paper_id}` - Get chat history
- `DELETE /api/chat/{session_id}` - Clear chat session

### System Endpoints
- `GET /api/health` - Health check
- `GET /api/isalive` - Service status

## Project Structure

```
research_summary_project/
├── backend/
│   ├── app/
│   │   ├── api/          # API routes
│   │   ├── core/         # Configuration
│   │   ├── data_models/  # Pydantic models
│   │   └── services/     # Business logic
│   ├── data/             # Uploads and databases
│   └── main.py           # Application entry
├── frontend/
│   └── src/
│       ├── app/          # Next.js pages
│       ├── components/   # React components
│       ├── contexts/     # State contexts
│       └── types/        # TypeScript types
├── src/
│   └── scripts/          # Utility scripts
└── .env                  # Environment variables
```

## Recent Enhancements

### v2.0 - Professional Analysis System
- Removed all decorative icons from codebase for professional appearance
- Implemented grammar correction in AI-generated summaries
- Added sentence redundancy removal for cleaner output
- Enhanced section summary extraction with 2000-character limit
- Improved formatting with hierarchical font sizes and bold headings

### Content Quality Improvements
- Real content extraction from abstracts and introductions
- Fallback mechanisms to avoid generic placeholder text
- Minimum content length validation (50 characters per section)
- Short section title filtering (removes fragments like "The", "A")
- Full text extraction including nested XML elements

### UI/UX Enhancements
- Professional heading hierarchy (24px main, 20px sub, 16px content)
- Bold formatting for all section headings
- Increased line heights for better readability
- Responsive detailed breakdown section with proper spacing

### Performance Optimizations
- Memory-efficient model loading (DistilBART-only preloading)
- On-demand loading for larger models (BART-Large-CNN, Longformer)
- Background async model initialization
- Graceful degradation for memory-constrained systems

## Configuration

### Model Settings
The system supports three summarization quality levels:
- **Fast**: DistilBART (preloaded, ~5 seconds startup)
- **Balanced**: BART-Large-CNN (on-demand, high quality)
- **Long Documents**: Longformer/LED (on-demand, 16K tokens)

### Processing Modes
- **Enhanced Mode**: Full AI analysis with all features
- **Fast Mode**: Quick processing with optimized pipeline
- **Basic Mode**: Rule-based extraction only

## Troubleshooting

### Common Issues

**GROBID Connection Errors**:
```bash
# Verify GROBID is running
curl http://localhost:8070/api/isalive

# Restart GROBID if needed
docker restart grobid-research
```

**Memory Errors (Paging File Too Small)**:
- System uses DistilBART by default for memory efficiency
- Larger models (BART-Large-CNN) require sufficient virtual memory
- Increase Windows paging file or use default DistilBART

**Model Loading Timeouts**:
- First run downloads models (may take several minutes)
- Models cached in `~/.cache/huggingface/`
- Subsequent runs use cached models

**Frontend Connection Issues**:
- Ensure backend is running on port 8003
- Check CORS settings in backend configuration
- Verify environment variables are loaded

## Contributing

Contributions are welcome. Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Make changes with clear commit messages
4. Ensure code passes linting and type checks
5. Submit pull request with detailed description

## License

This project is licensed under the MIT License.

## Support

For issues, questions, or feature requests, please open an issue on the repository issue tracker.

## Acknowledgments

- GROBID for academic PDF parsing
- HuggingFace Transformers for NLP models
- FastAPI and Next.js communities for excellent frameworks
- Groq for LLM API services
