"""
Minimal backend for testing - just health endpoint
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI Research Paper Analyzer - Debug",
    version="1.0.0", 
    description="Debug version",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Debug backend running", "status": "ok"}

@app.get("/api/v1/health")
async def health():
    return {"status": "healthy", "message": "Debug backend is running"}

@app.post("/api/v1/chat/ask")
async def debug_chat_ask(request: dict):
    """Debug chat endpoint that returns the error message format"""
    return {
        "message": "I encountered an error while processing your question. Please try again.",
        "sources": [],
        "confidence": 0,
        "timestamp": "2026-01-14T08:15:59.081975"
    }