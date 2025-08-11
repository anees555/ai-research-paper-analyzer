#!/usr/bin/env python3
"""
FastAPI Backend for Research Paper Summarizer
Wraps HybridSummaryGenerator for web API access
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import uuid
import shutil
import asyncio
from typing import Dict, Any, Optional
import sys
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import with error handling
try:
    from hybrid_summary_generator import HybridSummaryGenerator

    HYBRID_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Warning: Could not import HybridSummaryGenerator: {e}")
    print("📄 API will run in limited mode")
    HYBRID_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="AI Research Paper Summarizer API",
    description="Upload PDFs and get AI-powered multi-level summaries",
    version="1.0.0",
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
UPLOAD_DIR = "backend/uploads"
OUTPUT_DIR = "backend/outputs"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB limit

# Storage for background task results
processing_results: Dict[str, Dict[str, Any]] = {}

# Initialize directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize the summary generator (this takes time on first load)
print("🚀 Initializing AI models...")
if HYBRID_AVAILABLE:
    try:
        generator = HybridSummaryGenerator(output_dir=OUTPUT_DIR)
        print("✅ AI models loaded and ready!")
    except Exception as e:
        print(f"❌ Failed to initialize HybridSummaryGenerator: {e}")
        print("📄 Running in fallback mode")
        generator = None
else:
    print("❌ HybridSummaryGenerator not available")
    generator = None


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Research Paper Summarizer API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload-pdf/",
            "process": "/process/{task_id}/",
            "status": "/status/{task_id}/",
            "health": "/health/",
        },
        "models_loaded": {
            "grobid_available": generator.grobid_processor is not None
            if generator
            else False,
            "bart_loaded": generator.bart_summarizer is not None
            if generator
            else False,
            "longformer_loaded": generator.longformer_summarizer is not None
            if generator
            else False,
            "hybrid_generator_available": generator is not None,
        },
    }


@app.get("/health/")
async def health_check():
    """Comprehensive health check"""
    grobid_healthy = False
    if generator and generator.grobid_processor:
        try:
            grobid_healthy = generator.grobid_processor.check_grobid_health()
        except Exception:
            pass

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "api": "running",
            "hybrid_generator": "available" if generator else "unavailable",
            "grobid_server": "healthy" if grobid_healthy else "unavailable",
            "bart_model": "loaded"
            if generator and generator.bart_summarizer
            else "unavailable",
            "longformer_model": "loaded"
            if generator and generator.longformer_summarizer
            else "unavailable",
        },
        "processing_stats": generator.processing_stats if generator else {},
        "fallback_available": True,  # PyPDF2 is always available
    }


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file for processing
    Returns a task_id for tracking processing status
    """
    # Validate file
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {MAX_FILE_SIZE / 1024 / 1024}MB",
        )

    # Generate unique task ID
    task_id = str(uuid.uuid4())

    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, f"{task_id}.pdf")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Initialize processing status
    processing_results[task_id] = {
        "task_id": task_id,
        "filename": file.filename,
        "status": "uploaded",
        "uploaded_at": datetime.now().isoformat(),
        "file_path": file_path,
        "file_size": os.path.getsize(file_path),
    }

    return {
        "task_id": task_id,
        "filename": file.filename,
        "status": "uploaded",
        "message": "File uploaded successfully. Use /process/{task_id}/ to start processing.",
    }


async def process_paper_background(task_id: str):
    """Background processing function"""
    try:
        # Update status
        processing_results[task_id].update(
            {"status": "processing", "started_at": datetime.now().isoformat()}
        )

        file_path = processing_results[task_id]["file_path"]

        if not generator:
            raise Exception("HybridSummaryGenerator not available - check dependencies")

        # Process the paper using your hybrid generator
        print(f"📄 Starting background processing for {task_id}")
        result = generator.process_single_paper(file_path)

        # Update with results
        processing_results[task_id].update(
            {
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "result": result,
            }
        )

        print(f"✅ Background processing completed for {task_id}")

    except Exception as e:
        print(f"❌ Background processing failed for {task_id}: {e}")
        processing_results[task_id].update(
            {
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.now().isoformat(),
            }
        )


@app.post("/process/{task_id}/")
async def process_paper(task_id: str, background_tasks: BackgroundTasks):
    """
    Start processing an uploaded PDF
    Processing happens in background, check status with /status/{task_id}/
    """
    if task_id not in processing_results:
        raise HTTPException(status_code=404, detail="Task ID not found")

    task_info = processing_results[task_id]

    if task_info["status"] != "uploaded":
        raise HTTPException(
            status_code=400, detail=f"Task is already {task_info['status']}"
        )

    # Start background processing
    background_tasks.add_task(process_paper_background, task_id)

    return {
        "task_id": task_id,
        "status": "processing_started",
        "message": "Processing started in background. Check status with /status/{task_id}/",
    }


@app.get("/status/{task_id}/")
async def get_processing_status(task_id: str):
    """Get the processing status and results for a task"""
    if task_id not in processing_results:
        raise HTTPException(status_code=404, detail="Task ID not found")

    task_info = processing_results[task_id]

    # Return appropriate response based on status
    response = {
        "task_id": task_id,
        "filename": task_info.get("filename"),
        "status": task_info["status"],
        "uploaded_at": task_info.get("uploaded_at"),
    }

    if task_info["status"] == "processing":
        response.update(
            {
                "started_at": task_info.get("started_at"),
                "message": "Processing in progress...",
            }
        )

    elif task_info["status"] == "completed":
        response.update(
            {
                "started_at": task_info.get("started_at"),
                "completed_at": task_info.get("completed_at"),
                "result": task_info.get("result"),
                "message": "Processing completed successfully",
            }
        )

    elif task_info["status"] == "failed":
        response.update(
            {
                "failed_at": task_info.get("failed_at"),
                "error": task_info.get("error"),
                "message": "Processing failed",
            }
        )
    print(response)

    return response


@app.post("/process-sync/{task_id}/")
async def process_paper_sync(task_id: str):
    """
    Synchronous processing endpoint (blocks until complete)
    Use this for immediate results, but may timeout on large files
    """
    if task_id not in processing_results:
        raise HTTPException(status_code=404, detail="Task ID not found")

    task_info = processing_results[task_id]

    if task_info["status"] != "uploaded":
        raise HTTPException(
            status_code=400, detail=f"Task is already {task_info['status']}"
        )

    try:
        # Update status
        processing_results[task_id].update(
            {"status": "processing", "started_at": datetime.now().isoformat()}
        )

        file_path = task_info["file_path"]

        if not generator:
            raise Exception("HybridSummaryGenerator not available - check dependencies")

        # Process synchronously (this blocks)
        result = generator.process_single_paper(file_path)

        # Update with results
        processing_results[task_id].update(
            {
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "result": result,
            }
        )

        return {"task_id": task_id, "status": "completed", "result": result}

    except Exception as e:
        processing_results[task_id].update(
            {
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.now().isoformat(),
            }
        )

        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/tasks/")
async def list_all_tasks():
    """List all processing tasks and their status"""
    return {
        "total_tasks": len(processing_results),
        "tasks": [
            {
                "task_id": task_id,
                "filename": info.get("filename"),
                "status": info["status"],
                "uploaded_at": info.get("uploaded_at"),
            }
            for task_id, info in processing_results.items()
        ],
    }


@app.delete("/tasks/{task_id}/")
async def delete_task(task_id: str):
    """Delete a task and its associated files"""
    if task_id not in processing_results:
        raise HTTPException(status_code=404, detail="Task ID not found")

    task_info = processing_results[task_id]

    # Delete uploaded file if exists
    file_path = task_info.get("file_path")
    if file_path and os.path.exists(file_path):
        os.remove(file_path)

    # Remove from memory
    del processing_results[task_id]

    return {"message": f"Task {task_id} deleted successfully"}


if __name__ == "__main__":
    print("🚀 Starting AI Research Paper Summarizer API...")
    print("📄 Ready to process PDF uploads!")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable for development
        log_level="info",
    )
