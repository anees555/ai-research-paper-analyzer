from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Depends, Query
from app.data_models.schemas import JobResponse, JobStatus, AnalysisRequest
from app.services.file_service import file_service
from app.services.queue_service import job_queue
from app.services.analysis_service import analysis_service
from app.services.fast_analysis_service import fast_analysis_service
from app.services.enhanced_analysis_service import enhanced_analysis_service
from app.core.database import get_db, SessionLocal
from sqlalchemy.orm import Session
from app.api import deps
from typing import Optional
import asyncio
import logging
from enum import Enum

logger = logging.getLogger(__name__)
router = APIRouter()

class ProcessingMode(str, Enum):
    FAST = "fast"          # 30-60 seconds, basic analysis
    BALANCED = "balanced"  # 60-120 seconds, AI-powered but optimized
    COMPREHENSIVE = "comprehensive"  # 120-300 seconds, full analysis
    ENHANCED = "enhanced"  # Professional structured analysis with figures

async def process_analysis_task(job_id: str, file_path: str, mode: str = "balanced"):
    """
    Background task to run analysis
    """
    db = SessionLocal()
    try:
        # Update status to processing
        job_queue.update_job(db, job_id, JobStatus.PROCESSING)
        
        # Run analysis with selected mode
        if mode in ["fast", "balanced"]:
            result = await fast_analysis_service.analyze_paper_fast(file_path, job_id, mode)
        elif mode == "enhanced":
            result = await enhanced_analysis_service.analyze_paper_enhanced(file_path, job_id, mode)
        else:
            # Use original service for comprehensive mode  
            result = await analysis_service.analyze_paper(file_path, job_id)
        
        # Update status to completed
        job_queue.update_job(db, job_id, JobStatus.COMPLETED, result=result)
        
        # Try to index for chat (non-blocking)
        try:
            from app.services.chat_service import chat_service
            # Get the stored paper data from analysis service
            paper_data = analysis_service.get_paper_data(job_id)
            if paper_data:
                success = chat_service.index_paper_for_chat(job_id, paper_data)
                if success:
                    logger.info(f"Paper {job_id} indexed for chat successfully")
                else:
                    logger.warning(f"Failed to index paper {job_id} for chat")
            else:
                logger.warning(f"No paper data found for job {job_id} to index for chat")
        except Exception as e:
            logger.error(f"Chat indexing failed for {job_id}: {e}")
            # Don't fail the entire analysis just because chat indexing failed
        
    except Exception as e:
        job_queue.update_job(db, job_id, JobStatus.FAILED, error=str(e))
    finally:
        db.close()

@router.post("/upload", response_model=JobResponse, summary="Upload PDF for analysis")
async def upload_paper(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: ProcessingMode = Query(ProcessingMode.BALANCED, description="Processing mode: fast (~60s), balanced (~120s), comprehensive (~300s)"),
    db: Session = Depends(get_db),
    # current_user = Depends(deps.get_current_user)  # Temporarily disabled for testing
):
    """
    Upload a PDF paper and start background analysis with configurable processing mode.
    
    Processing Modes:
    - **fast**: 30-60 seconds, extractive summaries, basic metadata
    - **balanced**: 60-120 seconds, lightweight AI models, good quality/speed trade-off
    - **comprehensive**: 120-300 seconds, full AI analysis, highest quality
    """
    try:
        # 1. Save file
        file_path = await file_service.save_upload_file(file)
        
        # 2. Create Job
        job = job_queue.create_job(db, file_path, user_id=None)  # Use None for guest uploads
        
        # 3. Start Background Task with selected mode
        background_tasks.add_task(process_analysis_task, job.job_id, file_path, mode.value)
        
        return JobResponse(
            job_id=job.job_id,
            status=job.status,
            created_at=job.created_at
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{job_id}", response_model=JobResponse, summary="Check analysis status")
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """
    Get the current status and result of an analysis job.
    """
    job = job_queue.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        error=job.error,
        result=job.result
    )

@router.post("/analyze-instant", summary="Instant analysis for fast mode")
async def analyze_paper_instant(
    file: UploadFile = File(...),
    mode: ProcessingMode = Query(ProcessingMode.FAST, description="Processing mode - only 'fast' and 'balanced' supported for instant analysis"),
    current_user = Depends(deps.get_current_user)
):
    """
    Perform instant paper analysis without background processing.
    
    **Only supports 'fast' and 'balanced' modes for quick response.**
    
    - **fast**: Returns in 30-60 seconds with extractive analysis
    - **balanced**: Returns in 60-120 seconds with lightweight AI analysis  
    - **comprehensive**: Use /upload endpoint for background processing
    """
    if mode == ProcessingMode.COMPREHENSIVE:
        raise HTTPException(
            status_code=400, 
            detail="Comprehensive mode requires background processing. Use /upload endpoint."
        )
    
    try:
        # 1. Save file temporarily
        file_path = await file_service.save_upload_file(file)
        
        # 2. Generate unique job ID for tracking
        import uuid
        job_id = str(uuid.uuid4())
        
        # 3. Perform instant analysis
        result = await fast_analysis_service.analyze_paper_fast(file_path, job_id, mode.value)
        
        # 4. Clean up temporary file
        import os
        try:
            os.remove(file_path)
        except:
            pass  # Ignore cleanup errors
        
        return {
            "status": "completed",
            "processing_mode": mode.value,
            "analysis_result": result,
            "processing_info": {
                "instant_processing": True,
                "estimated_time": "30-60s" if mode == ProcessingMode.FAST else "60-120s"
            }
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Instant analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/modes", summary="Get available processing modes")
async def get_processing_modes():
    """
    Get information about available processing modes and their characteristics.
    """
    return {
        "modes": {
            "fast": {
                "name": "Fast Processing",
                "estimated_time": "30-60 seconds",
                "description": "Extractive summaries, basic metadata, rule-based analysis",
                "features": ["Quick summary", "Basic section analysis", "Metadata extraction"],
                "ai_models": False,
                "recommended_for": "Quick overview, large batches"
            },
            "balanced": {
                "name": "Balanced Processing", 
                "estimated_time": "60-120 seconds",
                "description": "Lightweight AI models, good quality/speed trade-off",
                "features": ["AI summaries", "Section analysis", "Key insights", "Concurrent processing"],
                "ai_models": "DistilBART or T5-small",
                "recommended_for": "Most use cases, daily workflow"
            },
            "comprehensive": {
                "name": "Comprehensive Analysis",
                "estimated_time": "120-300 seconds", 
                "description": "Full AI analysis with BART-Large, detailed insights",
                "features": ["Full AI analysis", "Detailed summaries", "Deep insights", "Research quality"],
                "ai_models": "BART-Large-CNN",
                "recommended_for": "Research papers, detailed analysis"
            },
            "enhanced": {
                "name": "Enhanced Professional Analysis",
                "estimated_time": "90-150 seconds",
                "description": "Professional structured summaries with figures and glossary",
                "features": ["HTML templates", "Figure extraction", "Technical glossary", "Clean output"],
                "ai_models": "BART + Custom processing",
                "recommended_for": "Professional reports, presentations"
            }
        },
        "default_mode": "enhanced",
        "instant_analysis_supported": ["fast", "balanced", "enhanced"],
        "background_processing_required": ["comprehensive"]
    }

@router.post("/enhanced", response_model=JobResponse)
async def upload_and_analyze_enhanced(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload and analyze a paper using enhanced professional analysis with structured output,
    figure extraction, and technical glossary.
    """
    try:
        # Validate file
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file
        file_path = await file_service.save_upload_file(file)
        
        # Create job
        job_id = job_queue.create_job(db, file.filename, JobStatus.QUEUED)
        
        # Start enhanced analysis in background
        background_tasks.add_task(process_analysis_task, job_id, file_path, "enhanced")
        
        return JobResponse(job_id=job_id, status=JobStatus.QUEUED, message="Enhanced analysis started")
        
    except Exception as e:
        logger.error(f"Enhanced analysis upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced analysis failed: {str(e)}")

@router.post("/professional", response_model=JobResponse)  
async def upload_and_analyze_professional(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload and analyze a paper using professional analysis optimized for business presentations
    and clean, human-like output without AI artifacts.
    """
    try:
        # Validate file
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file  
        file_path = await file_service.save_upload_file(file)
        
        # Create job
        job_id = job_queue.create_job(db, file.filename, JobStatus.QUEUED)
        
        # Start enhanced analysis in background (enhanced mode is professional)
        background_tasks.add_task(process_analysis_task, job_id, file_path, "enhanced")
        
        return JobResponse(job_id=job_id, status=JobStatus.QUEUED, 
                          message="Professional analysis started - clean, structured output without AI artifacts")
        
    except Exception as e:
        logger.error(f"Professional analysis upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Professional analysis failed: {str(e)}")

