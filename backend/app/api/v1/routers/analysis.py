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
    FAST = "fast"          # 30-60 seconds, abstract + 3-paragraph summary  
    ENHANCED = "enhanced"  # 120-300 seconds, professional structured analysis with figures


async def process_analysis_task(job_id: str, file_path: str, mode: str = "fast"):
    """
    Background task to run analysis
    """
    db = SessionLocal()
    try:
        # Update status to processing
        job_queue.update_job(db, job_id, JobStatus.PROCESSING)
        
        # Run analysis with selected mode
        if mode == "fast":
            result = await fast_analysis_service.analyze_paper_fast(file_path, job_id, mode="fast")
        elif mode == "enhanced":
            result = await enhanced_analysis_service.analyze_paper_enhanced(file_path, job_id, mode="enhanced")
        else:
            # Default to fast mode
            result = await fast_analysis_service.analyze_paper_fast(file_path, job_id, mode="fast")
        
        # Update status to completed
        job_queue.update_job(db, job_id, JobStatus.COMPLETED, result=result)
        
        # Try to index for chat (non-blocking)
        try:
            from app.services.chat_service import chat_service
            
            # Build paper data from result instead of cache
            # This ensures it works regardless of which analysis service was used
            if result and hasattr(result, 'metadata') and result.metadata:
                paper_data = {
                    "title": result.metadata.title if hasattr(result.metadata, 'title') else 'Unknown',
                    "abstract": result.original_abstract if hasattr(result, 'original_abstract') else '',
                    "sections": result.detailed_summary if hasattr(result, 'detailed_summary') else {}
                }
                
                logger.info(f"Indexing paper {job_id} for chat with {len(paper_data.get('sections', {}))} sections")
                success = chat_service.index_paper_for_chat(job_id, paper_data)
                
                if success:
                    logger.info(f"Paper {job_id} indexed for chat successfully")
                else:
                    logger.warning(f"Failed to index paper {job_id} for chat")
            else:
                logger.warning(f"No valid result data for job {job_id} to index for chat")
        except Exception as e:
            import traceback
            logger.error(f"Chat indexing failed for {job_id}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Don't fail the entire analysis just because chat indexing failed
        
    except Exception as e:
        job_queue.update_job(db, job_id, JobStatus.FAILED, error=str(e))
    finally:
        db.close()

@router.post("/upload", response_model=JobResponse, summary="Upload PDF for analysis")
async def upload_paper(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: ProcessingMode = Query(ProcessingMode.FAST, description="Processing mode: fast (~60s, 3-paragraph summary) or enhanced (~300s, professional analysis)"),
    db: Session = Depends(get_db),
    # current_user = Depends(deps.get_current_user)  # Temporarily disabled for testing
):
    """
    Upload a PDF paper and start background analysis with configurable processing mode.
    
    Processing Modes:
    - **fast**: 30-60 seconds, original abstract + 3-paragraph summary covering problem, solution, technology, and conclusion
    - **enhanced**: 120-300 seconds, professional structured analysis with figures, glossary, hierarchical TOC, and section summaries
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
    Get information about available processing modes.
    
    Two modes available:
    - fast: Quick 3-paragraph summary (abstract + problem + solution + conclusion)
    - enhanced: Professional structured analysis with figures and glossary
    """
    return {
        "modes": {
            "fast": {
                "name": "Fast Mode",
                "estimated_time": "30-60 seconds",
                "description": "Original abstract + 3-paragraph summary in simple language covering: problem statement, technology/solution, implementation & conclusion",
                "features": [
                    "Original abstract",
                    "3-paragraph summary (~50 lines)",
                    "Problem statement",
                    "Technology and solution",
                    "Implementation details",
                    "Conclusions"
                ],
                "ai_models": False,
                "recommended_for": "Quick overview, fast processing"
            },
            "enhanced": {
                "name": "Enhanced Professional Mode",
                "estimated_time": "120-300 seconds",
                "description": "Comprehensive professional analysis with hierarchical table of contents, section summaries, figure extraction, and technical glossary",
                "features": [
                    "Executive summary",
                    "Research analysis breakdown",
                    "Technical details",
                    "Figure extraction with captions",
                    "Technical glossary (20+ concepts)",
                    "Hierarchical table of contents",
                    "Section-level summaries",
                    "Professional HTML output"
                ],
                "ai_models": "PRIMERA (scientific papers)",
                "recommended_for": "Professional reports, detailed analysis, presentations"
            }
        },
        "default_mode": "fast",
        "supported_modes": ["fast", "enhanced"]
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

