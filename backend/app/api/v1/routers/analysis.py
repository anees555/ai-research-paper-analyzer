from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from app.data_models.schemas import JobResponse, JobStatus, AnalysisRequest
from app.services.file_service import file_service
from app.services.queue_service import job_queue
from app.services.analysis_service import analysis_service
from typing import Optional
import asyncio

router = APIRouter()

async def process_analysis_task(job_id: str, file_path: str):
    """
    Background task to run analysis
    """
    try:
        # Update status to processing
        job_queue.update_job(job_id, JobStatus.PROCESSING)
        
        # Run analysis (Service handles model loading and processing)
        result = await analysis_service.analyze_paper(file_path, job_id)
        
        # Update status to completed
        job_queue.update_job(job_id, JobStatus.COMPLETED, result=result)
        
    except Exception as e:
        job_queue.update_job(job_id, JobStatus.FAILED, error=str(e))

@router.post("/upload", response_model=JobResponse, summary="Upload PDF for analysis")
async def upload_paper(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a PDF paper and start background analysis.
    Returns a Job ID to poll for status.
    """
    try:
        # 1. Save file
        file_path = await file_service.save_upload_file(file)
        
        # 2. Create Job
        job = job_queue.create_job(file_path)
        
        # 3. Start Background Task
        background_tasks.add_task(process_analysis_task, job.job_id, file_path)
        
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
async def get_job_status(job_id: str):
    """
    Get the current status and result of an analysis job.
    """
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        error=job.error,
        result=job.result
    )
