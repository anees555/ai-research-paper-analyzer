import uuid
from typing import Dict, Optional
from app.data_models.job import Job
from app.data_models.schemas import JobStatus, AnalysisResult
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class JobQueueService:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}

    def create_job(self, file_path: str) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(job_id=job_id, file_path=file_path, status=JobStatus.PENDING)
        self.jobs[job_id] = job
        logger.info(f"Created job {job_id}")
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def update_job(self, job_id: str, status: JobStatus, result: Optional[AnalysisResult] = None, error: Optional[str] = None):
        job = self.jobs.get(job_id)
        if job:
            job.status = status
            job.updated_at = datetime.now()
            if result:
                job.result = result
            if error:
                job.error = error
            logger.info(f"Updated job {job_id} to status {status}")

job_queue = JobQueueService()
