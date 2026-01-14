import uuid
from typing import Dict, Optional
from app.data_models.job import Job
from app.data_models.schemas import JobStatus, AnalysisResult
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

import uuid
from typing import Dict, Optional
from app.data_models.models import Job
from app.data_models.schemas import JobStatus, AnalysisResult
from datetime import datetime
from sqlalchemy.orm import Session
import logging

logger = logging.getLogger(__name__)

class JobQueueService:
    def create_job(self, db: Session, file_path: str, user_id: Optional[str] = None) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id, 
            file_path=file_path, 
            status=JobStatus.PENDING,
            user_id=user_id
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        logger.info(f"Created job {job_id}")
        return job

    def get_job(self, db: Session, job_id: str) -> Optional[Job]:
        return db.query(Job).filter(Job.job_id == job_id).first()

    def update_job(self, db: Session, job_id: str, status: JobStatus, result: Optional[AnalysisResult] = None, error: Optional[str] = None):
        job = self.get_job(db, job_id)
        if job:
            job.status = status
            job.updated_at = datetime.utcnow()
            if result:
                # Store as dict/json
                job.result = result.dict() if hasattr(result, 'dict') else result
            if error:
                job.error = error
            db.add(job)
            db.commit()
            db.refresh(job)
            logger.info(f"Updated job {job_id} to status {status}")

job_queue = JobQueueService()

