import uuid
from typing import Dict, Optional
from app.data_models.job import Job
from app.data_models.schemas import JobStatus, AnalysisResult
from datetime import datetime
import logging
import asyncio

logger = logging.getLogger(__name__)

import uuid
from typing import Dict, Optional
from app.data_models.models import Job
from app.data_models.schemas import JobStatus, AnalysisResult
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError
import logging

logger = logging.getLogger(__name__)

class JobQueueService:
    def create_job(self, db: Session, file_path: str, user_id: Optional[str] = None) -> Job:
        from app.core.database import safe_db_commit
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id, 
            file_path=file_path, 
            status=JobStatus.PENDING,
            user_id=user_id
        )
        try:
            db.add(job)
            safe_db_commit(db)
            db.refresh(job)
            logger.info(f"Created job {job_id}")
            return job
        except Exception as e:
            logger.error(f"[DB ERROR] Failed to create job {job_id}: {e}")
            db.rollback()
            raise

    def get_job(self, db: Session, job_id: str) -> Optional[Job]:
        return db.query(Job).filter(Job.job_id == job_id).first()

    def update_job(self, db: Session, job_id: str, status: JobStatus, result: Optional[AnalysisResult] = None, error: Optional[str] = None):
        """Update job with retry logic for database connection errors"""
        from app.core.database import safe_db_commit
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
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
                    safe_db_commit(db)
                    db.refresh(job)
                    logger.info(f"Updated job {job_id} to status {status}")
                    return  # Success, exit
                break
            except OperationalError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"[DB RETRY] Database error on attempt {attempt + 1}: {e}, retrying in {retry_delay}s...")
                    db.rollback()  # Rollback failed transaction
                    import time
                    time.sleep(retry_delay)
                else:
                    logger.error(f"[DB ERROR] Failed to update job {job_id} after {max_retries} attempts: {e}")
                    db.rollback()
            except Exception as e:
                logger.error(f"[DB ERROR] Unexpected error updating job {job_id}: {e}")
                db.rollback()
                break

job_queue = JobQueueService()

