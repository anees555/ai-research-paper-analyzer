from datetime import datetime
from typing import Optional
from app.data_models.schemas import JobStatus, AnalysisResult
from pydantic import BaseModel, Field

class Job(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None
    file_path: Optional[str] = None
    result: Optional[AnalysisResult] = None
