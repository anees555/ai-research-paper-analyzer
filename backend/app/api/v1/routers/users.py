from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.api import deps
from app.data_models.models import User, Job
from app.core.database import get_db
from pydantic import BaseModel

router = APIRouter()

class UserSchema(BaseModel):
    id: str
    email: str
    
    class Config:
        from_attributes = True

class JobSummary(BaseModel):
    job_id: str
    status: str
    created_at: Any
    file_path: Any
    
    class Config:
        from_attributes = True

@router.get("/me", response_model=UserSchema)
def read_user_me(
    current_user: User = Depends(deps.get_current_user),
):
    """
    Get current user.
    """
    return current_user

@router.get("/me/history", response_model=List[JobSummary])
def read_user_history(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(deps.get_current_user),
):
    """
    Retrieve user job history.
    """
    jobs = db.query(Job).filter(Job.user_id == current_user.id).offset(skip).limit(limit).all()
    return jobs
