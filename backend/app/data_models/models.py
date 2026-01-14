from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text, JSON, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from app.core.database import Base

def generate_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    jobs = relationship("Job", back_populates="user")
    chat_history = relationship("ChatHistory", back_populates="user")

class Job(Base):
    __tablename__ = "jobs"

    job_id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=True) # Optional for now to support anonymous if needed, or enforce later
    file_path = Column(String, nullable=True)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    error = Column(Text, nullable=True)
    result = Column(JSON, nullable=True)

    user = relationship("User", back_populates="jobs")
    chat_messages = relationship("ChatHistory", back_populates="job")

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(String, primary_key=True, default=generate_uuid)
    job_id = Column(String, ForeignKey("jobs.job_id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    sender = Column(String, nullable=False) # "user" or "ai"
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    job = relationship("Job", back_populates="chat_messages")
    user = relationship("User", back_populates="chat_history")
