from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SummaryType(str, Enum):
    QUICK = "quick"
    DETAILED = "detailed"
    ALL = "all"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class AnalysisRequest(BaseModel):
    summary_type: SummaryType = SummaryType.ALL
    include_comprehensive_analysis: bool = True


class SectionSummary(BaseModel):
    section_name: str
    content: str


class TOCSection(BaseModel):
    """Hierarchical table of contents section"""
    title: str
    level: int  # 0=abstract, 1=section, 2=subsection, 3=sub-subsection
    number: Optional[str] = None  # e.g., "1.2.3"
    content: str  # Original section content
    summary: Optional[str] = None  # AI-generated summary
    page: Optional[int] = None
    children: List["TOCSection"] = Field(default_factory=list)  # Nested sections


# Update forward references
TOCSection.model_rebuild()


class PaperMetadata(BaseModel):
    title: str = "Unknown"
    authors: List[str] = []
    paper_id: str
    num_sections: int = 0
    processing_method: str = "unknown"


class AnalysisResult(BaseModel):
    metadata: PaperMetadata
    quick_summary: Optional[str] = None
    detailed_summary: Optional[Dict[str, str]] = None
    comprehensive_analysis: Optional[Dict[str, Any]] = None
    original_abstract: Optional[str] = None
    table_of_contents: Optional[List[TOCSection]] = None  # Hierarchical TOC with summaries


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    error: Optional[str] = None
    result: Optional[AnalysisResult] = None


# Chat-related schemas
class ChatMessage(BaseModel):
    """Single chat message"""

    role: MessageRole
    content: str
    timestamp: Optional[datetime] = None


class ChatSource(BaseModel):
    """Source reference from paper"""

    section: str
    similarity: float


class ChatRequest(BaseModel):
    """Request to ask a question about a paper"""

    job_id: str
    question: str
    conversation_history: Optional[List[ChatMessage]] = None


class ChatResponse(BaseModel):
    """Response from chat service"""

    message: str
    sources: List[ChatSource] = []
    confidence: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)
