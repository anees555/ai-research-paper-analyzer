"""
Chat Router - API endpoints for paper Q&A chat
"""

from fastapi import APIRouter, HTTPException
from app.data_models.schemas import ChatRequest, ChatResponse
from app.services.chat_service import chat_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/ask", response_model=ChatResponse, summary="Ask a question about a paper"
)
async def ask_question(request: ChatRequest):
    """
    Ask a question about a specific analyzed paper.

    Uses semantic search to find relevant sections and generates
    an intelligent response based on the paper's content.
    """
    try:
        response = await chat_service.ask_question(
            job_id=request.job_id,
            question=request.question,
            conversation_history=request.conversation_history,
        )
        return response

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process question: {str(e)}"
        )


@router.post("/index/{job_id}", summary="Index a paper for chat")
async def index_paper(job_id: str):
    """
    Manually trigger indexing of a paper for chat.
    Usually called automatically after analysis completes.
    """
    from app.services.queue_service import job_queue

    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.result:
        raise HTTPException(
            status_code=400, detail="Job analysis not complete. Cannot index for chat."
        )

    # Build paper data from result
    paper_data = {
        "title": job.result.metadata.title,
        "abstract": job.result.original_abstract or "",
        "sections": job.result.detailed_summary or {},
    }

    success = chat_service.index_paper_for_chat(job_id, paper_data)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to index paper")

    return {"status": "indexed", "job_id": job_id}


@router.get("/status/{job_id}", summary="Check if paper is indexed for chat")
async def chat_status(job_id: str):
    """
    Check if a paper has been indexed and is ready for chat.
    """
    # Check if paper context exists
    is_indexed = job_id in chat_service._job_contexts

    return {"job_id": job_id, "indexed": is_indexed, "ready": is_indexed}
