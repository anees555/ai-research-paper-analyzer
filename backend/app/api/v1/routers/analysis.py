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


from app.interactive_navigation.navigation_service import interactive_navigation_service

class ProcessingMode(str, Enum):
    FAST = "fast"          # 30-60 seconds, abstract + 3-paragraph summary  
    ENHANCED = "enhanced"  # 120-300 seconds, professional structured analysis with figures
    INTERACTIVE = "interactive"  # Interactive navigation mode: diagrams, clickable TOC


async def process_analysis_task(job_id: str, file_path: str, mode: str = "fast"):
    """
    Background task to run analysis
    """
    db = SessionLocal()
    try:
        # Update status to processing
        job_queue.update_job(db, job_id, JobStatus.PROCESSING)


        # Use the new robust section-wise pipeline for all modes
        if mode == ProcessingMode.FAST:
            result = await fast_analysis_service.analyze_fast(file_path, job_id)
        elif mode == ProcessingMode.ENHANCED:
            result = await enhanced_analysis_service.analyze_paper_enhanced(file_path, job_id)
        elif mode == ProcessingMode.INTERACTIVE:
            # Extract real sections/subsections from the PDF using the robust parser
            from scripts.parse_pdf_optimized import parse_pdf_with_grobid_optimized as parse_pdf_with_grobid
            # Parse PDF and extract structure
            paper_data = analysis_service._paper_data_cache.get(job_id)
            if not paper_data:
                paper_data = parse_pdf_with_grobid(file_path, "output")
                analysis_service._store_paper_data(job_id, paper_data)


            # Build section list for TOC (with content for frontend display)
            sections = []
            section_id_map = {}
            if paper_data and "sections" in paper_data:
                for idx, (title, content) in enumerate(paper_data["sections"].items()):
                    # Heuristic: Level 1 for main sections, Level 2 for subsections (if ':' or '-' in title)
                    level = 2 if any(sep in title for sep in [":", "-"]) else 1
                    sec_id = f"sec{idx+1}"
                    section_id_map[title] = sec_id
                    sections.append({
                        "title": title,
                        "id": sec_id,
                        "level": level,
                        "content": content
                    })

            # Generate TOC (with content for each node)
            def toc_with_content(sections):
                toc = []
                stack = []
                for section in sections:
                    node = {"title": section["title"], "id": section["id"], "level": section["level"], "content": section["content"], "children": []}
                    while stack and section["level"] <= stack[-1]["level"]:
                        stack.pop()
                    if stack:
                        stack[-1]["node"]["children"].append(node)
                    else:
                        toc.append(node)
                    stack.append({"level": section["level"], "node": node})
                return toc

            toc = toc_with_content(sections)



            # Generate a Mermaid diagram string from the section flow
            try:
                mermaid_diagram = interactive_navigation_service.generate_mermaid_diagram(sections)
                if not mermaid_diagram or not mermaid_diagram.strip().startswith("graph"):
                    logger.warning(f"[Interactive] Mermaid diagram not generated or invalid for job {job_id}. Sections: {sections}")
                    mermaid_diagram = "graph TD\nA[No sections found]"
            except Exception as e:
                logger.error(f"[Interactive] Failed to generate Mermaid diagram for job {job_id}: {e}")
                mermaid_diagram = "graph TD\nA[Diagram generation error]"

            # Build PaperMetadata
            from app.data_models.schemas import AnalysisResult, PaperMetadata
            metadata = PaperMetadata(
                title=paper_data.get("title", "Unknown"),
                authors=paper_data.get("authors", []),
                paper_id=job_id,
                num_sections=len(sections),
                processing_method="interactive"
            )
            # Place toc and diagram in comprehensive_analysis for frontend access
            result = AnalysisResult(
                metadata=metadata,
                quick_summary=None,
                detailed_summary=None,
                comprehensive_analysis={
                    "toc": toc,
                    "diagram": mermaid_diagram
                },
                original_abstract=paper_data.get("abstract", None),
                table_of_contents=None
            )
        else:
            result = await analysis_service.analyze_paper(file_path, job_id)

        # Update status to completed
        job_queue.update_job(db, job_id, JobStatus.COMPLETED, result=result)

        # Try to index for chat (non-blocking)
        try:
            from app.services.chat_service import chat_service

            # Build paper data from result instead of cache
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
    mode: ProcessingMode = Query(ProcessingMode.FAST, description="Processing mode: fast (~60s, 3-paragraph summary), enhanced (~300s, professional analysis), or interactive (diagrams, clickable TOC)"),
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
    if mode not in [ProcessingMode.FAST, ProcessingMode.ENHANCED]:
        raise HTTPException(
            status_code=400,
            detail="Only 'fast' and 'enhanced' modes are supported for instant analysis. Use /upload endpoint for other modes."
        )
    
    try:
        # 1. Save file temporarily
        file_path = await file_service.save_upload_file(file)

        # 2. Generate unique job ID for tracking
        import uuid
        job_id = str(uuid.uuid4())

        # 3. Perform instant analysis
        result = await fast_analysis_service.analyze_fast(file_path, job_id)

        # 4. Clean up temporary file
        import os
        try:
            os.remove(file_path)
        except:
            pass  # Ignore cleanup errors

        # Only return a brief, simple summary for fast mode
        summary_text = result.quick_summary if hasattr(result, 'quick_summary') else None
        metadata = result.metadata if hasattr(result, 'metadata') else None

        return {
            "status": "completed",
            "processing_mode": mode.value,
            "analysis_result": {
                "metadata": metadata,
                "quick_summary": summary_text
            },
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
    
    Three modes available:
    - fast: Quick 3-paragraph summary (abstract + problem + solution + conclusion)
    - enhanced: Professional structured analysis with figures and glossary
    - interactive: Interactive navigation with diagrams and clickable TOC
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
            },
            "interactive": {
                "name": "Interactive Navigation Mode",
                "estimated_time": "10-30 seconds",
                "description": "Interactive diagrams and clickable table of contents for research paper navigation.",
                "features": [
                    "Clickable table of contents",
                    "Contribution flow diagrams",
                    "Section navigation",
                    "Integration-ready for frontend visualizations"
                ],
                "ai_models": False,
                "recommended_for": "Interactive exploration, presentations, teaching"
            }
        },
        "default_mode": "fast",
        "supported_modes": ["fast", "enhanced", "interactive"]
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

