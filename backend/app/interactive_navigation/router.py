"""
API router for interactive navigation features: clickable TOC and contribution diagrams.
"""
from fastapi import APIRouter, Body
from .navigation_service import interactive_navigation_service
from .schemas import Section, Contribution, TOCResponse, DiagramResponse
from typing import List

router = APIRouter(prefix="/interactive-navigation", tags=["Interactive Navigation"])

@router.post("/toc", response_model=TOCResponse)
def generate_toc(sections: List[Section] = Body(...)):
    """Generate a clickable table of contents from section metadata."""
    toc = interactive_navigation_service.generate_toc([s.dict() for s in sections])
    return toc

@router.post("/diagram", response_model=DiagramResponse)
def generate_diagram(contributions: List[Contribution] = Body(...)):
    """Generate a contribution flow diagram from contribution data."""
    diagram = interactive_navigation_service.generate_contribution_diagram([c.dict() for c in contributions])
    return diagram
