"""
Schemas for interactive navigation features: TOC and contribution diagrams.
"""
from typing import List, Dict, Any
from pydantic import BaseModel

class Section(BaseModel):
    title: str
    id: str
    level: int

class Contribution(BaseModel):
    source: str
    target: str
    label: str = ""

class TOCResponse(BaseModel):
    toc: List[Dict[str, Any]]

class DiagramResponse(BaseModel):
    nodes: List[str]
    edges: List[Dict[str, str]]
