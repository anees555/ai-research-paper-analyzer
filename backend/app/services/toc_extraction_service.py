#!/usr/bin/env python3
"""
Table of Contents (TOC) Extraction Service
Extracts hierarchical paper structure with heading levels and section-level summaries
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class TOCSection:
    """Represents a section in the table of contents"""
    title: str
    level: int  # 1 = top-level, 2 = subsection, 3 = sub-subsection
    number: Optional[str] = None  # e.g., "1.1.2"
    content: str = ""  # Original content from paper
    summary: Optional[str] = None  # AI-generated summary
    page: Optional[int] = None
    children: List["TOCSection"] = None  # Nested sections
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "title": self.title,
            "level": self.level,
            "number": self.number,
            "content": self.content,
            "summary": self.summary,
            "page": self.page,
            "children": [child.to_dict() for child in self.children]
        }


class TOCExtractionService:
    """
    Extracts hierarchical table of contents from paper sections
    Generates section-level summaries and maintains hierarchy
    """
    
    def __init__(self):
        self.section_counter = {}  # Track section numbering by level
        
        # Section ordering heuristic - common academic paper sections
        self.known_sections = {
            "abstract": 0,
            "introduction": 1,
            "background": 2,
            "related work": 2,
            "methodology": 3,
            "method": 3,
            "approach": 3,
            "results": 4,
            "experiments": 4,
            "evaluation": 4,
            "discussion": 5,
            "conclusion": 6,
            "future work": 6,
            "acknowledgments": 7,
            "references": 8,
        }
    
    async def extract_toc_structure(
        self, 
        sections: Dict[str, str],
        abstract: Optional[str] = None
    ) -> List[TOCSection]:
        """
        Extract hierarchical TOC from flat sections dictionary
        
        Args:
            sections: Dict of section_title -> content
            abstract: Paper abstract
            
        Returns:
            List of top-level TOCSection objects with hierarchy
        """
        try:
            logger.info(f"[TOC] Extracting hierarchical structure from {len(sections)} sections")
            
            # Create TOC sections
            toc_items = []
            
            # Add abstract if present
            if abstract:
                abstract_section = TOCSection(
                    title="Abstract",
                    level=0,
                    number="0",
                    content=abstract,
                    page=1
                )
                toc_items.append(abstract_section)
            
            # Process each section
            section_list = list(sections.items())
            for idx, (title, content) in enumerate(section_list):
                level = self._determine_section_level(title, content)
                number = self._get_section_number(title, level)
                
                section = TOCSection(
                    title=title,
                    level=level,
                    number=number,
                    content=content,
                    page=idx + 2  # Estimate page numbers
                )
                toc_items.append(section)
                
                logger.debug(f"[TOC] Section: {number} {title} (level={level}, content_len={len(content)})")
            
            # Build hierarchical structure
            hierarchical_toc = self._build_hierarchy(toc_items)
            
            logger.info(f"[TOC] Extracted {len(hierarchical_toc)} top-level sections with hierarchy")
            return hierarchical_toc
            
        except Exception as e:
            logger.error(f"[ERROR] TOC extraction failed: {e}")
            raise
    
    def _determine_section_level(self, title: str, content: str) -> int:
        """
        Determine the heading level (1-3) for a section
        
        Heuristic:
        - Abstract, Intro → level 1
        - Subsections (Background, Related Work) → level 2
        - Sub-subsections → level 3
        """
        title_lower = title.lower().strip()
        content_len = len(content)
        
        # Check known section types
        for known_section, level in self.known_sections.items():
            if known_section in title_lower:
                return level if level > 0 else 1
        
        # Heuristic: Short titles might be subheadings
        if len(title) < 15:
            return 3
        
        # Heuristic: Medium-length are usually subsections
        if len(title) < 30:
            return 2
        
        # Default to top-level
        return 1
    
    def _get_section_number(self, title: str, level: int) -> str:
        """Generate academic-style section numbering"""
        if level == 0:
            return "Abstract"
        
        if level not in self.section_counter:
            self.section_counter[level] = 0
        
        self.section_counter[level] += 1
        
        # Reset lower-level counters
        for l in list(self.section_counter.keys()):
            if l > level:
                del self.section_counter[l]
        
        # Generate numbering
        if level == 1:
            return str(self.section_counter[level])
        else:
            # Build nested number like "1.2.3"
            numbers = []
            for l in range(1, level + 1):
                numbers.append(str(self.section_counter.get(l, 0)))
            return ".".join(numbers)
    
    def _build_hierarchy(self, flat_sections: List[TOCSection]) -> List[TOCSection]:
        """
        Build hierarchical structure from flat list
        Returns list of top-level sections with nested children
        """
        if not flat_sections:
            return []
        
        root = []
        stack = []  # Stack of parent sections
        
        for section in flat_sections:
            # Abstract stays at root level
            if section.level == 0:
                root.append(section)
                stack = []
                continue
            
            # Adjust stack to current level
            while stack and stack[-1].level >= section.level:
                stack.pop()
            
            if not stack:
                # Top-level section
                root.append(section)
            else:
                # Add as child of current parent
                stack[-1].children.append(section)
            
            # Add to stack for potential children
            stack.append(section)
        
        return root
    
    async def generate_section_summaries(
        self,
        toc: List[TOCSection],
        summarizer_fn,
        max_workers: int = 8
    ) -> List[TOCSection]:
        """
        Generate AI summaries for all sections in parallel
        
        Args:
            toc: Hierarchical TOC with sections
            summarizer_fn: Async function that takes (title, content) -> summary
            max_workers: Number of parallel workers (default 8 for faster processing)
            
        Returns:
            TOC with populated summary fields
        """
        try:
            logger.info(f"[SUMMARY] Generating section summaries with {max_workers} parallel workers")
            
            # Flatten TOC for parallel processing
            all_sections = self._flatten_toc(toc)
            logger.info(f"[SUMMARY] Processing {len(all_sections)} sections")
            
            # Create tasks with semaphore for rate limiting
            semaphore = asyncio.Semaphore(max_workers)
            
            async def summarize_with_limit(section: TOCSection) -> TOCSection:
                async with semaphore:
                    try:
                        logger.debug(f"[SUMMARY] Summarizing: {section.title[:50]}...")
                        
                        # Skip if content is too short
                        if len(section.content) < 100:
                            section.summary = section.content
                            logger.debug(f"[SUMMARY] Content too short, using original")
                            return section
                        
                        # Generate summary with 60-second timeout per section
                        try:
                            summary = await asyncio.wait_for(
                                summarizer_fn(section.title, section.content),
                                timeout=60  # 60 seconds per section
                            )
                            section.summary = summary
                            logger.debug(f"[SUMMARY] Completed: {section.title[:50]}... ({len(summary)} chars)")
                        except asyncio.TimeoutError:
                            logger.warning(f"[TIMEOUT] Summarization timed out for '{section.title}', using original")
                            section.summary = section.content  # Use full content as fallback
                        
                        return section
                        
                    except Exception as e:
                        logger.error(f"[ERROR] Failed to summarize '{section.title}': {e}")
                        section.summary = section.content  # Use full content as fallback
                        return section
            
            # Execute all summaries in parallel with 60-second timeout per section
            # Create actual Task objects so we can cancel them if needed
            tasks = [asyncio.create_task(summarize_with_limit(section)) for section in all_sections]
            try:
                # Add timeout: 10 minutes max for all summaries (with 8 workers for speed)
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=600  # 10 minutes max
                )
            except asyncio.TimeoutError:
                logger.error("[TIMEOUT] Section summarization exceeded 10 minutes, cancelling remaining tasks")
                # Cancel remaining tasks - now they're actual Task objects
                for task in tasks:
                    if not task.done():
                        task.cancel()
                # Wait briefly for cancellations to propagate
                await asyncio.sleep(0.1)
                raise
            
            logger.info("[SUMMARY] Completed all section summaries")
            return toc
            
        except Exception as e:
            logger.error(f"[ERROR] Summary generation failed: {e}")
            raise
    
    def _flatten_toc(self, toc: List[TOCSection]) -> List[TOCSection]:
        """Flatten hierarchical TOC into single list"""
        flat = []
        
        def traverse(sections: List[TOCSection]):
            for section in sections:
                flat.append(section)
                if section.children:
                    traverse(section.children)
        
        traverse(toc)
        return flat
    
    def toc_to_html(self, toc: List[TOCSection], include_summaries: bool = True) -> str:
        """
        Convert hierarchical TOC to interactive HTML
        """
        html_parts = ['<div class="toc-container">']
        
        def render_section(section: TOCSection, depth: int = 0, path: str = "") -> str:
            indent = "  " * depth
            section_id = f"section-{section.number or section.title.replace(' ', '-')}"
            
            parts = [
                f'{indent}<article id="{section_id}" class="toc-section level-{section.level}">',
                f'{indent}  <div class="section-header">',
                f'{indent}    <h{min(3, section.level + 1)} class="section-title">'
                f'{section.number or ""} {section.title}</h{min(3, section.level + 1)}>',
                f'{indent}  </div>'
            ]
            
            # Original content
            if section.content:
                # Display full content
                display_content = section.content[:3000] + ("..." if len(section.content) > 3000 else "")
                parts.append(
                    f'{indent}  <div class="section-original">'
                    f'{indent}    <h5>Original Content</h5>'
                    f'{indent}    <div class="original-text">{display_content}</div>'
                    f'{indent}  </div>'
                )
            
            # AI Summary
            if include_summaries and section.summary:
                # Display full summary (limit to ~2000 chars for display = ~400 words)
                display_text = section.summary[:2000] + ("..." if len(section.summary) > 2000 else "")
                parts.append(
                    f'{indent}  <div class="section-summary">'
                    f'{indent}    <h5>AI Summary</h5>'
                    f'{indent}    <div class="summary-text">{display_text}</div>'
                    f'{indent}  </div>'
                )
            
            # Nested sections
            if section.children:
                parts.append(f'{indent}  <div class="nested-sections">')
                for child in section.children:
                    parts.append(render_section(child, depth + 1, f"{path}/{section_id}"))
                parts.append(f'{indent}  </div>')
            
            parts.append(f'{indent}</article>')
            return "\n".join(parts)
        
        for section in toc:
            html_parts.append(render_section(section))
        
        html_parts.append('</div>')
        return "\n".join(html_parts)
    
    def toc_to_sidebar(self, toc: List[TOCSection]) -> Dict[str, Any]:
        """
        Convert TOC to sidebar navigation structure
        Returns clickable TOC for sidebar navigation
        """
        def section_to_nav(section: TOCSection) -> Dict[str, Any]:
            section_id = f"section-{section.number or section.title.replace(' ', '-')}"
            
            return {
                "id": section_id,
                "title": section.title,
                "number": section.number,
                "level": section.level,
                "page": section.page,
                "children": [section_to_nav(child) for child in section.children] if section.children else []
            }
        
        return {
            "toc": [section_to_nav(s) for s in toc],
            "total_sections": len(self._flatten_toc(toc))
        }


# Global service instance
toc_service = TOCExtractionService()
