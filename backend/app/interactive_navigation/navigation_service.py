"""
Service for generating interactive navigation data, including diagrams and clickable table of contents (TOC),
for research summary web applications. Supports integration with fast, enhanced, and interactive modes.
"""
from typing import Dict, Any, List


class InteractiveNavigationService:
    def __init__(self):
        pass

    def generate_mermaid_diagram(self, sections: List[Dict[str, Any]]) -> str:
        """
        Generate a Mermaid flowchart diagram string from a list of sections.
        Args:
            sections: List of section dicts with 'title', 'id', and 'level'.
        Returns:
            Mermaid diagram string (flowchart)
        """
        if not sections:
            return "graph TD\nA[No sections found]"
        diagram = ["graph TD"]
        # Create nodes
        for sec in sections:
            node_id = sec["id"]
            label = sec["title"].replace('"', '\"')
            diagram.append(f"    {node_id}[\"{label}\"]")
        # Link each section to the next (linear flow)
        for i in range(len(sections) - 1):
            diagram.append(f"    {sections[i]['id']} --> {sections[i+1]['id']}")
        return "\n".join(diagram)

    def generate_toc(self, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a hierarchical, clickable table of contents structure from section metadata.
        Args:
            sections: List of section dicts with 'title', 'level', and 'id'.
        Returns:
            Nested dict representing the TOC tree.
        """
        # Example: flat to tree conversion (simple version)
        toc = []
        stack = []
        for section in sections:
            node = {"title": section["title"], "id": section["id"], "children": []}
            while stack and section["level"] <= stack[-1]["level"]:
                stack.pop()
            if stack:
                stack[-1]["node"]["children"].append(node)
            else:
                toc.append(node)
            stack.append({"level": section["level"], "node": node})
        return {"toc": toc}

    def generate_contribution_diagram(self, contributions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a diagram data structure (e.g., for Mermaid or D3.js) representing paper contributions/flow.
        Args:
            contributions: List of dicts with 'source', 'target', and 'label'.
        Returns:
            Dict with nodes and edges for diagram rendering.
        """
        nodes = set()
        edges = []
        for c in contributions:
            nodes.add(c["source"])
            nodes.add(c["target"])
            edges.append({"from": c["source"], "to": c["target"], "label": c.get("label", "")})
        return {
            "nodes": list(nodes),
            "edges": edges
        }

# Singleton instance for import
interactive_navigation_service = InteractiveNavigationService()
