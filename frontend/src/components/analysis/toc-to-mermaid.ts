// Utility to convert a hierarchical TOC to a Mermaid tree diagram string
// TOC format: Array<{ id: string, title: string, children?: TOCSection[] }>

export interface TOCSection {
  title: string;
  level: number;
  number?: string;
  content: string;
  summary?: string;
  page?: number;
  id: string;
  children?: TOCSection[];
}

export function tocToMermaid(toc: TOCSection[]): string {
  let nodes: string[] = [];
  let edges: string[] = [];

  function walk(section: TOCSection, parentId?: string) {
    // Sanitize node id for Mermaid
    const nodeId = section.id.replace(/[^a-zA-Z0-9_]/g, "_");
    nodes.push(`${nodeId}[\"${section.title.replace(/\"/g, '\\"')}\"]`);
    if (parentId) {
      edges.push(`${parentId} --> ${nodeId}`);
    }
    if (section.children && section.children.length > 0) {
      for (const child of section.children) {
        walk(child, nodeId);
      }
    }
  }

  for (const section of toc) {
    walk(section);
  }

  return [
    "graph LR",
    ...nodes,
    ...edges
  ].join("\n");
}
