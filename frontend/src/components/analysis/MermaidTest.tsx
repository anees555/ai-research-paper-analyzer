import React, { useEffect, useState } from "react";
import "mermaid/dist/mermaid.min.css";

const TEST_DIAGRAM = `graph TD\n    A[Start] --> B[Render Mermaid]\n    B --> C[Success]\n    B --> D[Fail]\n`;

const MermaidTest: React.FC = () => {
  const [svg, setSvg] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  useEffect(() => {
    if (!isClient) return;
    setError("");
    setSvg("");
    import("mermaid").then((mermaid) => {
      try {
        mermaid.default.initialize({ startOnLoad: false, theme: "default" });
        mermaid.default.render("mermaid-test-svg", TEST_DIAGRAM).then(({ svg }) => {
          setSvg(svg);
        }).catch((err: any) => {
          setError("Mermaid render error: " + String(err));
        });
      } catch (e) {
        setError("Mermaid render exception: " + String(e));
      }
    });
  }, [isClient]);

  if (!isClient) return null;

  return (
    <div style={{ padding: 24 }}>
      <h2>Mermaid Minimal Test</h2>
      <pre style={{ background: "#eee", padding: 8 }}>{TEST_DIAGRAM}</pre>
      {error && <div style={{ color: "red" }}>{error}</div>}
      {svg ? (
        <div style={{ border: "1px solid #888", padding: 8, marginTop: 12 }} dangerouslySetInnerHTML={{ __html: svg }} />
      ) : (
        <div style={{ color: "orange" }}>No SVG generated yet.</div>
      )}
    </div>
  );
};

export default MermaidTest;
