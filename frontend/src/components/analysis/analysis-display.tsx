"use client";

import { useState, useEffect } from "react";
import {
  Menu,
  X,
  Copy,
  ChevronRight,
  Moon,
  Sun,
  Download,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useTheme } from "@/contexts/theme-context";

interface TOCItem {
  id: string;
  title: string;
  level: number;
}

interface AnalysisDisplayProps {
  title: string;
  abstract?: string;
  htmlContent: string;
  glossary?: Record<string, string>;
  figures?: Array<{
    url: string;
    caption: string;
    page: number;
  }>;
  metadata?: {
    authors?: string[];
    num_sections?: number;
    processing_method?: string;
  };
}

export function AnalysisDisplay({
  title,
  abstract,
  htmlContent,
  glossary = {},
  figures = [],
  metadata,
}: AnalysisDisplayProps) {
  const { theme, toggleTheme } = useTheme();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [toc, setToc] = useState<TOCItem[]>([]);
  const [activeSection, setActiveSection] = useState<string>("");
  const [expandedGlossary, setExpandedGlossary] = useState<string | null>(null);

  // Generate TOC from HTML headers
  useEffect(() => {
    const parser = new DOMParser();
    const doc = parser.parseFromString(htmlContent, "text/html");
    const headers = Array.from(doc.querySelectorAll("h1, h2, h3"));

    const items: TOCItem[] = headers.map((header, idx) => ({
      id: `header-${idx}`,
      title: header.textContent || "Section",
      level: parseInt(header.tagName[1]),
    }));

    setToc(items);
  }, [htmlContent]);

  // Get processed HTML with semantic improvements
  const getProcessedHTML = () => {
    const parser = new DOMParser();
    const doc = parser.parseFromString(htmlContent, "text/html");

    // Add IDs to headers for TOC linking
    doc.querySelectorAll("h1, h2, h3").forEach((header, idx) => {
      header.id = `header-${idx}`;
    });

    // Enhanced styling for better formatting
    doc.querySelectorAll("p").forEach((p) => {
      p.classList.add("mb-4", "leading-relaxed", "text-gray-700", "dark:text-gray-300");
    });

    doc.querySelectorAll("h2").forEach((h) => {
      h.classList.add("mt-8", "mb-4", "text-2xl", "font-bold", "text-gray-900", "dark:text-white", "border-b", "pb-3", "border-gray-200", "dark:border-gray-700");
    });

    doc.querySelectorAll("h3").forEach((h) => {
      h.classList.add("mt-6", "mb-3", "text-xl", "font-semibold", "text-gray-800", "dark:text-gray-100");
    });

    doc.querySelectorAll("ul").forEach((ul) => {
      ul.classList.add("list-disc", "list-inside", "mb-4", "space-y-2", "text-gray-700", "dark:text-gray-300");
    });

    doc.querySelectorAll("li").forEach((li) => {
      li.classList.add("ml-2");
    });

    return doc.documentElement.innerHTML;
  };

  // --- PATCH: Show all section summaries if available ---
  // Accept detailed_summary as a prop (add to props if not present)
  // For now, try to get it from metadata (legacy) or window (hacky fallback)
  let detailedSummary: Record<string, string> | undefined = undefined;
  if (typeof window !== "undefined" && (window as any).detailed_summary) {
    detailedSummary = (window as any).detailed_summary;
  }
  // If you add detailed_summary to props, use that instead

  // Render all section summaries if present
  const renderSectionSummaries = () => {
    if (!detailedSummary || Object.keys(detailedSummary).length === 0) return null;
    return (
      <div className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Section Summaries</h2>
        {Object.entries(detailedSummary).map(([section, summary]) => (
          <div key={section} className="mb-6">
            <h3 className="text-xl font-semibold mb-2">{section}</h3>
            <p className="leading-relaxed text-gray-700 dark:text-gray-300">{summary}</p>
          </div>
        ))}
      </div>
    );
  };

  const handleCopyCode = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  return (
    <div className={cn("min-h-screen", theme === "dark" ? "bg-gray-900" : "bg-gray-50")}>
      {/* Header */}
      <header className={cn("sticky top-0 z-20 border-b", theme === "dark" ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200")}>
        <div className="px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className={cn(
                "p-2 rounded-lg transition-colors",
                theme === "dark"
                  ? "hover:bg-gray-700 text-gray-300"
                  : "hover:bg-gray-100 text-gray-600"
              )}
              title={sidebarOpen ? "Hide sidebar" : "Show sidebar"}
            >
              {sidebarOpen ? (
                <X className="w-5 h-5" />
              ) : (
                <Menu className="w-5 h-5" />
              )}
            </button>
            <h1 className={cn(
              "font-bold truncate",
              theme === "dark" ? "text-white" : "text-gray-900"
            )}>
              {title}
            </h1>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={toggleTheme}
              className={cn(
                "p-2 rounded-lg transition-colors",
                theme === "dark"
                  ? "bg-gray-700 text-yellow-400 hover:bg-gray-600"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              )}
              title={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
            >
              {theme === "light" ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
            </button>

            <button
              className={cn(
                "p-2 rounded-lg transition-colors",
                theme === "dark"
                  ? "text-gray-400 hover:bg-gray-700"
                  : "text-gray-600 hover:bg-gray-100"
              )}
              title="Download as PDF"
            >
              <Download className="w-5 h-5" />
            </button>
          </div>
        </div>
      </header>

      <div className="flex h-[calc(100vh-64px)]">
        {/* Sidebar TOC */}
        {sidebarOpen && (
          <aside className={cn(
            "w-64 border-r overflow-y-auto transition-all",
            theme === "dark" ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"
          )}>
            <div className="p-6 space-y-6">
              {/* Paper Info */}
              <div className={cn(
                "space-y-3 pb-6 border-b",
                theme === "dark" ? "border-gray-700" : "border-gray-200"
              )}>
                <h3 className={cn(
                  "text-xs font-semibold uppercase tracking-wider",
                  theme === "dark" ? "text-gray-400" : "text-gray-500"
                )}>
                  Paper Information
                </h3>
                {metadata?.authors && metadata.authors.length > 0 && (
                  <div className={cn(
                    "text-sm",
                    theme === "dark" ? "text-gray-300" : "text-gray-600"
                  )}>
                    <p className="font-medium mb-2">Authors</p>
                    <div className="space-y-1">
                      {metadata.authors.map((author, idx) => (
                        <p key={idx} className="break-words text-xs leading-tight">
                          {author}
                        </p>
                      ))}
                    </div>
                  </div>
                )}
                {metadata?.num_sections && (
                  <div className={cn(
                    "text-sm",
                    theme === "dark" ? "text-gray-300" : "text-gray-600"
                  )}>
                    <p className="font-medium">Sections: {metadata.num_sections}</p>
                  </div>
                )}
              </div>

              {/* Table of Contents */}
              <div className="space-y-3">
                <h3 className={cn(
                  "text-xs font-semibold uppercase tracking-wider",
                  theme === "dark" ? "text-gray-400" : "text-gray-500"
                )}>
                  Table of Contents
                </h3>
                <nav className="space-y-1">
                  {toc.map((item) => (
                    <a
                      key={item.id}
                      href={`#${item.id}`}
                      onClick={() => setActiveSection(item.id)}
                      className={cn(
                        "flex items-center gap-2 px-3 py-2 rounded-lg transition-colors text-sm truncate",
                        item.level === 1 && "font-bold",
                        item.level === 2 && "ml-4 font-semibold",
                        item.level === 3 && "ml-8",
                        activeSection === item.id
                          ? theme === "dark"
                            ? "bg-blue-900 text-blue-100"
                            : "bg-blue-100 text-blue-900"
                          : theme === "dark"
                          ? "text-gray-300 hover:bg-gray-700"
                          : "text-gray-600 hover:bg-gray-100"
                      )}
                    >
                      <ChevronRight className="w-4 h-4 flex-shrink-0" />
                      <span className="truncate">{item.title}</span>
                    </a>
                  ))}
                </nav>
              </div>

              {/* Glossary */}
              {Object.keys(glossary).length > 0 && (
                <div className={cn(
                  "space-y-2 pt-6 border-t",
                  theme === "dark" ? "border-gray-700" : "border-gray-200"
                )}>
                  <h3 className={cn(
                    "text-xs font-semibold uppercase tracking-wider",
                    theme === "dark" ? "text-gray-400" : "text-gray-500"
                  )}>
                    Key Terms
                  </h3>
                  <div className="space-y-1 max-h-64 overflow-y-auto">
                    {Object.entries(glossary).slice(0, 8).map(([term, def]) => (
                      <button
                        key={term}
                        onClick={() =>
                          setExpandedGlossary(
                            expandedGlossary === term ? null : term
                          )
                        }
                        className={cn(
                          "w-full text-left px-2 py-2 rounded transition-colors text-sm",
                          theme === "dark"
                            ? "hover:bg-gray-700 text-blue-300"
                            : "hover:bg-gray-100 text-blue-600"
                        )}
                      >
                        <div className="font-medium truncate">{term}</div>
                        {expandedGlossary === term && (
                          <div className={cn(
                            "mt-2 text-xs leading-relaxed",
                            theme === "dark" ? "text-gray-300" : "text-gray-600"
                          )}>
                            {def}
                          </div>
                        )}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </aside>
        )}

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto">
          <div className={cn(
            "max-w-4xl mx-auto p-8 prose prose-lg max-w-none",
            theme === "dark" ? "prose-invert bg-gray-900 text-gray-100" : "bg-gray-50"
          )}>
            {/* Abstract */}
            {abstract && (
              <div className={cn(
                "mb-8 p-6 rounded-xl border-l-4",
                theme === "dark"
                  ? "bg-gray-800 border-blue-500 text-gray-200"
                  : "bg-blue-50 border-blue-400"
              )}>
                <h2 className={cn(
                  "text-lg font-bold mb-4",
                  theme === "dark" ? "text-blue-300" : "text-blue-900"
                )}>
                  Abstract
                </h2>
                <p className={cn(
                  "italic leading-relaxed",
                  theme === "dark" ? "text-gray-300" : "text-gray-700"
                )}>
                  {abstract}
                </p>
              </div>
            )}

            {/* Main HTML Content */}
            <div
              className={cn(
                "space-y-6",
                theme === "dark" ? "text-gray-300" : "text-gray-700"
              )}
              dangerouslySetInnerHTML={{ __html: getProcessedHTML() }}
            />

            {/* Figures Section */}
            {figures.length > 0 && (
              <div className={cn(
                "mt-12 pt-8 border-t",
                theme === "dark" ? "border-gray-700" : "border-gray-200"
              )}>
                <h2 className={cn(
                  "text-2xl font-bold mb-6",
                  theme === "dark" ? "text-white" : "text-gray-900"
                )}>
                  Figures
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {figures.map((figure, idx) => (
                    <figure
                      key={idx}
                      className={cn(
                        "group relative rounded-lg overflow-visible border",
                        theme === "dark" ? "border-gray-700 bg-gray-800" : "border-gray-200 bg-white shadow-sm"
                      )}
                    >
                      <div className={cn(
                        "relative bg-center bg-cover rounded-t-lg overflow-hidden",
                        theme === "dark" ? "bg-gray-700" : "bg-gray-100"
                      )}>
                        <img
                          src={figure.url}
                          alt={figure.caption}
                          className="w-full h-64 object-cover"
                        />
                      </div>

                      {/* Hover preview: show full figure in larger panel */}
                      <div
                        className={cn(
                          "pointer-events-none absolute left-1/2 top-3 z-30 hidden w-[92vw] max-w-4xl -translate-x-1/2 rounded-xl border p-3 shadow-2xl group-hover:block",
                          theme === "dark" ? "border-gray-600 bg-gray-900/95" : "border-gray-300 bg-white/95"
                        )}
                      >
                        <div
                          className={cn(
                            "mb-2 text-xs font-semibold",
                            theme === "dark" ? "text-gray-200" : "text-gray-700"
                          )}
                        >
                          Full Figure Preview
                        </div>
                        <img
                          src={figure.url}
                          alt={figure.caption}
                          className="w-full max-h-[70vh] object-contain rounded-md"
                        />
                      </div>

                      <figcaption className="p-4">
                        <p className={cn(
                          "text-sm font-medium mb-1",
                          theme === "dark" ? "text-gray-200" : "text-gray-900"
                        )}>
                          Figure {figure.page}
                        </p>
                        <p className={cn(
                          "text-sm leading-relaxed",
                          theme === "dark" ? "text-gray-400" : "text-gray-600"
                        )}>
                          {figure.caption}
                        </p>
                      </figcaption>
                    </figure>
                  ))}
                </div>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
