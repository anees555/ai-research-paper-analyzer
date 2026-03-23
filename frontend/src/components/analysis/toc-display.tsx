"use client";

import { useState, useMemo } from "react";
import { ChevronDown, ChevronRight, Menu, X, Moon, Sun, Download } from "lucide-react";
import { cn } from "@/lib/utils";
import { useTheme } from "@/contexts/theme-context";

interface TOCSection {
  title: string;
  level: number;
  number?: string;
  content: string;
  summary?: string;
  page?: number;
  children?: TOCSection[];
}

interface TOCDisplayProps {
  toc: TOCSection[];
  metadata?: {
    title?: string;
    authors?: string[];
    sections?: number;
  };
  glossary?: Record<string, string>;
}

export function TOCDisplay({ toc, metadata, glossary = {} }: TOCDisplayProps) {
  const { theme, toggleTheme } = useTheme();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [expandedGlossary, setExpandedGlossary] = useState<string | null>(null);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());
  const [selectedSection, setSelectedSection] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"original" | "summary" | "both">("both");

  // Flatten TOC for navigation
  const flatTOC = useMemo(() => {
    const flat: (TOCSection & { id: string; path: string })[] = [];
    let counter = 0;

    function traverse(sections: TOCSection[], parentPath = "") {
      sections.forEach((section, idx) => {
        const id = `section-${counter++}`;
        const path = parentPath ? `${parentPath}/${idx}` : String(idx);
        flat.push({ ...section, id, path });

        if (section.children?.length) {
          traverse(section.children, path);
        }
      });
    }

    traverse(toc);
    return flat;
  }, [toc]);

  const toggleSection = (id: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(id)) {
      newExpanded.delete(id);
    } else {
      newExpanded.add(id);
    }
    setExpandedSections(newExpanded);
  };

  const selectedSectionData = selectedSection
    ? flatTOC.find((s) => s.id === selectedSection)
    : flatTOC[0];

  const renderTOCItem = (
    section: TOCSection & { id: string; path: string },
    index: number
  ) => {
    const hasChildren = section.children && section.children.length > 0;
    const isExpanded = expandedSections.has(section.id);
    const isSelected = selectedSection === section.id;

    return (
      <div key={section.id} className="space-y-1">
        <div
          className={`flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition-colors ${
            isSelected
              ? "bg-blue-100 text-blue-900 dark:bg-blue-900/30 dark:text-blue-300"
              : "hover:bg-gray-100 dark:hover:bg-gray-800"
          }`}
          onClick={() => setSelectedSection(section.id)}
        >
          {hasChildren && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                toggleSection(section.id);
              }}
              className="hover:bg-gray-200 dark:hover:bg-gray-700 rounded p-0.5"
            >
              {isExpanded ? (
                <ChevronDown className="w-4 h-4" />
              ) : (
                <ChevronRight className="w-4 h-4" />
              )}
            </button>
          )}
          {!hasChildren && <div className="w-4" />}

          <div className="flex-1 min-w-0">
            <div
              className={`text-sm font-medium ${
                section.level === 0
                  ? "text-base font-bold"
                  : section.level === 1
                  ? "font-semibold"
                  : "font-normal"
              }`}
            >
              {section.number && (
                <span className="text-gray-500 mr-2">
                  {section.number}
                </span>
              )}
              {section.title}
            </div>
            {section.page && (
              <div className="text-xs text-gray-500 dark:text-gray-400">
                p. {section.page}
              </div>
            )}
          </div>
        </div>

        {hasChildren && isExpanded && (
          <div className="ml-4 border-l border-gray-300 dark:border-gray-700 space-y-1">
            {section.children!.map((child, idx) => {
              const childId = `${section.id}-${idx}`;
              const childFlat: TOCSection & { id: string; path: string } = {
                ...child,
                id: childId,
                path: `${section.path}/${idx}`,
              };
              return renderTOCItem(childFlat, idx);
            })}
          </div>
        )}
      </div>
    );
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
              {metadata?.title || "Analysis"}
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
        {/* Sidebar */}
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
                {metadata?.sections && (
                  <div className={cn(
                    "text-sm",
                    theme === "dark" ? "text-gray-300" : "text-gray-600"
                  )}>
                    <p className="font-medium">Sections: {metadata.sections}</p>
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
                  {flatTOC.map((item) => (
                    <button
                      key={item.id}
                      onClick={() => setSelectedSection(item.id)}
                      className={cn(
                        "w-full text-left flex items-center gap-2 px-3 py-2 rounded-lg transition-colors text-sm truncate",
                        item.level === 0 && "font-bold",
                        item.level === 1 && "ml-0 font-semibold",
                        item.level === 2 && "ml-4",
                        item.level >= 3 && "ml-8",
                        selectedSection === item.id
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
                    </button>
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
        <main className={cn(
          "flex-1 overflow-y-auto",
          theme === "dark" ? "bg-gray-900" : "bg-gray-50"
        )}>
          <div className={cn(
            "max-w-4xl mx-auto p-8",
            theme === "dark" ? "text-gray-100" : "text-gray-900"
          )}>
            {selectedSectionData && (
              <div className="space-y-6">
            {/* View Mode Toggle */}
            <div className="flex gap-2 sticky top-0 z-10 bg-white dark:bg-gray-900 p-3 rounded-lg border dark:border-gray-800">
              <button
                onClick={() => setViewMode("original")}
                className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                  viewMode === "original"
                    ? "bg-blue-600 text-white"
                    : "bg-gray-200 text-gray-900 dark:bg-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600"
                }`}
              >
                Original
              </button>
              <button
                onClick={() => setViewMode("summary")}
                className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                  viewMode === "summary"
                    ? "bg-blue-600 text-white"
                    : "bg-gray-200 text-gray-900 dark:bg-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600"
                }`}
              >
                AI Summary
              </button>
              <button
                onClick={() => setViewMode("both")}
                className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                  viewMode === "both"
                    ? "bg-blue-600 text-white"
                    : "bg-gray-200 text-gray-900 dark:bg-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600"
                }`}
              >
                Both
              </button>
            </div>

            {/* Section Header */}
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl border border-blue-100 dark:border-blue-800 p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-1">
                    {selectedSectionData.number && (
                      <span className="text-blue-600 dark:text-blue-400 mr-2">
                        {selectedSectionData.number}
                      </span>
                    )}
                    {selectedSectionData.title}
                  </h2>
                  {selectedSectionData.page && (
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Page {selectedSectionData.page}
                    </p>
                  )}
                </div>
                <div className="text-xs bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 px-3 py-1 rounded-full font-medium">
                  Level {selectedSectionData.level}
                </div>
              </div>
            </div>

            {/* Content Panels */}
            {(viewMode === "original" || viewMode === "both") && (
              <div className="bg-white dark:bg-gray-900 rounded-xl border dark:border-gray-800 shadow-sm p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Original Content
                </h3>
                <div className="prose prose-sm dark:prose-invert max-w-none text-gray-700 dark:text-gray-300 leading-relaxed">
                  <p className="whitespace-pre-wrap text-justify">
                    {selectedSectionData.content}
                  </p>
                </div>
                {selectedSectionData.content.length > 300 && (
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-4 italic">
                    Content length: {selectedSectionData.content.length} characters
                  </p>
                )}
              </div>
            )}

            {(viewMode === "summary" || viewMode === "both") && (
              <div className="bg-green-50 dark:bg-green-900/20 rounded-xl border border-green-100 dark:border-green-800 shadow-sm p-6">
                <h3 className="text-lg font-semibold text-green-900 dark:text-green-300 mb-4">
                  AI-Generated Summary
                </h3>
                {selectedSectionData.summary ? (
                  <div className="prose prose-sm dark:prose-invert max-w-none text-gray-700 dark:text-gray-300 leading-relaxed">
                    <p className="whitespace-pre-wrap text-justify">
                      {selectedSectionData.summary}
                    </p>
                  </div>
                ) : (
                  <p className="text-gray-500 dark:text-gray-400 italic">
                    Summary not available for this section
                  </p>
                )}
                {selectedSectionData.summary && (
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-4 italic">
                    Summary length: {selectedSectionData.summary.length} characters (
                    {Math.round(
                      (selectedSectionData.summary.length /
                        selectedSectionData.content.length) *
                        100
                    )}
                    % compression)
                  </p>
                )}
              </div>
            )}

            {/* Nested Sections */}
            {selectedSectionData.children && selectedSectionData.children.length > 0 && (
              <div className="bg-white dark:bg-gray-900 rounded-xl border dark:border-gray-800 shadow-sm p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Subsections ({selectedSectionData.children.length})
                </h3>
                <div className="grid gap-2">
                  {selectedSectionData.children.map((child, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        const childFlat = flatTOC.find(
                          (s) =>
                            s.number === child.number || s.title === child.title
                        );
                        if (childFlat) setSelectedSection(childFlat.id);
                      }}
                      className="text-left px-4 py-2 rounded-lg hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-colors border border-gray-200 dark:border-gray-700"
                    >
                      <div className="font-medium text-gray-900 dark:text-white break-words">
                        {child.number} {child.title}
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        {child.content.substring(0, 200)}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
