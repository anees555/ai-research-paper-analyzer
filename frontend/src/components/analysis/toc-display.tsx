"use client";

import { useState, useMemo } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

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
}

export function TOCDisplay({ toc, metadata }: TOCDisplayProps) {
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
    <div className="flex flex-col min-h-screen">
      {/* Header with title only */}
      <div className="bg-white dark:bg-gray-900 border-b dark:border-gray-800 p-6 shadow-sm">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            {metadata?.title || "Analysis"}
          </h1>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1">
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-full max-w-7xl mx-auto w-full px-6 py-8">
      {/* Sidebar TOC */}
      <div className="lg:col-span-1 bg-white dark:bg-gray-900 rounded-xl border dark:border-gray-800 shadow-sm overflow-hidden flex flex-col">
        {/* Metadata section above TOC */}
        {(metadata?.authors?.length || metadata?.sections) && (
          <div className="bg-gray-50 dark:bg-gray-800 border-b dark:border-gray-700 p-4">
            {metadata?.authors && metadata.authors.length > 0 && (
              <div className="mb-4">
                <p className="text-xs font-semibold text-gray-700 dark:text-gray-300 mb-2 uppercase">Authors</p>
                <div className="space-y-1">
                  {metadata.authors.map((author, idx) => (
                    <p key={idx} className="text-xs text-gray-600 dark:text-gray-400 break-words leading-tight">
                      {author}
                    </p>
                  ))}
                </div>
              </div>
            )}
            {metadata?.sections && (
              <p className="text-xs text-gray-600 dark:text-gray-400">
                <span className="font-semibold">Sections:</span> {metadata.sections}
              </p>
            )}
          </div>
        )}
        
        <div className="sticky top-0 bg-gray-50 dark:bg-gray-800 border-b dark:border-gray-700 p-4 z-10">
          <h3 className="font-semibold text-gray-900 dark:text-white mb-1">
            Table of Contents
          </h3>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            {flatTOC.length} sections
          </p>
        </div>

        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {flatTOC.map((section, idx) => (
            section.level <= 2 && renderTOCItem(section, idx)
          ))}
        </div>
      </div>

      {/* Content Area */}
      <div className="lg:col-span-3 space-y-4">
        {selectedSectionData && (
          <>
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
          </>
        )}
      </div>
    </div>
      </div>
    </div>
  );
}
