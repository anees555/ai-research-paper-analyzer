import React, { useState } from "react";
import { uploadPaper, uploadPaperInstant } from "@/lib/api";
import { AnalysisDisplay } from "@/components/analysis/analysis-display";
import { TOCDisplay } from "@/components/analysis/toc-display";
import { PaperChat } from "@/components/chat/paper-chat";

const router = {
  push: (url: string) => {
    window.location.href = url;
  },
};

const FileUpload: React.FC = () => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedMode, setSelectedMode] = useState("fast");
  const [instantResult, setInstantResult] = useState<any | null>(null);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = async (file: File) => {
    // Validate file type
    if (file.type !== "application/pdf") {
      setError("Only PDF files are allowed. Please select a PDF document.");
      return;
    }

    // Validate file size
    if (file.size > 10 * 1024 * 1024) {
      setError("File size must be under 10MB. Please choose a smaller file.");
      return;
    }

    setError(null);
    setIsUploading(true);

    try {
      let response;
      if (selectedMode === "fast") {
        // Use instant endpoint for fast mode
        response = await uploadPaperInstant(file, "fast");
        if (!response || !response.status || !response.analysis_result) {
          throw new Error("Invalid server response - no result");
        }
        // Save result to localStorage and redirect to analysis page
        const instantJobId = `instant-${Date.now()}`;
        localStorage.setItem(
          `instant_result_${instantJobId}`,
          JSON.stringify(response.analysis_result)
        );
        // Save to local history
        const historyItem = {
          job_id: instantJobId,
          title: file.name.replace(".pdf", ""),
          date: new Date().toISOString(),
          status: response.status,
        };
        const existing = JSON.parse(localStorage.getItem("paper_history") || "[]");
        localStorage.setItem(
          "paper_history",
          JSON.stringify([historyItem, ...existing].slice(0, 10))
        );
        // Redirect to analysis page
        router.push(`/analysis/${instantJobId}`);
      } else {
        // Use normal upload for enhanced/professional
        response = await uploadPaper(file, selectedMode);
        if (!response || !response.job_id) {
          throw new Error("Invalid server response - no job ID received");
        }
        // Save to local history
        const historyItem = {
          job_id: response.job_id,
          title: file.name.replace(".pdf", ""),
          date: new Date().toISOString(),
          status: response.status || "pending",
        };
        const existing = JSON.parse(localStorage.getItem("paper_history") || "[]");
        localStorage.setItem(
          "paper_history",
          JSON.stringify([historyItem, ...existing].slice(0, 10))
        );
        // Redirect
        router.push(`/analysis/${response.job_id}`);
      }
    } catch (err: any) {
      console.error("Upload error:", err);
      if (err.code === "NETWORK_ERROR" || err.message?.includes("Network")) {
        setError(
          "Connection failed. Please check if the backend server is running."
        );
      } else if (err.response?.status === 413) {
        setError("File too large. Please choose a smaller PDF file.");
      } else if (err.response?.status === 415) {
        setError("Unsupported file type. Please upload a PDF file.");
      } else if (err.response?.status >= 500) {
        setError("Server error. Please try again in a moment.");
      } else {
        setError(
          `Upload failed: ${err.message || "Unknown error"}. Please try again.`
        );
      }
    } finally {
      setIsUploading(false);
    }
  };

  return (
    instantResult ? (
      <div className="flex flex-col min-h-screen bg-gray-50">
        {/* Main content area, same as enhanced mode */}
        <div className="flex flex-col min-h-screen">
          {instantResult.table_of_contents && instantResult.table_of_contents.length > 0 ? (
            <>
              <TOCDisplay
                toc={instantResult.table_of_contents}
                metadata={{
                  title: instantResult.metadata?.title || "Summary",
                  authors: instantResult.metadata?.authors || [],
                  sections: instantResult.metadata?.num_sections,
                }}
                glossary={instantResult.comprehensive_analysis?.glossary || {}}
              />
            </>
          ) : (
            <>
              <div className="w-full max-w-7xl mx-auto px-2 md:px-8 py-8">
                <h2 className="text-3xl font-bold mb-8 text-center">Quick Summary Result</h2>
                <AnalysisDisplay
                  title={instantResult.metadata?.title || "Summary"}
                  abstract={instantResult.original_abstract}
                  htmlContent={
                    instantResult.comprehensive_analysis?.html_summary ||
                    `<h2>Summary</h2><p>${instantResult.quick_summary || instantResult.original_abstract}</p>`
                  }
                  glossary={instantResult.comprehensive_analysis?.glossary || {}}
                  figures={instantResult.comprehensive_analysis?.figures?.map((fig: any) => ({
                    url: fig.url,
                    caption: fig.caption || fig.description || "Figure",
                    page: fig.page || 1,
                  })) || []}
                  metadata={{
                    authors: instantResult.metadata?.authors || [],
                    num_sections: instantResult.metadata?.num_sections,
                    processing_method: instantResult.metadata?.processing_method,
                  }}
                />
                {/* Section summaries, if present */}
                {instantResult.detailed_summary && Object.keys(instantResult.detailed_summary).length > 0 && (
                  <div className="max-w-4xl mx-auto p-8">
                    <h2 className="text-2xl font-bold mb-4">Section Summaries</h2>
                    {Object.entries(instantResult.detailed_summary).map(([section, summary]) => (
                      <div key={section} className="mb-6">
                        <h3 className="text-xl font-semibold mb-2">{section}</h3>
                        <p className="leading-relaxed text-gray-700 dark:text-gray-300">{summary}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>
          )}
          {/* Disabled PaperChat for fast mode (no jobId) */}
          <div className="max-w-4xl mx-auto pb-8">
            <PaperChat jobId={null} paperTitle={instantResult.metadata?.title || "Summary"} disabled reason="Chat is only available for enhanced mode analyses." />
          </div>
        </div>
      </div>
    ) : (
      <div className="max-w-md mx-auto p-6 bg-white rounded shadow">
        <h3 className="text-xl font-semibold mb-4">Upload a Research Paper</h3>
        <div className="mb-4">
          <label className="block mb-2 font-medium">Choose Mode:</label>
          <div className="flex gap-4">
            <label>
              <input
                type="radio"
                name="mode"
                value="fast"
                checked={selectedMode === "fast"}
                onChange={() => setSelectedMode("fast")}
              />
              <span className="ml-2">Fast (Quick summary)</span>
            </label>
            <label>
              <input
                type="radio"
                name="mode"
                value="enhanced"
                checked={selectedMode === "enhanced"}
                onChange={() => setSelectedMode("enhanced")}
              />
              <span className="ml-2">Enhanced/Professional (Detailed)</span>
            </label>
            <label>
              <input
                type="radio"
                name="mode"
                value="interactive"
                checked={selectedMode === "interactive"}
                onChange={() => setSelectedMode("interactive")}
              />
              <span className="ml-2">Interactive (TOC & Diagram)</span>
            </label>
          </div>
        </div>
        <div
          className={`border-2 border-dashed rounded p-6 text-center mb-4 ${isDragOver ? "border-blue-500 bg-blue-50" : "border-gray-300"}`}
          onDrop={handleDrop}
          onDragOver={e => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={e => { e.preventDefault(); setIsDragOver(false); }}
        >
          <p className="mb-2">Drag and drop a PDF file here, or</p>
          <input
            type="file"
            accept="application/pdf"
            onChange={handleFileSelect}
            className="block mx-auto"
            disabled={isUploading}
          />
        </div>
        {error && <div className="text-red-600 mb-2">{error}</div>}
        {isUploading && <div className="text-blue-600">Uploading...</div>}
      </div>
    )
  );
};

export default FileUpload;