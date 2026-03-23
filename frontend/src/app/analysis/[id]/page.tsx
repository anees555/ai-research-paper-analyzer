"use client";
import MermaidTest from "@/components/analysis/MermaidTest";

import { useAnalysis } from "@/hooks/use-analysis";
import { useEffect, useState } from "react";
import { Loader2, AlertCircle, CheckCircle, FileText } from "lucide-react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { cn } from "@/lib/utils";
import { PaperChat } from "@/components/chat/paper-chat";
import { AnalysisDisplay } from "@/components/analysis/analysis-display";
import { TOCDisplay } from "@/components/analysis/toc-display";
import type { TOCSection } from "@/components/analysis/toc-to-mermaid";
import InteractiveNavigation from "@/components/analysis/interactive-navigation";
import { tocToMermaid } from "@/components/analysis/toc-to-mermaid";

export default function AnalysisPage() {
  const params = useParams();
  const jobId = params.id as string;
  // Use correct TOCSection type for table_of_contents and TOCDisplay
  type ComprehensiveAnalysis = {
    toc?: TOCSection[];
    diagram?: string;
    glossary?: Record<string, string>;
    html_summary?: string;
    figures?: Array<{ url: string; caption?: string; description?: string; page?: number }>;
  };
  type ResultType = {
    table_of_contents?: TOCSection[];
    metadata: {
      title: string;
      authors: string[];
      num_sections?: number;
      processing_method?: string;
    };
    original_abstract?: string;
    quick_summary?: string;
    comprehensive_analysis?: ComprehensiveAnalysis;
    detailed_summary?: Record<string, string>;
  };
  const [instantResult, setInstantResult] = useState<ResultType | null>(null);
  const isInstant = jobId && jobId.startsWith("instant-");
  const { data: job, isLoading, error } = useAnalysis(isInstant ? "" : jobId);

  useEffect(() => {
    if (isInstant) {
      const stored = localStorage.getItem(`instant_result_${jobId}`);
      if (stored) {
        setInstantResult(JSON.parse(stored));
      }
    }
  }, [isInstant, jobId]);

  if (isInstant && instantResult) {
    // INTERACTIVE MODE: Only show interactive navigation UI, never fallback to summary/abstract
    const result = instantResult;
    // Fix: extract TOC array from object if needed (only use toc if it's an array)
    const toc = Array.isArray(result.comprehensive_analysis?.toc) ? result.comprehensive_analysis.toc : undefined;
    let diagram = toc && toc.length > 0 ? tocToMermaid(toc) : result.comprehensive_analysis?.diagram;
    if (toc && diagram) {
      return (
        <div className="flex flex-col min-h-screen">
          <InteractiveNavigation toc={toc} diagram={diagram} />
        </div>
      );
    }
    // If no TOC/diagram, show nothing (or a message)
    return (
      <div className="flex flex-col min-h-screen items-center justify-center">
        <div className="text-gray-500 text-lg mt-12">Interactive navigation data not available for this paper.</div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50">
        <Loader2 className="w-10 h-10 text-blue-600 animate-spin mb-4" />
        <h2 className="text-xl font-semibold text-gray-900">
          Initializing Analysis...
        </h2>
        <p className="text-gray-500">Connecting to server</p>
      </div>
      
    );
  }

  if (error || !job) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 p-6">
        <div className="bg-white p-8 rounded-2xl shadow-sm border max-w-md w-full text-center">
          <div className="mx-auto w-12 h-12 bg-red-100 rounded-full flex items-center justify-center mb-4">
            <AlertCircle className="w-6 h-6 text-red-600" />
          </div>
          <h2 className="text-xl font-bold text-gray-900 mb-2">
            Connection Error
          </h2>
          <p className="text-gray-500 mb-6">
            Failed to load analysis job. Please try again.
          </p>
          <Link href="/" className="text-blue-600 hover:underline font-medium">
            Return to Home
          </Link>
        </div>
      </div>
    );
  }

  if (job.status === "failed") {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 p-6">
        <div className="bg-white p-8 rounded-2xl shadow-sm border max-w-md w-full text-center">
          <div className="mx-auto w-12 h-12 bg-amber-100 rounded-full flex items-center justify-center mb-4">
            <AlertCircle className="w-6 h-6 text-amber-600" />
          </div>
          <h2 className="text-xl font-bold text-gray-900 mb-2">
            Analysis Failed
          </h2>
          <p className="text-gray-500 mb-6">
            {job.error || "An unknown error occurred during processing."}
          </p>
          <Link
            href="/"
            className="px-4 py-2 bg-gray-900 text-white rounded-lg hover:bg-gray-800 transition-colors"
          >
            Try Another Paper
          </Link>
        </div>
      </div>
    );
  }

  if (job.status === "pending" || job.status === "processing") {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50">
        <div className="w-full max-w-md space-y-6 text-center p-6">
          <div className="relative mx-auto w-24 h-24">
            <div className="absolute inset-0 border-4 border-gray-100 rounded-full"></div>
            <div className="absolute inset-0 border-4 border-blue-600 rounded-full border-t-transparent animate-spin"></div>
            <FileText className="absolute inset-0 m-auto w-8 h-8 text-blue-600 animate-pulse" />
          </div>

          <div className="space-y-2">
            <h2 className="text-2xl font-bold text-gray-900">
              {job.status === "pending" ? "Queued" : "Analyzing Paper"}...
            </h2>
            <p className="text-gray-500">
              This usually takes 30-60 seconds depending on complexity.
            </p>
          </div>

          <div className="bg-white rounded-lg p-4 border shadow-sm text-left text-sm space-y-3">
            <div className="flex items-center gap-3 text-gray-600">
              <CheckCircle className="w-4 h-4 text-green-500" />
              <span>Upload received</span>
            </div>
            <div
              className={cn(
                "flex items-center gap-3 transition-colors",
                job.status === "processing"
                  ? "text-gray-900 font-medium"
                  : "text-gray-400"
              )}
            >
              {job.status === "processing" ? (
                <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
              ) : (
                <div className="w-4 h-4 rounded-full border" />
              )}
              <span>Processing with AI models</span>
            </div>
            <div className="flex items-center gap-3 text-gray-400">
              <div className="w-4 h-4 rounded-full border" />
              <span>Generate summaries</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Completed State
  const result = job.result! as ResultType;
  // Interactive mode: check for toc/diagram in comprehensive_analysis
  const toc = Array.isArray(result.comprehensive_analysis?.toc) ? result.comprehensive_analysis.toc : undefined;
  let diagram = toc && toc.length > 0 ? tocToMermaid(toc) : result.comprehensive_analysis?.diagram;
  if (toc && diagram) {
    // Force interactive mode if TOC and diagram are present
    return (
      <div className="flex flex-col min-h-screen">
        <InteractiveNavigation toc={toc} diagram={diagram} />
      </div>
    );
  }
  // ENHANCED MODE: Show original abstract and sectionwise summary for Introduction, Method, Conclusion only
  const introSummary = result.detailed_summary?.Introduction || result.detailed_summary?.introduction;
  const methodSummary = result.detailed_summary?.Method || result.detailed_summary?.method;
  const conclusionSummary = result.detailed_summary?.Conclusion || result.detailed_summary?.conclusion;
  return (
    <div className="flex flex-col min-h-screen">
      <AnalysisDisplay
        title={result.metadata.title}
        abstract={result.original_abstract}
        htmlContent={`<h2>Introduction</h2><p>${introSummary || "No summary available."}</p>
          <h2>Method</h2><p>${methodSummary || "No summary available."}</p>
          <h2>Conclusion</h2><p>${conclusionSummary || "No summary available."}</p>`}
        glossary={result.comprehensive_analysis?.glossary || {}}
        figures={result.comprehensive_analysis?.figures?.map((fig) => ({
          url: fig.url?.startsWith("http") ? fig.url : `http://localhost:8003${fig.url}`,
          caption: fig.caption || fig.description || "Figure",
          page: fig.page || 1,
        })) || []}
        metadata={{
          authors: result.metadata.authors,
          num_sections: result.metadata.num_sections,
          processing_method: result.metadata.processing_method,
        }}
      />
      <PaperChat jobId={jobId} paperTitle={result.metadata.title} />
    </div>
  );
}
