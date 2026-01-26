"use client";

import { useAnalysis } from "@/hooks/use-analysis";
import { Loader2, AlertCircle, CheckCircle, FileText } from "lucide-react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { cn } from "@/lib/utils";
import { PaperChat } from "@/components/chat/paper-chat";

export default function AnalysisPage() {
  const params = useParams();
  const jobId = params.id as string;
  const { data: job, isLoading, error } = useAnalysis(jobId);

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
  const result = job.result!;

  return (
    <div className="min-h-screen bg-gray-50 pb-20">
      {/* Header */}
      <header className="bg-white border-b sticky top-0 z-10">
        <div className="container mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="p-2 hover:bg-gray-100 rounded-full transition-colors"
            >
              <FileText className="w-5 h-5 text-gray-600" />
            </Link>
            <div>
              <h1
                className="font-semibold text-gray-900 line-clamp-1 max-w-md"
                title={result.metadata.title}
              >
                {result.metadata.title}
              </h1>
              <p className="text-xs text-gray-500">
                Processed in {new Date(job.created_at).toLocaleDateString()}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="bg-green-100 text-green-700 text-xs px-2.5 py-1 rounded-full font-medium">
              Analysis Complete
            </span>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-6">
            {/* Original Abstract Card */}
            {result.original_abstract && (
              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-100 shadow-sm p-6 space-y-4">
                <div className="flex items-center gap-2">
                  <h2 className="text-lg font-bold text-gray-900 flex items-center gap-2">
                    Original Abstract
                  </h2>
                  <span className="bg-blue-100 text-blue-700 text-xs px-2 py-1 rounded-full font-medium">
                    Source
                  </span>
                </div>
                <div className="prose prose-blue max-w-none text-gray-700 leading-relaxed bg-white rounded-lg p-4 border border-blue-100">
                  <p className="italic">{result.original_abstract}</p>
                </div>
              </div>
            )}

            {/* Quick Summary Card */}
            <div className="bg-white rounded-xl border shadow-sm p-6 space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-bold text-gray-900 flex items-center gap-2">
                  AI Summary
                </h2>
              </div>
              {/* Check if enhanced HTML summary exists */}
              {result.comprehensive_analysis?.html_summary ? (
                <div className="prose prose-blue max-w-none text-gray-600 leading-relaxed">
                  <div 
                    dangerouslySetInnerHTML={{ 
                      __html: result.comprehensive_analysis.html_summary 
                    }} 
                  />
                </div>
              ) : (
                <div className="prose prose-blue max-w-none text-gray-600 leading-relaxed">
                  <p>{result.quick_summary || result.original_abstract}</p>
                </div>
              )}
            </div>

            {/* Detailed Analysis */}
            <div className="bg-white rounded-xl border shadow-sm overflow-hidden">
              <div className="border-b bg-gray-50/50 p-4">
                <h3 className="text-xl font-bold text-gray-900 flex items-center gap-2">
                  Detailed Breakdown
                  {result.detailed_summary && (
                    <span className="bg-blue-100 text-blue-700 text-xs px-2 py-1 rounded-full font-medium">
                      {Object.keys(result.detailed_summary).length} sections
                    </span>
                  )}
                </h3>
              </div>
              <div className="divide-y">
                {result.detailed_summary && Object.keys(result.detailed_summary).length > 0 ? (
                  Object.entries(result.detailed_summary).map(
                    ([section, content]) => (
                      <div key={section} className="p-6">
                        <h4 className="text-lg font-bold text-gray-900 mb-3 flex items-center gap-2">
                          <span className="w-2 h-2 bg-blue-600 rounded-full"></span>
                          {section}
                        </h4>
                        <div className="prose max-w-none text-gray-700 leading-relaxed">
                          <p className="text-base">{content}</p>
                        </div>
                      </div>
                    )
                  )
                ) : (
                  <div className="p-6 text-center text-gray-500">
                    <p>No detailed sections available for this document.</p>
                  </div>
                )}
              </div>
            </div>

            {/* Enhanced Features - Figures */}
            {result.comprehensive_analysis?.figures && result.comprehensive_analysis.figures.length > 0 && (
              <div className="bg-white rounded-xl border shadow-sm overflow-hidden">
                <div className="border-b bg-gray-50/50 p-4">
                  <h3 className="font-semibold text-gray-900">
                    Figures & Illustrations
                  </h3>
                </div>
                <div className="p-6 space-y-6">
                  {result.comprehensive_analysis.figures.map((figure: any, index: number) => (
                    <div key={index} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex items-start gap-4">
                        <div className="flex-shrink-0">
                          <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-3">
                            <span className="text-blue-600 font-semibold text-sm">
                              Fig {index + 1}
                            </span>
                          </div>
                          <p className="text-xs text-gray-500 text-center">
                            Page {figure.page}
                          </p>
                        </div>
                        <div className="flex-1 min-w-0">
                          {/* Figure Image */}
                          {figure.url && (
                            <div className="mb-4">
                              <img 
                                src={`http://localhost:8003${figure.url}`}
                                alt={figure.caption || `Figure ${index + 1}`}
                                className="max-w-full h-auto rounded-lg border border-gray-200 shadow-sm"
                                onError={(e) => {
                                  e.currentTarget.style.display = 'none';
                                  e.currentTarget.nextElementSibling!.style.display = 'block';
                                }}
                              />
                              <div className="hidden bg-gray-100 p-4 rounded-lg text-center">
                                <span className="text-gray-500 text-sm">
                                  Figure {index + 1} (Image not available)
                                </span>
                              </div>
                            </div>
                          )}
                          
                          {/* Figure Caption */}
                          <h4 className="text-sm font-medium text-gray-900 mb-2">
                            {figure.caption || figure.description || `Figure ${index + 1} from research paper`}
                          </h4>
                          
                          {/* Figure Details */}
                          {figure.width && figure.height && (
                            <p className="text-xs text-gray-400">
                              {figure.width} × {figure.height} pixels
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Enhanced Features - Technical Glossary */}
            {result.comprehensive_analysis?.glossary && Object.keys(result.comprehensive_analysis.glossary).length > 0 && (
              <div className="bg-white rounded-xl border shadow-sm overflow-hidden">
                <div className="border-b bg-gray-50/50 p-4">
                  <h3 className="font-semibold text-gray-900">
                    📚 Technical Glossary
                  </h3>
                </div>
                <div className="p-6">
                  <div className="grid gap-4">
                    {Object.entries(result.comprehensive_analysis.glossary).map(([term, definition]) => (
                      <div key={term} className="border-l-4 border-blue-200 pl-4">
                        <h4 className="font-semibold text-gray-900 text-sm mb-1">
                          {term}
                        </h4>
                        <p className="text-gray-600 text-sm leading-relaxed">
                          {definition}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Metadata */}
            <div className="bg-white rounded-xl border shadow-sm p-6">
              <h3 className="font-semibold text-gray-900 mb-4">
                Paper Details
              </h3>
              <div className="space-y-4 text-sm">
                <div>
                  <span className="block text-gray-500 text-xs mb-1">
                    AUTHORS
                  </span>
                  <div className="flex flex-wrap gap-2">
                    {result.metadata.authors.map((author, i) => (
                      <span
                        key={i}
                        className="bg-gray-100 text-gray-700 px-2 py-1 rounded-md text-xs font-medium"
                      >
                        {author}
                      </span>
                    ))}
                  </div>
                </div>
                <div>
                  <span className="block text-gray-500 text-xs mb-1">
                    SECTIONS
                  </span>
                  <span className="text-gray-900">
                    {result.metadata.num_sections} detected
                  </span>
                </div>
              </div>
            </div>

            {/* AI Insights */}
            <div className="bg-blue-50 rounded-xl border border-blue-100 p-6">
              <h3 className="font-semibold text-blue-900 mb-4 flex items-center gap-2">
                🧠 Key Contribution
              </h3>
              <p className="text-sm text-blue-800 leading-relaxed">
                {result.comprehensive_analysis?.main_contribution ||
                  "No contribution analysis available."}
              </p>
            </div>
          </div>
        </div>
      </main>

      {/* Chat Widget */}
      <PaperChat jobId={jobId} paperTitle={result.metadata.title} />
    </div>
  );
}
