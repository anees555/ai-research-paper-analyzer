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
            {/* Quick Summary Card */}
            <div className="bg-white rounded-xl border shadow-sm p-6 space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-bold text-gray-900 flex items-center gap-2">
                  ✨ AI Summary
                </h2>
              </div>
              <div className="prose prose-blue max-w-none text-gray-600 leading-relaxed">
                <p>{result.quick_summary || result.original_abstract}</p>
              </div>
            </div>

            {/* Detailed Analysis */}
            <div className="bg-white rounded-xl border shadow-sm overflow-hidden">
              <div className="border-b bg-gray-50/50 p-4">
                <h3 className="font-semibold text-gray-900">
                  Detailed Breakdown
                </h3>
              </div>
              <div className="divide-y">
                {result.detailed_summary &&
                  Object.entries(result.detailed_summary).map(
                    ([section, content]) => (
                      <div key={section} className="p-6">
                        <h4 className="text-sm font-bold text-gray-900 uppercase tracking-wider mb-2">
                          {section}
                        </h4>
                        <p className="text-gray-600 text-sm leading-relaxed">
                          {content}
                        </p>
                      </div>
                    )
                  )}
              </div>
            </div>
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
