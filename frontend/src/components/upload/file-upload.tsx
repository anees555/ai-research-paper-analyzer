"use client";

import { useState, useCallback, useRef } from "react";
import { UploadCloud, File, Loader2, X } from "lucide-react";
import { uploadPaper } from "@/lib/api";
import { useRouter } from "next/navigation";
import { cn } from "@/lib/utils";

export function FileUpload() {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

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
    if (file.type !== "application/pdf") {
      setError("Only PDF files are allowed.");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError("File size must be under 10MB.");
      return;
    }

    setError(null);
    setIsUploading(true);

    try {
      const response = await uploadPaper(file);
      
      // Save to local history
      const historyItem = {
        job_id: response.job_id,
        title: file.name.replace(".pdf", ""),
        date: new Date().toISOString(),
        status: 'pending'
      };
      
      const existing = JSON.parse(localStorage.getItem("paper_history") || "[]");
      localStorage.setItem("paper_history", JSON.stringify([historyItem, ...existing].slice(0, 10)));

      // Redirect to analysis page
      router.push(`/analysis/${response.job_id}`);
    } catch (err: any) {
      console.error(err);
      setError("Upload failed. Please try again.");
      setIsUploading(false);
    }
  };

  return (
    <div className="w-full max-w-xl mx-auto">
      <div
        className={cn(
          "relative border-2 border-dashed rounded-xl p-10 transition-all duration-200 ease-in-out text-center cursor-pointer",
          isDragOver
            ? "border-blue-500 bg-blue-50/50"
            : "border-gray-200 hover:border-gray-300 hover:bg-gray-50/50",
          isUploading && "opacity-50 pointer-events-none"
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          type="file"
          ref={fileInputRef}
          className="hidden"
          accept=".pdf"
          onChange={handleFileSelect}
        />

        <div className="flex flex-col items-center gap-4">
            <div className="p-4 bg-white rounded-full shadow-sm border border-gray-100">
                <UploadCloud className="w-8 h-8 text-blue-500" />
            </div>
            
            <div className="space-y-1">
                <h3 className="font-semibold text-lg text-gray-900">
                    Upload Research Paper
                </h3>
                <p className="text-sm text-gray-500">
                    Drag & drop your PDF here, or click to browse
                </p>
            </div>
            
            <div className="text-xs text-gray-400">
                Supports PDF up to 10MB
            </div>
        </div>

        {isUploading && (
          <div className="absolute inset-0 bg-white/80 flex flex-col items-center justify-center rounded-xl backdrop-blur-sm">
            <Loader2 className="w-8 h-8 text-blue-500 animate-spin mb-2" />
            <p className="text-sm font-medium text-blue-600">Uploading Paper...</p>
          </div>
        )}
      </div>

      {error && (
        <div className="mt-4 p-3 bg-red-50 text-red-600 rounded-lg text-sm flex items-center gap-2">
            <X className="w-4 h-4" />
            {error}
        </div>
      )}
    </div>
  );
}
