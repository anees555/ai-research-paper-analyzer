"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { UploadCloud, File, Loader2, X, ChevronDown, Settings } from "lucide-react";
import { uploadPaper, getProcessingModes, ProcessingMode, ProcessingModeInfo } from "@/lib/api";
import { useRouter } from "next/navigation";
import { cn } from "@/lib/utils";

export function FileUpload() {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedMode, setSelectedMode] = useState<ProcessingMode>("enhanced");
  const [availableModes, setAvailableModes] = useState<Record<ProcessingMode, ProcessingModeInfo>>({});
  const [showModeSelector, setShowModeSelector] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  // Load available processing modes on component mount
  useEffect(() => {
    const loadModes = async () => {
      try {
        const modesData = await getProcessingModes();
        setAvailableModes(modesData.modes);
      } catch (error) {
        console.error("Failed to load processing modes:", error);
      }
    };
    loadModes();
  }, []);

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
      setError("Only PDF files are allowed. Please select a PDF document.");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError("File size must be under 10MB. Please choose a smaller file.");
      return;
    }

    setError(null);
    setIsUploading(true);

    try {
      // Add timeout and better error handling
      const response = await uploadPaper(file, selectedMode);
      
      if (!response || !response.job_id) {
        throw new Error("Invalid server response - no job ID received");
      }
      
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
      console.error("Upload error:", err);
      
      // More specific error messages
      if (err.code === 'NETWORK_ERROR' || err.message.includes('Network')) {
        setError("Connection failed. Please check if the backend server is running.");
      } else if (err.response?.status === 413) {
        setError("File too large. Please choose a smaller PDF file.");
      } else if (err.response?.status === 415) {
        setError("Unsupported file type. Please upload a PDF file.");
      } else if (err.response?.status >= 500) {
        setError("Server error. Please try again in a moment.");
      } else {
        setError(`Upload failed: ${err.message || 'Unknown error'}. Please try again.`);
      }
      
      setIsUploading(false);
    }
  };

  return (
    <div className="w-full max-w-xl mx-auto space-y-6">
      {/* Processing Mode Selector */}
      <div className="bg-white rounded-xl p-6 border border-gray-200 shadow-sm">
        <div className="flex items-center gap-3 mb-4">
          <Settings className="w-5 h-5 text-blue-500" />
          <h3 className="font-semibold text-gray-900">Processing Mode</h3>
        </div>
        
        <div className="relative">
          <button
            onClick={() => setShowModeSelector(!showModeSelector)}
            className="w-full flex items-center justify-between p-3 border border-gray-200 rounded-lg hover:border-gray-300 transition-colors"
          >
            <div className="text-left">
              <div className="font-medium text-gray-900">
                {availableModes[selectedMode]?.name || "Enhanced Professional Analysis"}
              </div>
              <div className="text-sm text-gray-500">
                {availableModes[selectedMode]?.estimated_time || "90-150 seconds"}
              </div>
            </div>
            <ChevronDown className={cn(
              "w-4 h-4 text-gray-400 transition-transform",
              showModeSelector && "transform rotate-180"
            )} />
          </button>
          
          {showModeSelector && (
            <div className="absolute z-10 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg">
              {Object.entries(availableModes).map(([mode, info]) => (
                <button
                  key={mode}
                  onClick={() => {
                    setSelectedMode(mode as ProcessingMode);
                    setShowModeSelector(false);
                  }}
                  className={cn(
                    "w-full text-left p-3 hover:bg-gray-50 first:rounded-t-lg last:rounded-b-lg transition-colors",
                    selectedMode === mode && "bg-blue-50 border-blue-200"
                  )}
                >
                  <div className="font-medium text-gray-900">{info.name}</div>
                  <div className="text-sm text-gray-500">{info.estimated_time}</div>
                  <div className="text-xs text-gray-400 mt-1">{info.description}</div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* File Upload Area */}
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
            <p className="text-sm font-medium text-blue-600">
              Uploading Paper... ({availableModes[selectedMode]?.name})
            </p>
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
