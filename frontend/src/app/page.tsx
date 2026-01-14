"use client";

import { FileUpload } from "@/components/upload/file-upload";
import { BookOpen, FileText, Clock } from "lucide-react";
import { useEffect, useState } from "react";
import Link from "next/link";
import { cn } from "@/lib/utils";
import { useAuth } from "@/contexts/auth-context";
import { UserNav } from "@/components/nav/user-nav";

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-50/50 flex flex-col">
      <Navbar />
      
      <div className="flex-1 flex flex-col items-center justify-center p-6 pb-20">
        <div className="w-full max-w-2xl text-center mb-10 space-y-4">
            <h2 className="text-4xl font-extrabold tracking-tight text-gray-900 sm:text-5xl">
                Analyze Papers at <span className="text-blue-600">Light Speed</span>
            </h2>
            <p className="text-lg text-gray-600 max-w-lg mx-auto leading-relaxed">
                Upload your research paper PDF to get instant AI-generated summaries, 
                metadata extraction, and comprehensive analysis.
            </p>
        </div>
        
        <FileUpload />

        <div className="w-full max-w-xl mt-12">
            <HistoryList />
        </div>
        
        <div className="mt-16 grid grid-cols-1 sm:grid-cols-3 gap-8 max-w-4xl w-full text-center">
             <Feature 
                title="Smart Summaries" 
                desc="Get concise quick summaries or deep detailed breakdowns."
             />
             <Feature 
                title="Metadata Extraction" 
                desc="Automatically extract authors, sections, and citations."
             />
             <Feature 
                title="Batch Processing" 
                desc="Handle multiple papers efficiently in the background."
             />
        </div>
      </div>
    </main>
  );
}

function Feature({ title, desc }: { title: string; desc: string }) {
    return (
        <div className="space-y-2">
            <h4 className="font-semibold text-gray-900">{title}</h4>
            <p className="text-sm text-gray-500">{desc}</p>
        </div>
    )
}

function Navbar() {
  const { user, isLoading } = useAuth();

  return (
    <header className="border-b bg-white sticky top-0 z-30">
      <div className="container mx-auto px-6 h-16 flex items-center justify-between gap-2">
        <Link href="/" className="flex items-center gap-2">
          <div className="bg-blue-600 p-1.5 rounded-lg">
            <BookOpen className="w-5 h-5 text-white" />
          </div>
          <h1 className="font-bold text-xl tracking-tight text-gray-900">
            ResearchAI
          </h1>
        </Link>
        <div className="flex items-center gap-4">
          {isLoading ? (
            <div className="w-8 h-8 rounded-full bg-gray-100 animate-pulse" />
          ) : user ? (
            <UserNav />
          ) : (
            <>
              <Link
                href="/login"
                className="text-sm font-medium text-gray-600 hover:text-blue-600"
              >
                Login
              </Link>
              <Link
                href="/register"
                className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-500 transition-colors shadow-sm"
              >
                Sign Up
              </Link>
            </>
          )}
        </div>
      </div>
    </header>
  );
}

function HistoryList() {
    const [history, setHistory] = useState<any[]>([]);

    useEffect(() => {
        const saved = localStorage.getItem("paper_history");
        if (saved) {
            try {
                setHistory(JSON.parse(saved));
            } catch (e) {
                console.error("Failed to parse history", e);
            }
        }
    }, []);

    if (history.length === 0) return null;

    return (
        <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm text-gray-500 font-medium px-2">
                <Clock className="w-4 h-4" />
                <span>Recent Analysis</span>
            </div>
            <div className="bg-white rounded-xl border shadow-sm divide-y overflow-hidden">
                {history.map((item, i) => (
                    <Link 
                        key={i} 
                        href={`/analysis/${item.job_id}`}
                        className="flex items-center gap-3 p-4 hover:bg-gray-50 transition-colors"
                    >
                        <div className="p-2 bg-blue-50 text-blue-600 rounded-lg">
                            <FileText className="w-4 h-4" />
                        </div>
                        <div className="flex-1 min-w-0">
                            <h4 className="text-sm font-medium text-gray-900 truncate">
                                {item.title || "Untitled Paper"}
                            </h4>
                            <p className="text-xs text-gray-500">
                                {new Date(item.date).toLocaleDateString()}
                            </p>
                        </div>
                        <div className="text-xs text-gray-400">
                            View
                        </div>
                    </Link>
                ))}
            </div>
        </div>
    )
}
