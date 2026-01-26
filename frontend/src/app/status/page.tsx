"use client";

import { SystemStatus } from "@/components/status/system-status";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";

export default function StatusPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto py-8">
        <div className="mb-8">
          <Link 
            href="/" 
            className="inline-flex items-center gap-2 text-blue-600 hover:text-blue-700"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Home
          </Link>
        </div>
        
        <SystemStatus />
      </div>
    </div>
  );
}