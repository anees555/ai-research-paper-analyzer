"use client";

import { useEffect, useState } from "react";
import { getHistory } from "@/lib/api";
import { JobSummary } from "@/types/api";
import Link from "next/link";

export default function HistoryPage() {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getHistory()
      .then(setJobs)
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="flex min-h-screen flex-col p-24 max-w-4xl mx-auto">
      <div className="mb-8 flex justify-between items-center">
        <h1 className="text-3xl font-bold">Analysis History</h1>
        <Link href="/" className="text-indigo-500 hover:text-indigo-400">
          Analyze New Paper
        </Link>
      </div>

      {loading ? (
        <p>Loading history...</p>
      ) : jobs.length === 0 ? (
        <p>No history found.</p>
      ) : (
        <div className="grid gap-4">
          {jobs.map((job) => (
            <div
              key={job.job_id}
              className="p-4 border rounded-lg bg-gray-900 border-gray-700"
            >
              <div className="flex justify-between items-center">
                <div>
                  <Link
                    href={`/analysis/${job.job_id}`}
                    className="text-lg font-semibold hover:text-indigo-400"
                  >
                    {job.file_path.split("/").pop()}
                  </Link>
                  <p className="text-sm text-gray-400">
                    {new Date(job.created_at).toLocaleDateString()}
                  </p>
                </div>
                <span
                  className={`px-3 py-1 rounded-full text-sm ${
                    job.status === "completed"
                      ? "bg-green-900 text-green-200"
                      : job.status === "failed"
                      ? "bg-red-900 text-red-200"
                      : "bg-yellow-900 text-yellow-200"
                  }`}
                >
                  {job.status}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
