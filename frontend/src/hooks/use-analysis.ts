import { useQuery } from "@tanstack/react-query";
import { getJobStatus } from "@/lib/api";
import { JobResponse } from "@/types/api";

export function useAnalysis(jobId: string) {
  return useQuery({
    queryKey: ["analysis", jobId],
    queryFn: () => getJobStatus(jobId),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const data = query.state.data as JobResponse | undefined;
      // Stop polling if completed or failed
      if (data?.status === "completed" || data?.status === "failed") {
        return false;
      }
      // Poll every 2 seconds
      return 2000;
    },
    refetchOnWindowFocus: false,
  });
}
