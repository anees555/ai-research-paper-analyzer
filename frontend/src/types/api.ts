export type JobStatus = "pending" | "processing" | "completed" | "failed";

export interface User {
  id: string;
  email: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
}

export interface JobSummary {
    job_id: string;
    status: JobStatus;
    created_at: string;
    file_path: string;
}

export type MessageRole = "user" | "assistant" | "system";

export interface PaperMetadata {
  title: string;
  authors: string[];
  paper_id: string;
  num_sections: number;
  processing_method: string;
}

export interface AnalysisResult {
  metadata: PaperMetadata;
  quick_summary?: string;
  detailed_summary?: Record<string, string>;
  comprehensive_analysis?: {
    main_contribution: string;
    methodology: string;
    key_findings: string;
    limitations: string;
    future_work: string;
    ai_insights: string[];
  };
  original_abstract?: string;
}

export interface JobResponse {
  job_id: string;
  status: JobStatus;
  created_at: string;
  error?: string;
  result?: AnalysisResult;
}

// Chat types
export interface ChatMessage {
  role: MessageRole;
  content: string;
  timestamp?: string;
}

export interface ChatSource {
  section: string;
  similarity: number;
}

export interface ChatRequest {
  job_id: string;
  question: string;
  conversation_history?: ChatMessage[];
}

export interface ChatResponse {
  message: string;
  sources: ChatSource[];
  confidence: number;
  timestamp: string;
}

export interface ChatStatus {
  job_id: string;
  indexed: boolean;
  ready: boolean;
}
