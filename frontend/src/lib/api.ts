import axios from "axios";
import {
  JobResponse,
  AnalysisResult,
  ChatRequest,
  ChatResponse,
  ChatMessage,
  ChatStatus,
  User,
  AuthResponse,
  JobSummary
} from "@/types/api";

// Processing mode types
export type ProcessingMode = "fast" | "balanced" | "comprehensive" | "enhanced";

export interface ProcessingModeInfo {
  name: string;
  estimated_time: string;
  description: string;
  features: string[];
  ai_models: boolean | string;
  recommended_for: string;
}

export interface InstantAnalysisResponse {
  status: string;
  processing_mode: ProcessingMode;
  analysis_result: AnalysisResult;
  processing_info: {
    instant_processing: boolean;
    estimated_time: string;
  };
}

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8003/api/v1";

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300_000, // 300 second timeout for large files
});

// Add connection test function
export const testConnection = async (): Promise<boolean> => {
  try {
    const response = await api.get('/analysis/modes');
    return response.status === 200;
  } catch (error) {
    console.error('Backend connection test failed:', error);
    return false;
  }
};

// Add auth interceptor
api.interceptors.request.use((config) => {
    const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null;
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

export const login = async (email: string, password: string): Promise<AuthResponse> => {
    const formData = new FormData();
    formData.append('username', email); // OAuth2 expects username
    formData.append('password', password);
    const response = await api.post<AuthResponse>('/auth/login/access-token', formData);
    if (response.data.access_token) {
        localStorage.setItem('token', response.data.access_token);
    }
    return response.data;
};

export const register = async (email: string, password: string): Promise<User> => {
    const response = await api.post<User>('/auth/register', { email, password });
    return response.data;
};

export const getMe = async (): Promise<User> => {
    const response = await api.get<User>('/users/me');
    return response.data;
};

export const getHistory = async (): Promise<JobSummary[]> => {
    const response = await api.get<JobSummary[]>('/users/me/history');
    return response.data;
};

export const logout = () => {
    localStorage.removeItem('token');
    window.location.href = '/login';
};

export const uploadPaper = async (
  file: File,
  mode: ProcessingMode = "enhanced"
): Promise<JobResponse> => {
  const formData = new FormData();
  formData.append("file", file);
  
  const response = await api.post(`/analysis/upload?mode=${mode}`, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
};

export const uploadPaperEnhanced = async (
  file: File
): Promise<JobResponse> => {
  const formData = new FormData();
  formData.append("file", file);
  
  const response = await api.post(`/analysis/enhanced`, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
};

export const uploadPaperProfessional = async (
  file: File
): Promise<JobResponse> => {
  const formData = new FormData();
  formData.append("file", file);
  
  const response = await api.post(`/analysis/professional`, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
};

export const uploadPaperInstant = async (
  file: File,
  mode: ProcessingMode = "enhanced"
): Promise<InstantAnalysisResponse> => {
  if (mode === "comprehensive") {
    throw new Error("Comprehensive mode requires background processing. Use uploadPaper instead.");
  }
  
  const formData = new FormData();
  formData.append("file", file);
  
  const response = await api.post(`/analysis/analyze-instant?mode=${mode}`, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
};

export const getProcessingModes = async (): Promise<{
  modes: Record<ProcessingMode, ProcessingModeInfo>;
  default_mode: ProcessingMode;
  instant_analysis_supported: ProcessingMode[];
  background_processing_required: ProcessingMode[];
}> => {
  const response = await api.get("/analysis/modes");
  return response.data;
};

export const getJobStatus = async (jobId: string): Promise<JobResponse> => {
  const response = await api.get(`/analysis/status/${jobId}`);
  return response.data;
};

// Chat API functions
export const askQuestion = async (
  jobId: string,
  question: string,
  conversationHistory?: ChatMessage[]
): Promise<ChatResponse> => {
  const response = await api.post<ChatResponse>("/chat/ask", {
    job_id: jobId,
    question,
    conversation_history: conversationHistory,
  });
  return response.data;
};

export const indexPaperForChat = async (
  jobId: string
): Promise<{ status: string; job_id: string }> => {
  const response = await api.post(`/chat/index/${jobId}`);
  return response.data;
};

export const getChatStatus = async (jobId: string): Promise<ChatStatus> => {
  const response = await api.get<ChatStatus>(`/chat/status/${jobId}`);
  return response.data;
};
