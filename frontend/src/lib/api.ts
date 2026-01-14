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

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300_000, // 300 second timeout for large files
});

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

export const uploadPaper = async (file: File): Promise<JobResponse> => {
  const formData = new FormData();
  formData.append("file", file);
  const response = await api.post("/analysis/upload", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
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
