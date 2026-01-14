"use client";

import { useState, useRef, useEffect } from "react";
import {
  Send,
  Bot,
  User,
  Loader2,
  MessageCircle,
  X,
  Sparkles,
  AlertCircle,
} from "lucide-react";
import { askQuestion, indexPaperForChat, getChatStatus } from "@/lib/api";
import { ChatMessage, ChatResponse, ChatSource } from "@/types/api";
import { cn } from "@/lib/utils";

interface PaperChatProps {
  jobId: string;
  paperTitle: string;
}

interface DisplayMessage {
  role: "user" | "assistant";
  content: string;
  sources?: ChatSource[];
  confidence?: number;
  timestamp: Date;
}

export function PaperChat({ jobId, paperTitle }: PaperChatProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isIndexing, setIsIndexing] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Check/initialize chat when opened
  useEffect(() => {
    if (isOpen && !isReady && !isIndexing) {
      initializeChat();
    }
  }, [isOpen]);

  // Focus input when chat opens
  useEffect(() => {
    if (isOpen && isReady) {
      inputRef.current?.focus();
    }
  }, [isOpen, isReady]);

  const initializeChat = async () => {
    setIsIndexing(true);
    setError(null);

    try {
      // Check if already indexed
      const status = await getChatStatus(jobId);

      if (status.ready) {
        setIsReady(true);
        addWelcomeMessage();
      } else {
        // Index the paper
        await indexPaperForChat(jobId);
        setIsReady(true);
        addWelcomeMessage();
      }
    } catch (err) {
      console.error("Failed to initialize chat:", err);
      setError("Failed to initialize chat. Please try again.");
    } finally {
      setIsIndexing(false);
    }
  };

  const addWelcomeMessage = () => {
    setMessages([
      {
        role: "assistant",
        content: `Hi! I'm ready to answer questions about "${paperTitle}". Ask me anything about the methodology, findings, or concepts in this paper.`,
        timestamp: new Date(),
      },
    ]);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!input.trim() || isLoading) return;

    const userMessage: DisplayMessage = {
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setError(null);

    try {
      // Build conversation history for context
      const history: ChatMessage[] = messages.map((m) => ({
        role: m.role,
        content: m.content,
        timestamp: m.timestamp.toISOString(),
      }));

      const response = await askQuestion(jobId, userMessage.content, history);

      const assistantMessage: DisplayMessage = {
        role: "assistant",
        content: response.message,
        sources: response.sources,
        confidence: response.confidence,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      console.error("Failed to get response:", err);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "I'm sorry, I encountered an error processing your question. Please try again.",
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // Suggested questions
  const suggestions = [
    "What is the main contribution of this paper?",
    "What methodology was used?",
    "What are the key findings?",
    "What are the limitations?",
  ];

  const handleSuggestionClick = (question: string) => {
    setInput(question);
    inputRef.current?.focus();
  };

  return (
    <>
      {/* Chat Toggle Button */}
      <button
        onClick={() => setIsOpen(true)}
        className={cn(
          "fixed bottom-6 right-6 p-4 bg-blue-600 text-white rounded-full shadow-lg",
          "hover:bg-blue-700 transition-all duration-200 hover:scale-105",
          "flex items-center gap-2 z-50",
          isOpen && "hidden"
        )}
      >
        <MessageCircle className="w-6 h-6" />
        <span className="font-medium">Ask about this paper</span>
      </button>

      {/* Chat Panel */}
      <div
        className={cn(
          "fixed bottom-6 right-6 w-[420px] h-[600px] bg-white rounded-2xl shadow-2xl",
          "flex flex-col overflow-hidden z-50 border",
          "transition-all duration-300 ease-out",
          isOpen
            ? "opacity-100 translate-y-0"
            : "opacity-0 translate-y-4 pointer-events-none"
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b bg-gradient-to-r from-blue-600 to-blue-700 text-white">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-white/20 rounded-lg">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-semibold">Paper Assistant</h3>
              <p className="text-xs text-blue-100 line-clamp-1 max-w-[240px]">
                {paperTitle}
              </p>
            </div>
          </div>
          <button
            onClick={() => setIsOpen(false)}
            className="p-2 hover:bg-white/20 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
          {isIndexing ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-500">
              <Loader2 className="w-8 h-8 animate-spin mb-3 text-blue-600" />
              <p className="font-medium">Preparing paper for Q&A...</p>
              <p className="text-sm">This may take a moment</p>
            </div>
          ) : error ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-500">
              <AlertCircle className="w-8 h-8 mb-3 text-red-500" />
              <p className="font-medium text-red-600">{error}</p>
              <button
                onClick={initializeChat}
                className="mt-3 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Retry
              </button>
            </div>
          ) : (
            <>
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={cn(
                    "flex gap-3",
                    message.role === "user" ? "justify-end" : "justify-start"
                  )}
                >
                  {message.role === "assistant" && (
                    <div className="flex-shrink-0 w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                      <Bot className="w-4 h-4 text-blue-600" />
                    </div>
                  )}

                  <div
                    className={cn(
                      "max-w-[80%] rounded-2xl px-4 py-3",
                      message.role === "user"
                        ? "bg-blue-600 text-white"
                        : "bg-white border text-blue-400 shadow-sm"
                    )}
                  >
                    <p className="text-sm whitespace-pre-wrap">
                      {message.content}
                    </p>

                    {/* Sources */}
                    {message.sources && message.sources.length > 0 && (
                      <div className="mt-2 pt-2 border-t border-gray-100">
                        <p className="text-xs text-gray-400 mb-1">Sources:</p>
                        <div className="flex flex-wrap gap-1">
                          {message.sources.map((source, i) => (
                            <span
                              key={i}
                              className="text-xs bg-blue-50 text-blue-600 px-2 py-0.5 rounded"
                            >
                              {source.section}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Confidence */}
                    {message.confidence !== undefined &&
                      message.confidence > 0 && (
                        <div className="mt-1">
                          <span
                            className={cn(
                              "text-xs px-2 py-0.5 rounded",
                              message.confidence > 0.7
                                ? "bg-green-50 text-green-600"
                                : message.confidence > 0.4
                                ? "bg-yellow-50 text-yellow-600"
                                : "bg-gray-50 text-gray-500"
                            )}
                          >
                            {Math.round(message.confidence * 100)}% confidence
                          </span>
                        </div>
                      )}
                  </div>

                  {message.role === "user" && (
                    <div className="flex-shrink-0 w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
                      <User className="w-4 h-4 text-white" />
                    </div>
                  )}
                </div>
              ))}

              {/* Loading indicator */}
              {isLoading && (
                <div className="flex gap-3 justify-start">
                  <div className="flex-shrink-0 w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                    <Bot className="w-4 h-4 text-blue-600" />
                  </div>
                  <div className="bg-white border shadow-sm rounded-2xl px-4 py-3">
                    <div className="flex gap-1">
                      <div
                        className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                        style={{ animationDelay: "0ms" }}
                      />
                      <div
                        className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                        style={{ animationDelay: "150ms" }}
                      />
                      <div
                        className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                        style={{ animationDelay: "300ms" }}
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* Suggestions (show only when no messages or just welcome) */}
              {messages.length <= 1 && !isLoading && (
                <div className="space-y-2">
                  <p className="text-xs text-gray-400">Try asking:</p>
                  <div className="flex flex-wrap gap-2">
                    {suggestions.map((q, i) => (
                      <button
                        key={i}
                        onClick={() => handleSuggestionClick(q)}
                        className="text-xs bg-white border rounded-lg px-3 py-1.5 hover:bg-blue-50 hover:border-blue-200 transition-colors"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input Area */}
        <form onSubmit={handleSubmit} className="p-4 border-t bg-white">
          <div className="flex gap-2">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about this paper..."
              disabled={!isReady || isLoading}
              className={cn(
                "flex-1 px-4 py-3 border rounded-xl text-sm",
                "focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent",
                "disabled:bg-gray-100 disabled:cursor-not-allowed"
              )}
            />
            <button
              type="submit"
              disabled={!input.trim() || !isReady || isLoading}
              className={cn(
                "px-4 py-3 bg-blue-600 text-white rounded-xl",
                "hover:bg-blue-700 transition-colors",
                "disabled:bg-gray-300 disabled:cursor-not-allowed",
                "flex items-center justify-center"
              )}
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
        </form>
      </div>
    </>
  );
}
