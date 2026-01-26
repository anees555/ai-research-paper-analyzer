"use client";

import { useEffect, useState } from "react";
import { testConnection } from "@/lib/api";
import { CheckCircle, XCircle, Loader2 } from "lucide-react";

export function ConnectionStatus() {
  const [isConnected, setIsConnected] = useState<boolean | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const checkConnection = async () => {
      setIsLoading(true);
      try {
        const connected = await testConnection();
        setIsConnected(connected);
      } catch (error) {
        setIsConnected(false);
      } finally {
        setIsLoading(false);
      }
    };

    checkConnection();
    
    // Check connection every 30 seconds
    const interval = setInterval(checkConnection, 30000);
    
    return () => clearInterval(interval);
  }, []);

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-gray-500">
        <Loader2 className="w-4 h-4 animate-spin" />
        Checking backend connection...
      </div>
    );
  }

  if (isConnected) {
    return (
      <div className="flex items-center gap-2 text-sm text-green-600">
        <CheckCircle className="w-4 h-4" />
        Backend connected
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 text-sm text-red-600">
      <XCircle className="w-4 h-4" />
      Backend not available - Please check if the server is running
    </div>
  );
}