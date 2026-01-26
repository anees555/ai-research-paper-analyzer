"use client";

import { useEffect, useState } from "react";
import { testConnection, getProcessingModes } from "@/lib/api";
import { CheckCircle, XCircle, Loader2, AlertTriangle, Settings } from "lucide-react";

interface DiagnosticResult {
  name: string;
  status: 'success' | 'error' | 'warning' | 'loading';
  message: string;
  details?: string;
}

export function SystemStatus() {
  const [diagnostics, setDiagnostics] = useState<DiagnosticResult[]>([]);
  const [isRunning, setIsRunning] = useState(false);

  const runDiagnostics = async () => {
    setIsRunning(true);
    const results: DiagnosticResult[] = [];

    // Test 1: Backend Connection
    results.push({
      name: "Backend Connection",
      status: 'loading',
      message: "Testing connection..."
    });
    setDiagnostics([...results]);

    try {
      const connected = await testConnection();
      results[0] = {
        name: "Backend Connection",
        status: connected ? 'success' : 'error',
        message: connected ? "Backend is accessible" : "Backend connection failed",
        details: connected ? "API server responding on http://localhost:8003" : "Check if backend server is running"
      };
    } catch (error) {
      results[0] = {
        name: "Backend Connection",
        status: 'error',
        message: "Connection test failed",
        details: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`
      };
    }
    setDiagnostics([...results]);

    // Test 2: Processing Modes
    results.push({
      name: "Processing Modes",
      status: 'loading',
      message: "Loading processing modes..."
    });
    setDiagnostics([...results]);

    try {
      const modesData = await getProcessingModes();
      const hasEnhanced = Object.values(modesData.modes).some(mode => 
        mode.name.toLowerCase().includes('enhanced')
      );
      
      results[1] = {
        name: "Processing Modes",
        status: hasEnhanced ? 'success' : 'warning',
        message: hasEnhanced 
          ? `All modes available (${Object.keys(modesData.modes).length} total)`
          : `Modes loaded but Enhanced mode missing (${Object.keys(modesData.modes).length} available)`,
        details: Object.values(modesData.modes).map(m => m.name).join(', ')
      };
    } catch (error) {
      results[1] = {
        name: "Processing Modes",
        status: 'error',
        message: "Failed to load processing modes",
        details: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`
      };
    }
    setDiagnostics([...results]);

    // Test 3: Local Storage
    results.push({
      name: "Browser Storage",
      status: 'loading',
      message: "Checking browser storage..."
    });
    setDiagnostics([...results]);

    try {
      localStorage.setItem('test', 'value');
      localStorage.removeItem('test');
      results[2] = {
        name: "Browser Storage",
        status: 'success',
        message: "Local storage is working",
        details: "History and settings will be saved"
      };
    } catch (error) {
      results[2] = {
        name: "Browser Storage",
        status: 'warning',
        message: "Local storage not available",
        details: "Upload history may not be saved"
      };
    }
    setDiagnostics([...results]);

    setIsRunning(false);
  };

  useEffect(() => {
    runDiagnostics();
  }, []);

  const getStatusIcon = (status: DiagnosticResult['status']) => {
    switch (status) {
      case 'success': return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'error': return <XCircle className="w-5 h-5 text-red-500" />;
      case 'warning': return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
      case 'loading': return <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />;
    }
  };

  const getStatusColor = (status: DiagnosticResult['status']) => {
    switch (status) {
      case 'success': return 'bg-green-50 border-green-200';
      case 'error': return 'bg-red-50 border-red-200';
      case 'warning': return 'bg-yellow-50 border-yellow-200';
      case 'loading': return 'bg-blue-50 border-blue-200';
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6 space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">System Status</h2>
        <p className="text-gray-600">Diagnostic information for troubleshooting</p>
      </div>

      <div className="space-y-4">
        {diagnostics.map((diagnostic, index) => (
          <div
            key={index}
            className={`p-4 rounded-lg border ${getStatusColor(diagnostic.status)}`}
          >
            <div className="flex items-center gap-3 mb-2">
              {getStatusIcon(diagnostic.status)}
              <h3 className="font-semibold text-gray-900">{diagnostic.name}</h3>
            </div>
            <p className="text-sm text-gray-700 mb-1">{diagnostic.message}</p>
            {diagnostic.details && (
              <p className="text-xs text-gray-500">{diagnostic.details}</p>
            )}
          </div>
        ))}
      </div>

      <div className="flex justify-center">
        <button
          onClick={runDiagnostics}
          disabled={isRunning}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isRunning ? <Loader2 className="w-4 h-4 animate-spin" /> : <Settings className="w-4 h-4" />}
          {isRunning ? 'Running Diagnostics...' : 'Run Diagnostics Again'}
        </button>
      </div>
    </div>
  );
}