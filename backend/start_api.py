#!/usr/bin/env python3
"""
Simple API server startup script
"""

import uvicorn
import os
import sys

# Add the current directory to sys.path to make the app module importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("[STARTING] Starting AI Research Paper Analyzer API (Optimized Version)...")
    print("[INFO] Ready to process PDF uploads with fast processing modes!")
    print("[FEATURE] Performance Features:")
    print("   • Fast Mode: 30-60 seconds")
    print("   • Balanced Mode: 60-120 seconds") 
    print("   • Comprehensive Mode: 120-300 seconds")
    print("   • Models preloaded for instant response!")

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8003,
        reload=False,  # Disable reload to prevent model reloading
        log_level="info",
    )