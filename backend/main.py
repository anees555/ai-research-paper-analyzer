import uvicorn
import os
import sys

# Add the current directory to sys.path to make the app module importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("[STARTING] Starting AI Research Paper Summarizer API (Production Mode)...")
    print("[INFO] Ready to process PDF uploads!")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info",
    )
