#!/usr/bin/env python3
"""
Start backend server with proper path setup
"""
import sys
import os
import uvicorn

# Set working directory to backend
backend_dir = r"C:\Users\Lenovo\Desktop\research_summary_project\backend"
os.chdir(backend_dir)

# Add backend directory to Python path
sys.path.insert(0, backend_dir)

print(f"🚀 Starting server from: {os.getcwd()}")
print(f"📁 Python path includes: {backend_dir}")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )