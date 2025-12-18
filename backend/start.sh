#!/bin/bash
# Production startup script for Research Paper Summarizer API

echo "🚀 Starting Research Paper Summarizer API"
echo "=========================================="

# Check if GROBID is running
echo "🔍 Checking GROBID server..."
if curl -s http://localhost:8070/api/isalive > /dev/null; then
    echo "✅ GROBID server is running"
else
    echo "❌ GROBID server not found. Starting GROBID..."
    docker run -d --name grobid -p 8070:8070 lfoppiano/grobid:0.8.0
    echo "⏳ Waiting for GROBID to start..."
    sleep 30
fi

# Install dependencies if needed
echo "📦 Checking dependencies..."
pip install -q -r requirements.txt

# Start the API server
echo "🌐 Starting FastAPI server..."
echo "📊 API will be available at: http://localhost:8000"
echo "📚 API docs: http://localhost:8000/docs"

# Use gunicorn for production, uvicorn for development
if [ "$ENV" = "production" ]; then
    echo "🏭 Running in production mode with Gunicorn..."
    gunicorn -c gunicorn_config.py main:app
else
    echo "🛠️ Running in development mode with Uvicorn..."
    python main.py
fi
