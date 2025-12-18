#!/usr/bin/env python3
"""
Production-ready Gunicorn configuration for FastAPI backend
"""

import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
backlog = 2048

# Worker processes
workers = min(multiprocessing.cpu_count() * 2 + 1, 8)  # Cap at 8 workers
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 300  # 5 minutes for AI processing
keepalive = 2

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "research-paper-api"

# Server mechanics
preload_app = True  # Load app before forking workers (important for AI models)
max_requests = 1000
max_requests_jitter = 100

# SSL (uncomment for HTTPS)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"


def when_ready(server):
    """Called just after the server is started."""
    print("🚀 Research Paper Summarizer API is ready!")
    print(f"⚡ Running with {workers} workers")
    print(f"🔗 Available at: http://0.0.0.0:{os.getenv('PORT', '8000')}")


def worker_int(worker):
    """Called when a worker receives the INT or QUIT signal."""
    print(f"👷 Worker {worker.pid} received signal, shutting down gracefully...")
