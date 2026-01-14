from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.routers import health, analysis, chat, auth, users
from app.core.database import engine, Base

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API for AI-powered research paper summarization",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware - must be added before other routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for debugging, or use settings.BACKEND_CORS_ORIGINS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["analysis"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])


@app.get("/")
async def root():
    return {
        "message": "Welcome to AI Research Paper Analyzer API",
        "docs": "/docs",
        "version": settings.VERSION,
    }
