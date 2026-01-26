from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.api.v1.routers import health, analysis, chat, auth, users
from app.core.database import engine, Base
import os

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

# Mount static files for serving figures
static_dir = os.path.join(os.path.dirname(__file__), "../data/figures")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static/figures", StaticFiles(directory=static_dir), name="figures")

# Initialize optimized models on startup
@app.on_event("startup")
async def startup_event():
    """Initialize models and services on application startup"""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("[STARTING] Starting AI Research Paper Analyzer API")
    
    # Initialize optimized model engine if AI is enabled
    if settings.ENABLE_AI_MODELS and settings.ENABLE_MODEL_PRELOADING:
        try:
            from app.services.optimized_model_loader import init_models
            init_models()
            logger.info("[SUCCESS] Model preloading initiated")
        except Exception as e:
            logger.warning(f"[WARNING] Model preloading failed: {e}")
    
    logger.info("[COMPLETE] API initialization complete")

@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on application shutdown"""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("[CLEANUP] Cleaning up resources...")
    
    # Clear model cache to free memory
    try:
        from app.services.optimized_model_loader import optimized_model_engine
        optimized_model_engine.clear_cache()
        logger.info("[SUCCESS] Model cache cleared")
    except Exception as e:
        logger.warning(f"[WARNING] Cache cleanup failed: {e}")
    
    logger.info("[SHUTDOWN] API shutdown complete")


@app.get("/")
async def root():
    return {
        "message": "Welcome to AI Research Paper Analyzer API",
        "docs": "/docs",
        "version": settings.VERSION,
    }
