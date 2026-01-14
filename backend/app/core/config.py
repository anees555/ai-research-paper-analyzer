from typing import List, Union, Optional
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, validator
import os
from dotenv import load_dotenv

# Manually load .env file to ensure it's loaded
env_path = os.path.join(os.path.dirname(__file__), "../../../.env")
load_dotenv(env_path)
print(f"🔧 Loading .env from: {env_path}")
print(f"🔧 .env exists: {os.path.exists(env_path)}")
print(f"🔧 GROQ_API_KEY loaded: {bool(os.getenv('GROQ_API_KEY'))}")


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Research Paper Analyzer"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = "postgresql://localhost/research_db"

    # CORS - using plain strings for compatibility
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8000",
    ]

    # File Uploads
    UPLOAD_DIR: str = "data/uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10 MB

    # Model Settings (Can be overridden by env vars)
    ENABLE_AI_MODELS: bool = True
    MODEL_DEVICE: int = -1  # -1 for CPU, 0 for GPU

    # Security
    SECRET_KEY: str = "your-super-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Groq LLM Configuration
    GROQ_API_KEY: Optional[str] = None
    DEFAULT_MODEL: str = "llama-3.1-8b-instant"
    MAX_TOKENS: int = 2048
    TEMPERATURE: float = 0.3
    
    # File Processing
    MAX_FILE_SIZE_MB: int = 50
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    
    # Vector Store
    PERSIST_DIRECTORY: str = "./data/chroma_db"
    COLLECTION_NAME: str = "research_papers"
    
    # GROBID
    GROBID_URL: str = "http://localhost:8070"
    
    # Additional settings
    MAX_CONTEXT_CHUNKS: int = 10

    class Config:
        case_sensitive = True
        env_file = os.path.join(os.path.dirname(__file__), "../../../.env")  # Points to root .env file


settings = Settings()
