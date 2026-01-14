from typing import List, Union
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, validator


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Research Paper Analyzer"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # CORS - using plain strings for compatibility
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8000",
    ]

    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost/dbname"

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

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
