from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

DATABASE_URL = settings.DATABASE_URL

# Neon DB (Postgres) requires sslmode
# Enhanced connection pool settings for stability with long-running operations
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,           # Test connections before using them
    pool_recycle=3600,            # Recycle connections every 1 hour
    pool_size=10,                 # Keep 10 connections in the pool
    max_overflow=20,              # Allow up to 20 additional connections
    connect_args={"connect_timeout": 10}  # 10s connection timeout
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
