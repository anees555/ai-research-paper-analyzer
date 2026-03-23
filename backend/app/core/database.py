import time
from sqlalchemy.exc import OperationalError

def retry_db_operation(max_retries=3, delay=2):
    """Decorator to retry DB operations on OperationalError."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except OperationalError as exc:
                    last_exc = exc
                    time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

DATABASE_URL = settings.DATABASE_URL

# Neon DB (Postgres) requires sslmode
# Enhanced connection pool settings for stability with long-running operations
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,           # Test connections before using them
    pool_recycle=300,             # Recycle connections every 5 minutes
    pool_size=3,                  # Keep 3 connections in the pool (Neon prefers small pools)
    max_overflow=5,               # Allow up to 5 additional connections
    connect_args={
        "connect_timeout": 10,
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
        "sslmode": "require"
    }
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper for robust commit/rollback
def safe_db_commit(db):
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise
