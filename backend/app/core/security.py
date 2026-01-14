from datetime import datetime, timedelta
from typing import Optional, Union
from jose import JWTError, jwt
import hashlib
import secrets
from app.core.config import settings

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    # Extract salt and hash from stored password
    if ':' not in hashed_password:
        return False
    
    salt, stored_hash = hashed_password.split(':', 1)
    # Hash the provided password with the same salt
    password_hash = hashlib.pbkdf2_hmac('sha256', 
                                       plain_password.encode('utf-8'), 
                                       salt.encode('utf-8'), 
                                       100000)
    return secrets.compare_digest(password_hash.hex(), stored_hash)

def get_password_hash(password: str) -> str:
    """Hash a password for storing"""
    # Generate a random salt
    salt = secrets.token_hex(16)
    # Hash the password
    password_hash = hashlib.pbkdf2_hmac('sha256', 
                                       password.encode('utf-8'), 
                                       salt.encode('utf-8'), 
                                       100000)
    return f"{salt}:{password_hash.hex()}"

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt
