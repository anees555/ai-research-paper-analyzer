#!/usr/bin/env python3
"""
Debug Chat Service - Test chat functionality
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend" / "app"
sys.path.insert(0, str(backend_path))

semantic_path = Path(__file__).parent / "semantic-document-search" / "src"
sys.path.insert(0, str(semantic_path))

def test_chat_service():
    """Test chat service initialization and basic functionality"""
    
    print("=== Chat Service Debug Tool ===")
    print()
    
    # Check environment
    print("1. Environment Check:")
    print(f"   Backend path exists: {backend_path.exists()}")
    print(f"   Semantic path exists: {semantic_path.exists()}")
    print(f"   GROQ_API_KEY set: {'GROQ_API_KEY' in os.environ}")
    print(f"   .env file exists: {Path('.env').exists()}")
    print()
    
    # Test imports
    print("2. Import Test:")
    try:
        from services.chat_service import ChatService
        print("   ✓ ChatService imported successfully")
    except Exception as e:
        print(f"   ✗ ChatService import failed: {e}")
        return
    
    # Test initialization  
    print("3. Initialization Test:")
    try:
        chat_service = ChatService()
        print("   ✓ ChatService instance created")
    except Exception as e:
        print(f"   ✗ ChatService creation failed: {e}")
        return
        
    # Test lazy init
    try:
        chat_service._lazy_init()
        print("   ✓ ChatService initialized")
        print(f"   - Groq LLM available: {chat_service._groq_llm is not None}")
    except Exception as e:
        print(f"   ✗ ChatService initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✓ Chat service appears to be working correctly!")

if __name__ == "__main__":
    test_chat_service()