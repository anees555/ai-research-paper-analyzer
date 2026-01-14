#!/usr/bin/env python3
"""
Manual Paper Indexing Test
Tests chat indexing for a specific job ID
"""

import requests
import json

def test_manual_indexing(job_id):
    """Test manual paper indexing for chat"""
    
    base_url = "http://127.0.0.1:8002"
    
    print(f"Testing manual indexing for job: {job_id}")
    
    try:
        # Try manual indexing
        response = requests.post(f"{base_url}/api/v1/chat/index/{job_id}")
        print(f"Indexing response: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Indexing successful: {result}")
            
            # Test a simple chat question
            test_question = "What is this paper about?"
            chat_request = {
                "job_id": job_id,
                "question": test_question
            }
            
            print(f"\nTesting chat with question: '{test_question}'")
            chat_response = requests.post(f"{base_url}/api/v1/chat/ask", json=chat_request)
            print(f"Chat response: {chat_response.status_code}")
            
            if chat_response.status_code == 200:
                chat_result = chat_response.json()
                print(f"✅ Chat working: {chat_result['message'][:200]}...")
                print(f"   Confidence: {chat_result['confidence']}")
                print(f"   Sources: {len(chat_result['sources'])}")
                return True
            else:
                print(f"❌ Chat failed: {chat_response.text}")
                return False
                
        else:
            print(f"❌ Indexing failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    # Use the job ID from the failed analysis
    job_id = "54010976-8128-4b56-8bce-5f218e00663c"
    success = test_manual_indexing(job_id)
    
    if success:
        print("\n🎉 Manual indexing and chat test successful!")
    else:
        print("\n❌ Manual indexing test failed")