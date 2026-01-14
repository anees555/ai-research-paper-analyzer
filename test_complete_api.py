#!/usr/bin/env python3
"""
Complete Test Script for Backend API
Tests all major endpoints and functionality
"""

import requests
import json
import time
from pathlib import Path

BASE_URL = "http://127.0.0.1:8002"

def test_health():
    """Test health endpoint"""
    print("1. Testing Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"   Error: {e}")
        return False

def test_upload_analysis():
    """Test file upload and analysis"""
    print("\n2. Testing File Upload & Analysis...")
    
    # Check if there are sample PDFs
    pdf_path = Path("../data/papers")
    sample_files = list(pdf_path.glob("*.pdf")) if pdf_path.exists() else []
    
    if not sample_files:
        print("   No sample PDFs found in data/papers/")
        return None
        
    sample_pdf = sample_files[0]
    print(f"   Using sample file: {sample_pdf.name}")
    
    try:
        with open(sample_pdf, 'rb') as f:
            files = {'file': (sample_pdf.name, f, 'application/pdf')}
            response = requests.post(f"{BASE_URL}/api/v1/analysis/upload", files=files)
        
        print(f"   Upload Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Job ID: {result['job_id']}")
            return result['job_id']
        else:
            print(f"   Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"   Error: {e}")
        return None

def test_job_status(job_id):
    """Test job status checking"""
    print(f"\n3. Testing Job Status for {job_id}...")
    
    for i in range(10):  # Poll for up to 10 times
        try:
            response = requests.get(f"{BASE_URL}/api/v1/analysis/status/{job_id}")
            if response.status_code == 200:
                result = response.json()
                status = result['status']
                print(f"   Attempt {i+1}: Status = {status}")
                
                if status == 'completed':
                    print("   ✅ Analysis completed successfully!")
                    return True
                elif status == 'failed':
                    print(f"   ❌ Analysis failed: {result.get('error', 'Unknown error')}")
                    return False
                else:
                    print("   ⏳ Still processing...")
                    time.sleep(5)
            else:
                print(f"   Error checking status: {response.status_code}")
                return False
        except Exception as e:
            print(f"   Error: {e}")
            return False
    
    print("   ⚠️ Analysis timed out")
    return False

def test_chat(job_id):
    """Test chat functionality"""
    print(f"\n4. Testing Chat with Job {job_id}...")
    
    test_questions = [
        "What is this paper about?",
        "What are the main contributions?",
        "What methodology was used?"
    ]
    
    for question in test_questions:
        print(f"   Asking: '{question}'")
        try:
            payload = {
                "job_id": job_id,
                "question": question
            }
            response = requests.post(f"{BASE_URL}/api/v1/chat/ask", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print(f"   Response: {result['message'][:100]}...")
                print(f"   Confidence: {result['confidence']}")
                print(f"   Sources: {len(result['sources'])}")
            else:
                print(f"   Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"   Error: {e}")

def main():
    print("=== Backend API Complete Test ===")
    print(f"Testing backend at: {BASE_URL}")
    
    # Test 1: Health check
    if not test_health():
        print("❌ Health check failed - backend may not be running")
        return
    
    # Test 2: Upload and analysis
    job_id = test_upload_analysis()
    if not job_id:
        print("❌ File upload failed")
        return
    
    # Test 3: Wait for analysis
    if not test_job_status(job_id):
        print("❌ Analysis failed or timed out")
        return
    
    # Test 4: Chat functionality
    test_chat(job_id)
    
    print("\n=== Test Complete ===")
    print("✅ All major functionality tested")
    print(f"🌐 API Documentation: {BASE_URL}/docs")

if __name__ == "__main__":
    main()