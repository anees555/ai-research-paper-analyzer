#!/usr/bin/env python3
"""
Simple test client for the FastAPI backend
"""

import requests
import json
import time
import os

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health/")
    print(f"Health Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200


def test_upload_and_process(pdf_path: str):
    """Test file upload and processing"""
    if not os.path.exists(pdf_path):
        print(f"❌ Test PDF not found: {pdf_path}")
        return False

    print(f"📄 Testing upload and processing with: {pdf_path}")

    # Step 1: Upload file
    print("\n1️⃣ Uploading PDF...")
    with open(pdf_path, "rb") as f:
        files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
        response = requests.post(f"{BASE_URL}/upload-pdf/", files=files)

    if response.status_code != 200:
        print(f"❌ Upload failed: {response.status_code}")
        print(response.text)
        return False

    upload_result = response.json()
    task_id = upload_result["task_id"]
    print(f"✅ Upload successful! Task ID: {task_id}")

    # Step 2: Start processing
    print("\n2️⃣ Starting processing...")
    response = requests.post(f"{BASE_URL}/process/{task_id}/")

    if response.status_code != 200:
        print(f"❌ Process start failed: {response.status_code}")
        print(response.text)
        return False

    print("✅ Processing started!")

    # Step 3: Poll for results
    print("\n3️⃣ Polling for results...")
    max_wait = 300  # 5 minutes
    start_time = time.time()

    while time.time() - start_time < max_wait:
        response = requests.get(f"{BASE_URL}/status/{task_id}/")

        if response.status_code != 200:
            print(f"❌ Status check failed: {response.status_code}")
            return False

        status_data = response.json()
        current_status = status_data["status"]

        print(f"📊 Status: {current_status}")

        if current_status == "completed":
            print("✅ Processing completed!")
            print("\n📋 RESULTS:")
            result = status_data["result"]
            print(f"Title: {result.get('title', 'N/A')}")
            print(f"Authors: {result.get('authors', [])}")
            print(f"Processing Method: {result.get('processing_method', 'N/A')}")
            print(f"Sections Found: {len(result.get('sections_found', []))}")
            print(f"Quick Summary: {result.get('quick_summary', 'N/A')[:100]}...")
            return True

        elif current_status == "failed":
            print(f"❌ Processing failed: {status_data.get('error', 'Unknown error')}")
            return False

        # Wait before next poll
        time.sleep(10)

    print("⏰ Timeout waiting for results")
    return False


def test_sync_processing(pdf_path: str):
    """Test synchronous processing endpoint"""
    if not os.path.exists(pdf_path):
        print(f"❌ Test PDF not found: {pdf_path}")
        return False

    print(f"🚀 Testing sync processing with: {pdf_path}")

    # Upload file
    with open(pdf_path, "rb") as f:
        files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
        response = requests.post(f"{BASE_URL}/upload-pdf/", files=files)

    if response.status_code != 200:
        print(f"❌ Upload failed: {response.status_code}")
        return False

    task_id = response.json()["task_id"]
    print(f"✅ Upload successful! Task ID: {task_id}")

    # Process synchronously
    print("⏳ Processing synchronously (this may take a while)...")
    start_time = time.time()

    response = requests.post(f"{BASE_URL}/process-sync/{task_id}/", timeout=300)

    processing_time = time.time() - start_time

    if response.status_code != 200:
        print(f"❌ Sync processing failed: {response.status_code}")
        print(response.text)
        return False

    result = response.json()["result"]
    print(f"✅ Sync processing completed in {processing_time:.1f}s!")
    print(f"Title: {result.get('title', 'N/A')}")
    print(f"Quick Summary: {result.get('quick_summary', 'N/A')[:100]}...")

    return True


def main():
    """Run all tests"""
    print("🧪 Testing FastAPI Backend for Research Paper Summarizer")
    print("=" * 60)

    # Test health first
    if not test_health():
        print("❌ Health check failed - is the server running?")
        print("Start server with: python backend/main.py")
        return

    # Find test PDF
    test_pdf_paths = [
        "papers/attention_is_all_you_need.pdf",
        "../papers/attention_is_all_you_need.pdf",
    ]

    test_pdf = None
    for path in test_pdf_paths:
        if os.path.exists(path):
            test_pdf = path
            break

    if not test_pdf:
        print("❌ No test PDF found. Please ensure a PDF exists in papers/ directory")
        return

    print(f"\n{'=' * 60}")
    print("🔄 Testing Async Processing")
    print(f"{'=' * 60}")
    test_upload_and_process(test_pdf)

    print(f"\n{'=' * 60}")
    print("⚡ Testing Sync Processing")
    print(f"{'=' * 60}")
    test_sync_processing(test_pdf)

    print(f"\n{'=' * 60}")
    print("🎉 Testing Complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
