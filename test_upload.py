#!/usr/bin/env python3
"""
Test script to verify full upload and processing workflow
"""

import requests
import time


def test_upload_and_process():
    base_url = "http://localhost:8000"

    # Step 1: Upload PDF
    print("📤 Step 1: Uploading PDF...")
    upload_url = f"{base_url}/upload-pdf/"

    with open("papers/attention_is_all_you_need.pdf", "rb") as f:
        files = {"file": ("attention_is_all_you_need.pdf", f, "application/pdf")}
        response = requests.post(upload_url, files=files)

    if response.status_code != 200:
        print(f"❌ Upload failed: {response.status_code}")
        print(response.text)
        return None

    upload_result = response.json()
    task_id = upload_result["task_id"]
    print(f"✅ Upload successful! Task ID: {task_id}")

    # Step 2: Start Processing
    print("\n🔄 Step 2: Starting processing...")
    process_url = f"{base_url}/process/{task_id}/"
    response = requests.post(process_url)

    if response.status_code != 200:
        print(f"❌ Processing start failed: {response.status_code}")
        print(response.text)
        return None

    process_result = response.json()
    print(f"✅ Processing started: {process_result['message']}")

    # Step 3: Monitor Status
    print("\n👀 Step 3: Monitoring status...")
    status_url = f"{base_url}/status/{task_id}/"

    while True:
        response = requests.get(status_url)
        if response.status_code != 200:
            print(f"❌ Status check failed: {response.status_code}")
            break

        status_result = response.json()
        status = status_result["status"]

        print(f"Status: {status}")

        if status == "completed":
            print("🎉 Processing completed!")
            print("\n📊 Results:")
            result = status_result.get("result", {})

            # Print summary of results
            if "summaries" in result:
                summaries = result["summaries"]
                print(f"✅ Generated {len(summaries)} summaries:")
                for summary_type, content in summaries.items():
                    print(f"  - {summary_type}: {len(content)} characters")

            if "structured_data" in result:
                structured = result["structured_data"]
                print(f"✅ Extracted structured data:")
                print(f"  - Title: {structured.get('title', 'N/A')[:50]}...")
                print(f"  - Authors: {len(structured.get('authors', []))} authors")
                print(
                    f"  - References: {len(structured.get('references', []))} references"
                )

            return status_result

        elif status == "failed":
            print("❌ Processing failed!")
            print(f"Error: {status_result.get('error', 'Unknown error')}")
            return None

        elif status == "processing":
            print("⏳ Still processing... (waiting 5 seconds)")
            time.sleep(5)

        else:
            print(f"Unknown status: {status}")
            break

    return None


def test_sync_processing():
    """Test synchronous processing (faster for testing)"""
    base_url = "http://localhost:8000"

    # Step 1: Upload PDF
    print("📤 Uploading PDF...")
    upload_url = f"{base_url}/upload-pdf/"

    with open("papers/attention_is_all_you_need.pdf", "rb") as f:
        files = {"file": ("attention_is_all_you_need.pdf", f, "application/pdf")}
        response = requests.post(upload_url, files=files)

    if response.status_code != 200:
        print(f"❌ Upload failed: {response.status_code}")
        return None

    upload_result = response.json()
    task_id = upload_result["task_id"]
    print(f"✅ Upload successful! Task ID: {task_id}")

    # Step 2: Process Synchronously
    print("\n🔄 Processing synchronously (this may take a while)...")
    process_url = f"{base_url}/process-sync/{task_id}/"
    response = requests.post(process_url)

    if response.status_code != 200:
        print(f"❌ Processing failed: {response.status_code}")
        print(response.text)
        return None

    result = response.json()
    print("🎉 Processing completed!")

    # Print results summary
    if "result" in result:
        process_result = result["result"]

        if "summaries" in process_result:
            summaries = process_result["summaries"]
            print(f"\n📊 Generated {len(summaries)} summaries:")
            for summary_type, content in summaries.items():
                print(f"  - {summary_type}: {len(content)} characters")

        if "structured_data" in process_result:
            structured = process_result["structured_data"]
            print(f"\n📋 Extracted structured data:")
            print(f"  - Title: {structured.get('title', 'N/A')[:50]}...")
            print(f"  - Authors: {len(structured.get('authors', []))} authors")
            print(f"  - References: {len(structured.get('references', []))} references")

    return result


if __name__ == "__main__":
    print("🚀 AI Research Paper Analyzer - Full Workflow Test")
    print("=" * 60)

    choice = input(
        "\nChoose test mode:\n1. Background processing (recommended)\n2. Synchronous processing (may timeout)\nEnter choice (1/2): "
    )

    if choice == "2":
        test_sync_processing()
    else:
        test_upload_and_process()
