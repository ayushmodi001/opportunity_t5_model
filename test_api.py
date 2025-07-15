#!/usr/bin/env python3
"""
Test script for the MCQ Generation API
"""
import asyncio
import httpx
import json
import subprocess
import time
import sys

async def test_api():
    """Test the MCQ generation API"""
    
    # Test data
    test_payload = {
        "topics": ["Python programming"]
    }
    
    print("🧪 Testing MCQ Generation API")
    print("=" * 40)
    
    # Test the endpoint
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            print("📤 Sending request to /generate-mcqs/")
            print(f"Payload: {json.dumps(test_payload, indent=2)}")
            
            response = await client.post(
                "http://localhost:8000/generate-mcqs/",
                json=test_payload
            )
            
            print(f"\n📥 Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                questions = result.get("questions", [])
                
                print(f"✅ Success! Generated {len(questions)} MCQs")
                print("\n📝 Sample MCQs:")
                
                for i, mcq in enumerate(questions[:3]):  # Show first 3
                    print(f"\n--- MCQ {i+1} ---")
                    print(f"Topic: {mcq['topic']}")
                    print(f"Question: {mcq['question']}")
                    print(f"Correct Answer: {mcq['correct_answer']}")
                    print(f"Distractors: {mcq['distractors']}")
                
                # Save full results to file
                with open("test_results.json", "w") as f:
                    json.dump(result, f, indent=2)
                print(f"\n💾 Full results saved to test_results.json")
                
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                
    except httpx.ConnectError:
        print("❌ Connection error. Make sure the server is running on localhost:8000")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting FastAPI server...")
    try:
        # Start server in background
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "main:app", 
            "--host", "0.0.0.0", "--port", "8000", "--reload"
        ])
        
        print("⏳ Waiting for server to start...")
        time.sleep(5)  # Give server time to start
        
        return process
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return None

async def main():
    """Main test function"""
    server_process = None
    
    try:
        # Start server
        server_process = start_server()
        
        if server_process:
            # Test the API
            await test_api()
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    finally:
        # Clean up
        if server_process:
            print("\n🛑 Stopping server...")
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    asyncio.run(main())
