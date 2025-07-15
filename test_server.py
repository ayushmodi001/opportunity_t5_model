#!/usr/bin/env python3
"""
Test the MCQ API server
"""
import asyncio
import json
import subprocess
import time
import sys
import requests
from pathlib import Path

def start_server():
    """Start the FastAPI server in background"""
    print("🚀 Starting FastAPI server...")
    
    try:
        # Start server process
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "main:app", 
            "--host", "localhost", "--port", "8000"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        time.sleep(8)
        
        # Check if server is responding
        try:
            response = requests.get("http://localhost:8000/docs", timeout=5)
            if response.status_code == 200:
                print("✅ Server started successfully!")
                return process
            else:
                print(f"❌ Server not responding properly: {response.status_code}")
                process.terminate()
                return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to server: {e}")
            process.terminate()
            return None
            
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return None

def test_api():
    """Test the MCQ generation API"""
    print("\n📤 Testing MCQ API...")
    
    test_data = {
        "topics": ["Python programming"]
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/generate-mcqs/",
            json=test_data,
            timeout=60
        )
        
        print(f"📥 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            questions = result.get("questions", [])
            
            print(f"✅ Success! Generated {len(questions)} MCQs")
            
            # Display first MCQ
            if questions:
                mcq = questions[0]
                print(f"\n📝 Sample MCQ:")
                print(f"Topic: {mcq['topic']}")
                print(f"Question: {mcq['question']}")
                print(f"Correct Answer: {mcq['correct_answer']}")
                print(f"Distractors: {', '.join(mcq['distractors'])}")
            
            # Save results
            with open("api_test_results.json", "w") as f:
                json.dump(result, f, indent=2)
            print("💾 Full results saved to api_test_results.json")
            
            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timeout - server may be processing")
        return False
    except Exception as e:
        print(f"❌ Request error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing MCQ API Server")
    print("=" * 40)
    
    server_process = None
    
    try:
        # Start server
        server_process = start_server()
        
        if server_process:
            # Test API
            success = test_api()
            
            if success:
                print("\n🎉 API test completed successfully!")
            else:
                print("\n❌ API test failed")
        else:
            print("\n❌ Could not start server")
            
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted")
    finally:
        if server_process:
            print("\n🛑 Stopping server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()

if __name__ == "__main__":
    main()
