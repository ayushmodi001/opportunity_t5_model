#!/usr/bin/env python3
"""
Test server startup and basic functionality
"""
import asyncio
import json
import requests
import time
import subprocess
import sys

def test_server_startup():
    """Test if the server can start without import errors"""
    print("🧪 Testing Server Startup")
    print("=" * 30)
    
    try:
        # Test imports first
        print("1️⃣ Testing imports...")
        import main
        print("✅ All imports successful")
        
        # Start server process
        print("2️⃣ Starting server...")
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "main:app", 
            "--host", "127.0.0.1", "--port", "8000"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        print("⏳ Waiting for server...")
        time.sleep(5)
        
        # Test if server is responding
        try:
            response = requests.get("http://127.0.0.1:8000/docs", timeout=3)
            if response.status_code == 200:
                print("✅ Server started successfully!")
                print("📄 API documentation available at: http://127.0.0.1:8000/docs")
                
                # Test the MCQ endpoint
                print("3️⃣ Testing MCQ endpoint...")
                test_data = {"topics": ["Python programming"]}
                
                mcq_response = requests.post(
                    "http://127.0.0.1:8000/generate-mcqs/",
                    json=test_data,
                    timeout=30
                )
                
                if mcq_response.status_code == 200:
                    result = mcq_response.json()
                    questions = result.get("questions", [])
                    print(f"✅ MCQ endpoint working! Generated {len(questions)} questions")
                    
                    if questions:
                        sample = questions[0]
                        print(f"📝 Sample question: {sample['question']}")
                        print(f"📝 Answer: {sample['correct_answer']}")
                        
                else:
                    print(f"⚠️ MCQ endpoint returned: {mcq_response.status_code}")
                    print(f"Response: {mcq_response.text[:200]}...")
                
                return True
            else:
                print(f"❌ Server not responding: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to server: {e}")
            return False
        finally:
            # Stop the server
            print("🛑 Stopping server...")
            process.terminate()
            process.wait()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_server_startup()
    if success:
        print("\n🎉 Server test completed successfully!")
        print("\nTo start the server manually, run:")
        print("python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000")
    else:
        print("\n❌ Server test failed")
