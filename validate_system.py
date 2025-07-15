#!/usr/bin/env python3
"""
Final validation of the MCQ system
"""

def validate_system():
    print("🔍 Final System Validation")
    print("=" * 35)
    
    # Test 1: Import validation
    print("1️⃣ Testing imports...")
    try:
        import main
        import config
        import context_fetcher
        import answer_extractor
        import question_generator
        import distractor_generator
        import summarizer
        print("✅ All modules import successfully")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test 2: Configuration validation
    print("\n2️⃣ Testing configuration...")
    try:
        from config import MISTRAL_API_KEY, MAX_QUESTIONS_PER_TOPIC
        print(f"✅ Config loaded - Target questions: {MAX_QUESTIONS_PER_TOPIC}")
        if MISTRAL_API_KEY:
            print("✅ Mistral API key found")
        else:
            print("⚠️ Mistral API key not found (will use fallbacks)")
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False
    
    # Test 3: FastAPI app validation
    print("\n3️⃣ Testing FastAPI app...")
    try:
        from main import app
        print("✅ FastAPI app created successfully")
        print("✅ Ready to start with: python -m uvicorn main:app --reload")
    except Exception as e:
        print(f"❌ FastAPI error: {e}")
        return False
    
    print("\n🎉 System validation completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Start server: python -m uvicorn main:app --reload")
    print("2. Visit API docs: http://127.0.0.1:8000/docs")
    print("3. Test endpoint: POST /generate-mcqs/")
    print("   Payload: {\"topics\": [\"Python programming\"]}")
    
    return True

if __name__ == "__main__":
    validate_system()
