#!/usr/bin/env python3
"""
Test Mistral API integration
"""
import asyncio

async def test_mistral_api():
    print("🧪 Testing Mistral API Integration")
    print("=" * 40)
    
    try:
        from context_fetcher import fetch_context
        from distractor_generator import generate_distractors
        
        # Test context fetching
        print("1️⃣ Testing context fetching...")
        context = await fetch_context("Python programming")
        
        if context and len(context) > 50:
            print(f"✅ Context fetched successfully ({len(context)} characters)")
            print(f"Sample: {context[:200]}...")
            
            # Test distractor generation
            print("\n2️⃣ Testing distractor generation...")
            distractors = await generate_distractors(context, "Python")
            
            if distractors and len(distractors) == 3:
                print(f"✅ Distractors generated successfully: {distractors}")
            else:
                print(f"⚠️ Distractors generated but may be fallback: {distractors}")
        else:
            print("⚠️ Context fetching used fallback (API may not be available)")
            
    except Exception as e:
        print(f"❌ Error testing Mistral API: {e}")
        
    print("\n🎉 Mistral API test completed!")

if __name__ == "__main__":
    asyncio.run(test_mistral_api())
