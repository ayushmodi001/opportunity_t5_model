#!/usr/bin/env python3
"""
Debug Mistral API structure
"""
import os
from dotenv import load_dotenv

load_dotenv()

def test_mistral_api_structure():
    print("🔍 Debugging Mistral API Structure")
    print("=" * 40)
    
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("❌ No API key found")
        return
    
    try:
        from mistralai import Mistral
        
        print("✅ Mistral library imported successfully")
        
        # Initialize client
        client = Mistral(api_key=api_key)
        print("✅ Client initialized")
        
        # Check client structure
        print(f"Client type: {type(client)}")
        print(f"Client attributes: {dir(client)}")
        
        # Check chat attribute
        if hasattr(client, 'chat'):
            print(f"Chat type: {type(client.chat)}")
            print(f"Chat attributes: {dir(client.chat)}")
            
            # Try the API call
            try:
                response = client.chat.complete(
                    model="mistral-small-latest",
                    messages=[{"role": "user", "content": "Hello"}]
                )
                print("✅ API call successful!")
                print(f"Response type: {type(response)}")
                print(f"Response: {response}")
                
            except Exception as e:
                print(f"❌ API call failed: {e}")
                
                # Try alternative method
                try:
                    response = client.chat(
                        model="mistral-small-latest",
                        messages=[{"role": "user", "content": "Hello"}]
                    )
                    print("✅ Alternative API call successful!")
                    print(f"Response: {response}")
                except Exception as e2:
                    print(f"❌ Alternative API call also failed: {e2}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_mistral_api_structure()
