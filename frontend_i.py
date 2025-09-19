#!/usr/bin/env python3
"""
Quick Test for Fixed API
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def quick_test():
    print("⚡ QUICK TEST - FIXED API")
    print("="*40)
    
    # Test 1: Conversation Start
    print("\n🔍 Testing conversation start...")
    try:
        response = requests.post(f"{BASE_URL}/api/v1/conversation/start", 
                               json={"client_id": "test", "preferred_language": "en"})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Conversation start: {data.get('success')}")
            print(f"   Greeting: {data.get('greeting_text', '')[:50]}...")
        else:
            print(f"❌ Failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Voice Query  
    print("\n🔍 Testing voice query...")
    try:
        response = requests.post(f"{BASE_URL}/api/v1/voice/query",
                               json={"text": "Where is bus 101A?", "language": "en"})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Voice query: {data.get('success')}")
            print(f"   Response: {data.get('response_text', '')[:50]}...")
        else:
            print(f"❌ Failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n🎉 Quick test complete!")
    print("✅ Your API is ready for frontend integration!")

if __name__ == "__main__":
    quick_test()
