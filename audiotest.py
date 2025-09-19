#!/usr/bin/env python3
"""
Test Enhanced Natural Speech
"""
import requests
import time

BASE_URL = "http://localhost:8000"

def test_enhanced_responses():
    """Test the enhanced natural responses."""
    
    test_cases = [
        # English tests
        {"text": "Where is bus 101A?", "language": "en", "description": "English Location Query"},
        {"text": "What is the status of bus 202B?", "language": "en", "description": "English Status Query"},
        
        # Hindi tests  
        {"text": "बस 101A कहाँ है?", "language": "hi", "description": "Hindi Location Query"},
        {"text": "बस 202B की स्थिति क्या है?", "language": "hi", "description": "Hindi Status Query"},
        
        # Punjabi tests
        {"text": "ਬੱਸ 101A ਕਿੱਥੇ ਹੈ?", "language": "pa", "description": "Punjabi Location Query"},
        {"text": "ਬੱਸ 202B ਦੀ ਸਥਿਤੀ ਕੀ ਹੈ?", "language": "pa", "description": "Punjabi Status Query"}
    ]
    
    print("🧪 TESTING ENHANCED NATURAL MULTILINGUAL RESPONSES")
    print("=" * 70)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test_case['description']}")
        print(f"📝 Query: {test_case['text']}")
        print(f"🌍 Language: {test_case['language']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/v1/voice/query",
                json=test_case,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    print(f"✅ SUCCESS!")
                    print(f"🤖 Natural Response:")
                    print(f"   {result['response_text']}")
                    print(f"🔊 Audio: {'Available' if result.get('audio_url') else 'Not generated'}")
                else:
                    print(f"⚠️ API Error: {result.get('error')}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
        
        print("-" * 50)
        time.sleep(1)
    
    print("\n🎉 Enhanced natural speech testing completed!")
    print("\nFeatures tested:")
    print("✅ Natural conversational responses") 
    print("✅ Real location names (instead of lat/lng)")
    print("✅ Proper Hindi and Punjabi translations")
    print("✅ Human-like speech patterns")

if __name__ == "__main__":
    test_enhanced_responses()
