#!/usr/bin/env python3
"""
Test Language Matching
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_language_matching():
    print("🧪 TESTING AUTOMATIC LANGUAGE MATCHING")
    print("="*50)
    
    test_cases = [
        # Hindi queries
        {"text": "बस 101A कहाँ है?", "expected_lang": "hi", "description": "Hindi Location Query"},
        {"text": "बस 202B की स्थिति क्या है?", "expected_lang": "hi", "description": "Hindi Status Query"},
        
        # English queries  
        {"text": "Where is bus 101A?", "expected_lang": "en", "description": "English Location Query"},
        {"text": "What is the status of bus 202B?", "expected_lang": "en", "description": "English Status Query"},
        
        # Punjabi queries
        {"text": "ਬੱਸ 101A ਕਿੱਥੇ ਹੈ?", "expected_lang": "pa", "description": "Punjabi Location Query"},
        {"text": "ਬੱਸ 202B ਦੀ ਸਥਿਤੀ ਕੀ ਹੈ?", "expected_lang": "pa", "description": "Punjabi Status Query"},
    ]
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test['description']}")
        print(f"📝 Query: {test['text']}")
        print(f"🎯 Expected Response Language: {test['expected_lang']}")
        
        try:
            response = requests.post(f"{BASE_URL}/api/v1/voice/query", json={
                "text": test["text"],
                "detect_language": True
            }, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("success"):
                    response_text = data["response_text"]
                    detected_lang = data["detected_language"]
                    
                    print(f"🌍 Detected Language: {detected_lang}")
                    print(f"🤖 Response: {response_text}")
                    
                    # Check if language matches
                    lang_match = detected_lang == test["expected_lang"]
                    status = "✅ PASS" if lang_match else "❌ FAIL"
                    
                    print(f"   {status} Language Match: {lang_match}")
                    
                    results.append({
                        "test": test["description"],
                        "expected": test["expected_lang"],
                        "detected": detected_lang,
                        "success": lang_match
                    })
                else:
                    print(f"   ❌ FAIL Query failed: {data.get('error')}")
                    results.append({"test": test["description"], "success": False})
            else:
                print(f"   ❌ FAIL HTTP {response.status_code}")
                results.append({"test": test["description"], "success": False})
                
        except Exception as e:
            print(f"   ❌ FAIL Error: {e}")
            results.append({"test": test["description"], "success": False})
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 LANGUAGE MATCHING TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for r in results if r.get("success"))
    total = len(results)
    
    for result in results:
        status = "✅" if result.get("success") else "❌"
        print(f"{status} {result['test']}")
        if "expected" in result and "detected" in result:
            print(f"    Expected: {result['expected']}, Got: {result['detected']}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 Perfect! Language matching is working correctly!")
    else:
        print("⚠️ Some language matching issues found. Check the logs.")

if __name__ == "__main__":
    test_language_matching()
