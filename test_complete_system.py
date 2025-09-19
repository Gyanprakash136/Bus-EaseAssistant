#!/usr/bin/env python3
"""
Comprehensive Test Script for AI Bus Assistant
"""
import requests
import json
import asyncio
import websockets
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def print_test_header(test_name: str):
    print(f"\n{'='*60}")
    print(f"🧪 {test_name}")
    print(f"{'='*60}")

def test_server_health():
    """Test if server is running."""
    print_test_header("SERVER HEALTH CHECK")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print("✅ Server is running")
            print(f"📋 App: {data['message']}")
            print(f"🔢 Version: {data['version']}")
            print(f"🌍 Languages: {', '.join(data['supported_languages'])}")
            return True
        else:
            print(f"❌ Server responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server connection failed: {e}")
        return False

def test_system_status():
    """Test system status endpoint."""
    print_test_header("SYSTEM STATUS CHECK")
    
    try:
        response = requests.get(f"{BASE_URL}/status")
        if response.status_code == 200:
            data = response.json()
            print("✅ System status retrieved successfully")
            
            services = data['services']
            for service_name, service_data in services.items():
                print(f"🔧 {service_name.upper()}: ", end="")
                
                if service_name == 'nlu':
                    status = "🟢 Ready (Gemini)" if service_data.get('api_key_set') else "🟡 Mock mode"
                elif service_name == 'tts':
                    engines = [k for k, v in service_data.get('engines', {}).items() if v == 'available']
                    status = f"🟢 Ready ({', '.join(engines)})"
                elif service_name == 'stt':
                    status = "🟢 Vosk" if service_data.get('vosk_available') else "🟡 Mock"
                else:
                    status = "🟢 Ready"
                
                print(status)
            
            return True
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Status check error: {e}")
        return False

def test_voice_queries():
    """Test voice query API with multiple languages."""
    print_test_header("VOICE QUERY TESTS")
    
    test_queries = [
        {"text": "Where is bus 101A?", "language": "en", "description": "English query"},
        {"text": "बस 101A कहाँ है?", "language": "hi", "description": "Hindi query"},
        {"text": "ਬੱਸ 101A ਕਿੱਥੇ ਹੈ?", "language": "pa", "description": "Punjabi query"},
        {"text": "What is the status of bus 202B?", "language": "en", "description": "Status query"},
        {"text": "Bus 303C running?", "language": "en", "description": "Simple query"}
    ]
    
    success_count = 0
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {query['description']}")
        print(f"📝 Query: {query['text']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/v1/voice/query",
                json=query,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    print(f"✅ Success")
                    print(f"🤖 Response: {result['response_text'][:100]}...")
                    print(f"🌍 Language: {result['language']}")
                    success_count += 1
                else:
                    print(f"⚠️ API returned success=False")
                    print(f"❌ Error: {result.get('error', 'No error message')}")
            else:
                print(f"❌ HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Query failed: {e}")
        
        time.sleep(0.5)  # Rate limiting
    
    print(f"\n📊 Voice Query Results: {success_count}/{len(test_queries)} successful")
    return success_count > 0

def test_bus_data():
    """Test bus data API."""
    print_test_header("BUS DATA TESTS")
    
    tests = [
        {"params": {}, "description": "All buses"},
        {"params": {"bus_number": "101A"}, "description": "Specific bus (101A)"},
        {"params": {"bus_number": "999Z"}, "description": "Non-existent bus"}
    ]
    
    for test in tests:
        print(f"\n🚌 Test: {test['description']}")
        
        try:
            response = requests.get(f"{BASE_URL}/api/v1/bus/data", params=test['params'])
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    buses = data.get("buses", [])
                    print(f"✅ Success: Found {len(buses)} buses")
                    if buses:
                        bus = buses[0]
                        print(f"🚌 Sample: {bus.get('bus_number')} - {bus.get('bus_name')}")
                else:
                    print(f"⚠️ No buses found: {data.get('error', 'Unknown error')}")
            else:
                print(f"❌ HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Bus data test failed: {e}")

async def test_websocket():
    """Test WebSocket connection."""
    print_test_header("WEBSOCKET TEST")
    
    try:
        uri = f"ws://localhost:8000/api/v1/ws/voice/test_client"
        print(f"🔌 Connecting to {uri}")
        
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected")
            
            # Send a test message
            test_message = {
                "type": "text_query",
                "data": {"text": "Where is bus 101A?"},
                "language": "en"
            }
            
            await websocket.send(json.dumps(test_message))
            print(f"📤 Sent: {test_message['data']['text']}")
            
            # Wait for responses
            response_count = 0
            try:
                while response_count < 3:  # Expect multiple responses
                    response = await asyncio.wait_for(websocket.recv(), timeout=10)
                    response_data = json.loads(response)
                    response_count += 1
                    
                    msg_type = response_data.get("type")
                    print(f"📥 Received [{msg_type}]: ", end="")
                    
                    if msg_type == "connection":
                        print("Connection established")
                    elif msg_type == "processing":
                        print(response_data.get("message", "Processing..."))
                    elif msg_type == "response":
                        print(f"AI Response: {response_data.get('text', 'No text')[:50]}...")
                        break
                    elif msg_type == "error":
                        print(f"Error: {response_data.get('message')}")
                        break
                    else:
                        print(f"Unknown: {response_data}")
                        
            except asyncio.TimeoutError:
                print("⚠️ WebSocket response timeout")
            
            print("✅ WebSocket test completed")
            return True
            
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False

def test_language_detection():
    """Test language detection endpoint."""
    print_test_header("LANGUAGE DETECTION TEST")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/voice/languages")
        if response.status_code == 200:
            data = response.json()
            print("✅ Language detection available")
            print(f"🌍 Supported: {', '.join(data['supported_languages'])}")
            print(f"🏠 Default: {data['default_language']}")
            return True
        else:
            print(f"❌ Language detection failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Language detection error: {e}")
        return False

async def run_all_tests():
    """Run comprehensive test suite."""
    print("🚀 AI BUS ASSISTANT - COMPREHENSIVE TEST SUITE")
    print(f"📅 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test sequence
    tests = [
        ("Server Health", test_server_health),
        ("System Status", test_system_status),
        ("Voice Queries", test_voice_queries),
        ("Bus Data", test_bus_data),
        ("Language Detection", test_language_detection),
        ("WebSocket", test_websocket)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED! Your AI Bus Assistant is ready for use!")
    elif passed > len(results) // 2:
        print("⚠️ Most tests passed. Some features may be limited but core functionality works.")
    else:
        print("❌ Multiple test failures. Please check your configuration.")
    
    print("\n🚀 Next Steps:")
    print("1. Visit http://localhost:8000/docs for API documentation")
    print("2. Test the assistant with real queries")
    print("3. Integrate with your frontend application")

if __name__ == "__main__":
    asyncio.run(run_all_tests())
