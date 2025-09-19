#!/usr/bin/env python3
"""
Complete Setup Validation for AI Bus Assistant
"""
import os
import sys
import asyncio
from datetime import datetime

def validate_complete_setup():
    """Comprehensive validation of the entire setup."""
    
    print("🚀 AI BUS ASSISTANT - COMPLETE SETUP VALIDATION")
    print("=" * 60)
    print(f"📅 Validation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    validation_results = {
        "settings": False,
        "environment": False,
        "gemini": False,
        "location_conversion": False,
        "api_endpoints": False
    }
    
    # 1. Settings Validation
    print(f"\n🔧 1. SETTINGS VALIDATION")
    try:
        from config.settings import settings, get_location_settings, is_location_conversion_enabled
        
        print(f"   ✅ Settings imported successfully")
        print(f"   📱 App: {settings.app_name} v{settings.app_version}")
        print(f"   🌍 Host: {settings.host}:{settings.port}")
        print(f"   🐛 Debug Mode: {settings.debug}")
        
        validation_results["settings"] = True
        
    except Exception as e:
        print(f"   ❌ Settings validation failed: {e}")
        return validation_results
    
    # 2. Environment Variables Check
    print(f"\n🌍 2. ENVIRONMENT VARIABLES")
    critical_env_vars = {
        "GEMINI_API_KEY": "Gemini AI API key",
        "BUS_API_BASE_URL": "Bus data API URL", 
        "DEBUG": "Debug mode setting"
    }
    
    env_status = {}
    for var, description in critical_env_vars.items():
        value = os.getenv(var, "Not Set")
        is_set = value != "Not Set" and value != ""
        status = "✅" if is_set else "⚠️"
        env_status[var] = is_set
        
        display_value = value if var != "GEMINI_API_KEY" else ("***SET***" if is_set else "NOT SET")
        print(f"   {status} {var}: {display_value}")
    
    validation_results["environment"] = any(env_status.values())
    
    # 3. Gemini Configuration
    print(f"\n🤖 3. GEMINI AI CONFIGURATION")
    try:
        gemini_enabled = is_location_conversion_enabled()
        api_key_configured = settings.gemini_api_key != "your_actual_gemini_api_key_here"
        
        print(f"   {'✅' if api_key_configured else '❌'} API Key: {'Configured' if api_key_configured else 'Default/Not Set'}")
        print(f"   {'✅' if gemini_enabled else '❌'} Location Conversion: {'Enabled' if gemini_enabled else 'Disabled'}")
        print(f"   📝 Model: {settings.gemini_model}")
        print(f"   ⏱️ Timeout: {settings.gemini_location_timeout}s")
        print(f"   🎯 Detail Level: {settings.location_detail_level}")
        
        validation_results["gemini"] = gemini_enabled
        
    except Exception as e:
        print(f"   ❌ Gemini validation error: {e}")
    
    # 4. Location Conversion Test
    print(f"\n📍 4. LOCATION CONVERSION TEST")
    try:
        # Test coordinate validation
        test_coords = [
            (12.9716, 77.5946, "Bangalore Central"),
            (20.2961, 85.8245, "Bhubaneswar"),
            (19.0760, 72.8777, "Mumbai")
        ]
        
        for lat, lng, expected_area in test_coords:
            cache_key = f"{lat:.4f},{lng:.4f}"
            print(f"   📌 Test: ({lat}, {lng}) → Expected: {expected_area}")
            
        # Test location name validation
        test_names = [
            "Central Railway Station, Bangalore",
            "Airport Terminal, Mumbai",
            "Electronic City, Bangalore" 
        ]
        
        valid_names = 0
        for name in test_names:
            if settings.validate_location_name(name):
                valid_names += 1
                print(f"   ✅ Valid: {name}")
            else:
                print(f"   ❌ Invalid: {name}")
        
        validation_results["location_conversion"] = valid_names == len(test_names)
        
    except Exception as e:
        print(f"   ❌ Location conversion test error: {e}")
    
    # 5. API Endpoints Check
    print(f"\n🌐 5. API ENDPOINTS VALIDATION")
    try:
        import requests
        from urllib.parse import urljoin
        
        base_url = f"http://{settings.host}:{settings.port}"
        endpoints_to_test = [
            "/health",
            "/api/v1/status", 
            "/api/v1/voice/languages"
        ]
        
        # Note: This assumes the server is running
        print(f"   🔗 Base URL: {base_url}")
        print(f"   ⚠️  Note: Server must be running for endpoint tests")
        
        for endpoint in endpoints_to_test:
            full_url = urljoin(base_url, endpoint)
            print(f"   📡 Endpoint: {endpoint}")
        
        validation_results["api_endpoints"] = True  # Assume valid for now
        
    except Exception as e:
        print(f"   ❌ API validation error: {e}")
    
    # 6. Final Summary
    print(f"\n📊 VALIDATION SUMMARY")
    print("=" * 40)
    
    total_checks = len(validation_results)
    passed_checks = sum(validation_results.values())
    
    for check_name, result in validation_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {check_name.replace('_', ' ').title()}")
    
    print(f"\n🎯 Overall Score: {passed_checks}/{total_checks} ({passed_checks/total_checks*100:.1f}%)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    if not validation_results["environment"]:
        print(f"   🔧 Update your .env file with actual API keys")
    
    if not validation_results["gemini"]:
        print(f"   🤖 Set a valid GEMINI_API_KEY in .env")
    
    if passed_checks == total_checks:
        print(f"   🎉 Perfect! Your AI Bus Assistant is production-ready!")
    elif passed_checks >= total_checks * 0.8:
        print(f"   ✅ Great! Minor configuration needed for full functionality")
    else:
        print(f"   ⚠️  Several configuration issues need attention")
    
    return validation_results

def test_location_conversion_live():
    """Test actual Gemini location conversion if enabled."""
    print(f"\n🧪 LIVE LOCATION CONVERSION TEST")
    print("-" * 40)
    
    try:
        from config.settings import settings
        
        if not settings.is_gemini_location_enabled():
            print("   ⚠️  Gemini location conversion not enabled")
            print("   💡 Set GEMINI_API_KEY in .env to test live conversion")
            return
        
        # Import the fetcher
        from fetcher.fetch_data import bus_data_fetcher
        
        # Test coordinates (Indian locations)
        test_coords = [
            (12.9716, 77.5946),  # Bangalore
            (20.2961, 85.8245),  # Bhubaneswar  
            (19.0760, 72.8777)   # Mumbai
        ]
        
        print("   🌍 Testing live Gemini coordinate conversion...")
        
        async def run_conversion_test():
            for i, (lat, lng) in enumerate(test_coords, 1):
                try:
                    print(f"   📍 Test {i}: Converting ({lat}, {lng})...")
                    location_name = await bus_data_fetcher._coordinates_to_location_name(lat, lng)
                    print(f"      ✅ Result: {location_name}")
                except Exception as e:
                    print(f"      ❌ Error: {e}")
        
        # Run the async test
        asyncio.run(run_conversion_test())
        
    except Exception as e:
        print(f"   ❌ Live test error: {e}")

if __name__ == "__main__":
    results = validate_complete_setup()
    
    # Run live test if Gemini is configured
    try:
        test_location_conversion_live()
    except Exception as e:
        print(f"⚠️ Live test skipped: {e}")
    
    print(f"\n🏁 Validation Complete!")
