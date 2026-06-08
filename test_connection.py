"""
Test script to verify backend and database connection
"""
import json
import requests
import time

# Test configuration
BACKEND_URL = "http://localhost:8000"
API_ENDPOINT = f"{BACKEND_URL}/api/chat"
HEALTH_ENDPOINT = f"{BACKEND_URL}/api/health"

def test_backend_health():
    """Test if backend is running"""
    print("\n1️⃣  Testing backend health...")
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to backend. Is it running on port 8000?")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_chat_api():
    """Test if chat API works"""
    print("\n2️⃣  Testing chat API...")
    payload = {
        "message": "¿Qué productos tienes?",
        "imagePath": None
    }
    
    try:
        print(f"   Sending: {json.dumps(payload, indent=2)}")
        response = requests.post(API_ENDPOINT, json=payload, timeout=30)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except requests.exceptions.Timeout:
        print("   ⏱️  Request timed out (30s). Backend might be slow or stuck.")
        return False
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to backend")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_cors():
    """Test CORS headers"""
    print("\n3️⃣  Testing CORS...")
    try:
        headers = {
            "Origin": "http://localhost:3000",
            "Content-Type": "application/json"
        }
        response = requests.options(API_ENDPOINT, headers=headers, timeout=5)
        print(f"   Status: {response.status_code}")
        
        cors_headers = {k: v for k, v in response.headers.items() 
                       if 'access' in k.lower() or 'cors' in k.lower()}
        if cors_headers:
            print(f"   CORS Headers: {cors_headers}")
        else:
            print("   ⚠️  No CORS headers found")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("🧪 Backend Connection Test")
    print("=" * 60)
    
    results = []
    
    # Test 1: Health
    results.append(("Backend Health", test_backend_health()))
    
    # Only continue if backend is up
    if results[0][1]:
        results.append(("Chat API", test_chat_api()))
        results.append(("CORS", test_cors()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    if all(result[1] for result in results):
        print("\n🎉 All tests passed! Backend is connected.")
    else:
        print("\n⚠️  Some tests failed. Check the details above.")

if __name__ == "__main__":
    main()
