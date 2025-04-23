#!/usr/bin/env python3
"""
Diagnostic script for testing the object detection API
"""
import os
import sys
import requests
import json
from datetime import datetime

def test_detect_minimal(image_path, api_url="http://localhost:8000"):
    """Test the detect endpoint with minimal functionality"""
    print(f"Testing API with image: {image_path}")
    
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return False
    
    # Test the /test endpoint first
    try:
        print(f"Testing connection to {api_url}/test...")
        response = requests.get(f"{api_url}/test", timeout=5)
        if response.status_code != 200:
            print(f"Error: Test endpoint failed with status {response.status_code}")
            return False
        print(f"Test endpoint returned: {response.json()}")
    except Exception as e:
        print(f"Error connecting to test endpoint: {e}")
        return False
    
    # Send the image for detection with very minimal parameters
    try:
        print(f"Sending image for detection with minimal parameters...")
        
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            
            # Add some test classes
            data = {
                "classes": ["0", "2", "5", "7"]  # Test with person, car, bus, truck
            }
            
            print(f"POST {api_url}/detect")
            print(f"File: {image_path} (content-type: image/jpeg)")
            print(f"Classes: {data['classes']}")
            
            response = requests.post(f"{api_url}/detect", files=files, data=data, timeout=30)
            
            print(f"Response status: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            
            try:
                print(f"Response content: {response.text[:500]}...")
            except:
                print(f"Response content not displayable")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    print(f"Detection successful!")
                    print(f"Result keys: {list(result.keys())}")
                    return True
                except:
                    print(f"Error parsing JSON response")
                    return False
            else:
                print(f"Error: API returned status {response.status_code}")
                try:
                    error_json = response.json()
                    print(f"Error details: {json.dumps(error_json, indent=2)}")
                except:
                    print(f"Response: {response.text}")
                return False
    except Exception as e:
        import traceback
        print(f"Error during detection request: {e}")
        print(traceback.format_exc())
        return False

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python test_detect_diagnostic.py <image_path> [api_url]")
        return
    
    image_path = sys.argv[1]
    api_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    
    print(f"=== API Diagnostic Test Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    success = test_detect_minimal(image_path, api_url)
    print(f"=== Test {'Passed' if success else 'Failed'} ===")

if __name__ == "__main__":
    main() 