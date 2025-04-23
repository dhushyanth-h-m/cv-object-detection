#!/usr/bin/env python3
"""
Simple test script for the Object Detection API
This script tests the HTTP API directly without requiring dependencies
"""
import os
import sys
import requests
import json
from datetime import datetime

def test_api(image_path, api_url="http://localhost:8000"):
    """Test the API with an image"""
    print(f"Testing API with image: {image_path}")
    
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return False
    
    # Make sure the API is running
    try:
        response = requests.get(f"{api_url}/", timeout=5)
        if response.status_code != 200:
            print(f"Error: API is not running at {api_url}")
            return False
        print(f"API is running at {api_url}")
    except Exception as e:
        print(f"Error connecting to API: {e}")
        print("Make sure the server is running with './start.sh'")
        return False
    
    # Send the image for detection
    try:
        print(f"Sending image for detection...")
        
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            data = {"conf": 0.25}
            
            # Add default test classes
            classes = [0, 2, 5, 7]  # person, car, bus, truck
            for class_id in classes:
                data.setdefault("classes", []).append(str(class_id))
            
            response = requests.post(f"{api_url}/detect", files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print(f"Detection successful!")
                print(f"Inference time: {result.get('inference_time', 'N/A')}")
                print(f"Objects detected: {len(result.get('objects_detected', []))}")
                
                for i, det in enumerate(result.get('objects_detected', [])):
                    print(f"  Object {i+1}: {det.get('class_name', 'unknown')} "
                          f"(Confidence: {det.get('confidence', 0):.2f})")
                
                # Save the URLs
                print(f"Result image: {api_url}{result.get('result_image_url', '')}")
                print(f"Original image: {api_url}{result.get('original_image_url', '')}")
                
                return True
            else:
                print(f"Error: API returned status {response.status_code}")
                try:
                    error_json = response.json()
                    print(f"Error details: {json.dumps(error_json, indent=2)}")
                except:
                    print(f"Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"Error during detection: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python test_simple.py <image_path> [api_url]")
        return
    
    image_path = sys.argv[1]
    api_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    
    print(f"=== API Test Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    success = test_api(image_path, api_url)
    print(f"=== Test {'Passed' if success else 'Failed'} ===")

if __name__ == "__main__":
    main() 