#!/usr/bin/env python3
"""
Test script for the Object Detection API
This demonstrates how to use the API programmatically from Python
"""
import os
import requests
import json
from pprint import pprint
import argparse
from PIL import Image, ImageDraw, ImageFont
import io

def detect_objects(image_path, confidence=0.25, classes=None, api_url="http://localhost:8000"):
    """
    Detect objects in an image using the API
    
    Args:
        image_path: Path to the image file
        confidence: Confidence threshold (0-1)
        classes: List of class IDs to detect (0=person, 2=car, 5=bus, 7=truck)
        api_url: Base URL of the API
    
    Returns:
        API response JSON
    """
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None
    
    # Prepare the request
    url = f"{api_url}/detect"
    files = {"file": open(image_path, "rb")}
    
    data = {"conf": confidence}
    if classes:
        data["classes"] = classes
    
    try:
        # Send the request
        print(f"Sending request to {url}...")
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the response
        result = response.json()
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None
    finally:
        files["file"].close()

def draw_detections(image_path, detections, output_path=None):
    """
    Draw detection results on an image
    
    Args:
        image_path: Path to the original image
        detections: List of detection results from the API
        output_path: Path to save the output image, if None uses 'output_{original_name}'
    
    Returns:
        Path to the output image
    """
    # Load the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Define colors for different classes
    colors = {
        0: (255, 0, 0),      # Person - Red
        2: (0, 255, 0),      # Car - Green
        5: (0, 0, 255),      # Bus - Blue
        7: (255, 255, 0)     # Truck - Yellow
    }
    
    # Draw bounding boxes
    for detection in detections:
        bbox = detection["bbox"]
        class_id = detection["class_id"]
        confidence = detection["confidence"]
        
        # Get the color for this class
        color = colors.get(class_id, (128, 128, 128))
        
        # Draw rectangle
        draw.rectangle(
            [(bbox["x1"], bbox["y1"]), (bbox["x2"], bbox["y2"])],
            outline=color,
            width=3
        )
        
        # Draw label
        label = f"{detection['class_name']} ({confidence:.2f})"
        draw.text((bbox["x1"], bbox["y1"] - 10), label, fill=color)
    
    # Save the image
    if output_path is None:
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = f"output_{name}{ext}"
    
    image.save(output_path)
    print(f"Output image saved to: {output_path}")
    return output_path

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test the Object Detection API")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0-1)")
    parser.add_argument("--classes", type=int, nargs="+", help="Class IDs to detect (0=person, 2=car, 5=bus, 7=truck)")
    parser.add_argument("--api", default="http://localhost:8000", help="API URL")
    parser.add_argument("--output", help="Output image path")
    args = parser.parse_args()
    
    # Detect objects
    result = detect_objects(
        args.image, 
        confidence=args.conf, 
        classes=args.classes,
        api_url=args.api
    )
    
    if result:
        print("Detection Results:")
        print(f"Inference Time: {result['inference_time']}")
        print(f"Objects Detected: {len(result['objects_detected'])}")
        
        # Print detection details
        for i, detection in enumerate(result["objects_detected"]):
            print(f"\nDetection {i+1}:")
            print(f"  Class: {detection['class_name']}")
            print(f"  Confidence: {detection['confidence']:.4f}")
            print(f"  Bounding Box: ({int(detection['bbox']['x1'])}, {int(detection['bbox']['y1'])}) to "
                  f"({int(detection['bbox']['x2'])}, {int(detection['bbox']['y2'])})")
        
        # Draw detections on the image
        if len(result["objects_detected"]) > 0:
            draw_detections(args.image, result["objects_detected"], args.output)

if __name__ == "__main__":
    main() 