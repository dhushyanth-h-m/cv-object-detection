#!/usr/bin/env python3
"""
Troubleshooting script for the YOLO model
This script directly tests the YOLO model without going through the API
"""
import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import io

# Set up the Python path to include the current directory
sys.path.append(os.getcwd())

# Import the YOLO model
from app.models.yolo_model import YOLOModel

def test_model_directly(image_path, conf=0.25, classes=None):
    """Test the YOLO model directly with an image file"""
    print(f"Testing model with image: {image_path}")
    
    # Create the model
    model = YOLOModel()
    
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    # Open the image
    try:
        image = Image.open(image_path)
        print(f"Image opened successfully: {image.size}")
        
        # Run detection
        print(f"Running detection with conf={conf}, classes={classes}")
        results = model.detect(image, conf_threshold=conf, classes=classes)
        
        # Print results
        print(f"Detection results:")
        print(f"  Found {len(results['detections'])} objects")
        
        for i, det in enumerate(results['detections']):
            print(f"  Detection {i+1}:")
            print(f"    Class: {det['class_name']}")
            print(f"    Confidence: {det['confidence']:.4f}")
            print(f"    Bounding Box: ({int(det['bbox']['x1'])}, {int(det['bbox']['y1'])}) to "
                  f"({int(det['bbox']['x2'])}, {int(det['bbox']['y2'])})")
        
        # Print the output image path
        print(f"Result image saved to: {results['image_path']}")
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(description="Test the YOLO model directly")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0-1)")
    parser.add_argument("--classes", type=int, nargs="+", help="Class IDs to detect (0=person, 2=car, 5=bus, 7=truck)")
    
    args = parser.parse_args()
    test_model_directly(args.image, args.conf, args.classes)

if __name__ == "__main__":
    main() 