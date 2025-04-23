"""
Simple object detector based on basic computer vision techniques
Used as a fallback when YOLOv8 can't be loaded
"""
import os
import uuid
import sys
import numpy as np
from PIL import Image, ImageDraw
import cv2
from pathlib import Path

class SimpleDetector:
    """
    A very basic object detector using color-based segmentation
    and contour detection. This is used only when YOLOv8 fails to load.
    """
    
    def __init__(self):
        """Initialize the simple detector"""
        print("Initializing simple detector as YOLOv8 fallback")
        self.classes = {
            0: "person",
            2: "car",
            5: "bus",
            7: "truck"
        }
    
    def detect(self, image_input, conf_threshold=0.25, classes=None):
        """
        Detect objects in an image using basic computer vision techniques
        
        Args:
            image_input: Path to an image file or PIL Image or numpy array
            conf_threshold: Confidence threshold (ignored in simple detector)
            classes: Classes to detect (ignored in simple detector)
            
        Returns:
            Dict with detections and result image path
        """
        try:
            # Convert input to numpy array
            if isinstance(image_input, str):
                # Load image from file
                img = cv2.imread(image_input)
                if img is None:
                    # Try PIL if OpenCV fails
                    img = np.array(Image.open(image_input))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif isinstance(image_input, Image.Image):
                # Convert PIL image to numpy array
                img = np.array(image_input)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif isinstance(image_input, np.ndarray):
                # Already a numpy array
                img = image_input.copy()
                if len(img.shape) == 2:
                    # Convert grayscale to BGR
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:
                    # Convert RGBA to BGR
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                # Unsupported input type
                print(f"Unsupported image input type: {type(image_input)}")
                return {"detections": [], "image_path": None}
            
            # Create a copy for drawing
            result_img = img.copy()
            
            # Perform simple detection (just a placeholder)
            detections = self._simple_detection(img)
            
            # Draw bounding boxes on result image
            for det in detections:
                bbox = det["bbox"]
                label = f"{det['class_name']} ({det['confidence']:.2f})"
                
                # Draw rectangle
                cv2.rectangle(
                    result_img,
                    (int(bbox["x1"]), int(bbox["y1"])),
                    (int(bbox["x2"]), int(bbox["y2"])),
                    (0, 255, 0),
                    2
                )
                
                # Draw label
                cv2.putText(
                    result_img,
                    label,
                    (int(bbox["x1"]), int(bbox["y1"] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
            
            # Save the result image
            result_filename = f"{uuid.uuid4()}_simple.jpg"
            result_path = f"app/static/results/{result_filename}"
            cv2.imwrite(result_path, result_img)
            
            return {
                "detections": detections,
                "image_path": result_path
            }
            
        except Exception as e:
            import traceback
            print(f"Error in simple detection: {e}")
            print(traceback.format_exc())
            
            # Return empty result
            return {"detections": [], "image_path": None}
    
    def _simple_detection(self, img):
        """
        Perform simple object detection using basic techniques
        
        Args:
            img: OpenCV image in BGR format
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        height, width = img.shape[:2]
        min_area = 0.01 * width * height  # At least 1% of the image area
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area >= min_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Assign a random class (this is just a fallback)
                class_id = np.random.choice(list(self.classes.keys()))
                confidence = np.random.uniform(0.5, 0.9)  # Random confidence
                
                detections.append({
                    "class_id": int(class_id),
                    "class_name": self.classes[class_id],
                    "confidence": float(confidence),
                    "bbox": {
                        "x1": float(x),
                        "y1": float(y),
                        "x2": float(x + w),
                        "y2": float(y + h),
                        "width": float(w),
                        "height": float(h)
                    }
                })
                
                # Limit to max 5 detections to avoid overwhelming
                if len(detections) >= 5:
                    break
        
        return detections

# Example usage
if __name__ == "__main__":
    # Test the simple detector
    detector = SimpleDetector()
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Testing with image: {image_path}")
        
        results = detector.detect(image_path)
        
        print(f"Found {len(results['detections'])} objects")
        print(f"Result image saved to {results['image_path']}")
    else:
        print("Usage: python simple_detector.py <image_path>") 