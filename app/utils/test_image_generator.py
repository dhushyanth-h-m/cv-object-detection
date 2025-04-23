import numpy as np
import cv2
import random
import os
import argparse
from pathlib import Path

def generate_test_image(width=640, height=480, num_shapes=5, output_path='test_image.jpg'):
    """
    Generate a test image with random shapes for testing object detection.
    
    Args:
        width (int): Width of the image
        height (int): Height of the image
        num_shapes (int): Number of shapes to draw
        output_path (str): Path to save the image
    
    Returns:
        str: Path to the saved image
    """
    # Create a blank image
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Define colors for shapes
    colors = [
        (255, 0, 0),    # Blue (BGR)
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 128, 128) # Gray
    ]
    
    # Add a background pattern
    for i in range(0, width, 40):
        cv2.line(img, (i, 0), (i, height), (240, 240, 240), 1)
    for i in range(0, height, 40):
        cv2.line(img, (0, i), (width, i), (240, 240, 240), 1)
    
    # Draw random shapes
    for _ in range(num_shapes):
        # Select random shape type: rectangle, circle, or triangle
        shape_type = random.choice(['rectangle', 'circle', 'triangle'])
        color = random.choice(colors)
        
        # Generate random position and size
        x = random.randint(50, width - 100)
        y = random.randint(50, height - 100)
        size = random.randint(30, min(width, height) // 4)
        
        if shape_type == 'rectangle':
            cv2.rectangle(img, (x, y), (x + size, y + size), color, -1)
            # Add border
            cv2.rectangle(img, (x, y), (x + size, y + size), (0, 0, 0), 2)
            
        elif shape_type == 'circle':
            radius = size // 2
            center = (x + radius, y + radius)
            cv2.circle(img, center, radius, color, -1)
            # Add border
            cv2.circle(img, center, radius, (0, 0, 0), 2)
            
        elif shape_type == 'triangle':
            points = np.array([
                [x, y + size],
                [x + size // 2, y],
                [x + size, y + size]
            ], np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(img, [points], color)
            # Add border
            cv2.polylines(img, [points], True, (0, 0, 0), 2)
    
    # Add a text label
    cv2.putText(img, 'Test Image', (width // 2 - 70, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Ensure the directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the image
    cv2.imwrite(output_path, img)
    print(f"Test image generated and saved to: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Generate a test image with random shapes')
    parser.add_argument('--width', type=int, default=640, help='Image width')
    parser.add_argument('--height', type=int, default=480, help='Image height')
    parser.add_argument('--shapes', type=int, default=5, help='Number of shapes to draw')
    parser.add_argument('--output', type=str, default='test_image.jpg', help='Output path')
    
    args = parser.parse_args()
    generate_test_image(args.width, args.height, args.shapes, args.output)

if __name__ == "__main__":
    main() 