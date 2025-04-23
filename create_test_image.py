#!/usr/bin/env python3
"""
Create a simple test image for object detection
"""
from PIL import Image, ImageDraw
import os

# Create directory if it doesn't exist
os.makedirs("test_images", exist_ok=True)

# Create a blank image with white background
width, height = 640, 480
image = Image.new('RGB', (width, height), color='white')
draw = ImageDraw.Draw(image)

# Draw a simple "person" silhouette (just a stick figure)
# Head
draw.ellipse([(width//2-30, 100), (width//2+30, 160)], outline='black', width=3)
# Body
draw.line([(width//2, 160), (width//2, 300)], fill='black', width=3)
# Arms
draw.line([(width//2, 200), (width//2-70, 220)], fill='black', width=3)
draw.line([(width//2, 200), (width//2+70, 220)], fill='black', width=3)
# Legs
draw.line([(width//2, 300), (width//2-50, 400)], fill='black', width=3)
draw.line([(width//2, 300), (width//2+50, 400)], fill='black', width=3)

# Draw a simple "car" shape
car_left = width//4 - 50
car_top = height - 120
car_width = 100
car_height = 50
# Car body
draw.rectangle([(car_left, car_top), (car_left + car_width, car_top + car_height)], outline='blue', width=3)
# Car windows
draw.rectangle([(car_left + 20, car_top + 10), (car_left + car_width - 20, car_top + 30)], outline='blue', width=1)
# Car wheels
wheel_radius = 10
draw.ellipse([(car_left + 15 - wheel_radius, car_top + car_height - wheel_radius), 
              (car_left + 15 + wheel_radius, car_top + car_height + wheel_radius)], fill='black')
draw.ellipse([(car_left + car_width - 15 - wheel_radius, car_top + car_height - wheel_radius), 
              (car_left + car_width - 15 + wheel_radius, car_top + car_height + wheel_radius)], fill='black')

# Save the image
filename = "test_images/test_image.jpg"
image.save(filename)
print(f"Created test image: {filename}")

# Display some info
print("This image contains simple drawings that may or may not be detected by the model.")
print("The YOLOv8 model is trained on real photos, so detection of simple drawings is not guaranteed.")
print("For best results, use real photographs containing people and vehicles.") 