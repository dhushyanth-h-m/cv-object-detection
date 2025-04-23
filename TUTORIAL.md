# Object Detection API Tutorial

This tutorial walks you through using the Object Detection API for identifying pedestrians and vehicles in images.

## Getting Started

### 1. Running the API Server

Start the API server using the provided script:

```bash
./start.sh
```

This script will:
1. Create a Python virtual environment if one doesn't exist
2. Install all dependencies
3. Start the FastAPI server

Alternatively, you can use Docker Compose:

```bash
docker-compose up -d
```

### 2. Accessing the Web Interface

Once the server is running, you can access the web interface by opening your browser and navigating to:

```
http://localhost:8000
```

The web interface provides a simple way to upload images and perform object detection.

### 3. API Documentation

The API documentation is available at:

```
http://localhost:8000/docs
```

This interactive documentation allows you to test all the API endpoints directly from your browser.

## Using the API

### Web Interface

The web interface provides a simple way to:
1. Upload an image
2. Set the confidence threshold
3. Select which classes to detect (pedestrians, cars, buses, trucks)
4. View detection results

### Programmatic Access

#### Using cURL

You can use cURL to send requests to the API:

```bash
curl -X POST "http://localhost:8000/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg" \
  -F "conf=0.25"
```

#### Using Python

The included `test_api.py` script demonstrates how to use the API from Python:

```bash
# Basic usage
python test_api.py /path/to/your/image.jpg

# With custom confidence threshold
python test_api.py /path/to/your/image.jpg --conf 0.5

# Detect only specific classes (0=person, 2=car, 5=bus, 7=truck)
python test_api.py /path/to/your/image.jpg --classes 0 2

# Save output to a specific path
python test_api.py /path/to/your/image.jpg --output result.jpg
```

You can also integrate the API into your own Python code:

```python
import requests

def detect_objects(image_path, confidence=0.25):
    url = "http://localhost:8000/detect"
    
    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {"conf": confidence}
        
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            return None

# Example usage
results = detect_objects("image.jpg")
if results:
    for detection in results["objects_detected"]:
        print(f"Detected {detection['class_name']} with confidence {detection['confidence']}")
```

## Example Response

The API returns a JSON response with the following structure:

```json
{
  "message": "Detection completed successfully",
  "objects_detected": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.94,
      "bbox": {
        "x1": 234.5,
        "y1": 148.2,
        "x2": 374.1,
        "y2": 512.3,
        "width": 139.6,
        "height": 364.1
      }
    },
    {
      "class_id": 2,
      "class_name": "car",
      "confidence": 0.87,
      "bbox": {
        "x1": 45.2,
        "y1": 220.5,
        "x2": 210.8,
        "y2": 310.1,
        "width": 165.6,
        "height": 89.6
      }
    }
  ],
  "inference_time": "0.1423s",
  "result_image_url": "/static/results/be7f3a2d-6f24-4e6a-9f23-bce4a134b2e7.jpg",
  "original_image_url": "/static/uploads/5a2e8f7d-3c12-4e89-bc35-6f9a234d7e1c.jpg"
}
```

## Tips and Best Practices

1. **Confidence Threshold**: Adjust the confidence threshold to balance between detection accuracy and false positives. A value between 0.25 and 0.5 is usually reasonable.

2. **Classes**: By default, the API detects pedestrians and vehicles (cars, buses, trucks). You can specify which classes to detect to get more focused results.

3. **Image Size**: YOLOv8 works best with images that are clear and not too small. If your image is very large, consider resizing it to improve performance.

4. **Local Processing**: This API is designed to run locally or on an EC2 instance. It doesn't send your images to any external service, ensuring your data privacy.

5. **File Cleanup**: The API automatically cleans up old uploaded images and results to prevent disk space issues.

## Extending the API

If you want to customize the API for your own needs:

1. **Custom Model**: You can replace the YOLOv8 model with your own trained model by updating the `app/models/yolo_model.py` file.

2. **Additional Classes**: Modify the `CLASS_NAMES` dictionary in `app/models/yolo_model.py` to add support for detecting more object types.

3. **Performance Optimization**: For better performance, consider using a GPU-enabled instance when deploying to EC2. 