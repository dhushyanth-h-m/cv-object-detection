# Object Detection for Autonomous Systems

This project implements a YOLOv8-based object detection system for identifying pedestrians and vehicles in images. It provides a FastAPI web service for real-time inference and is containerized with Docker.

## Features

- YOLOv8 object detection model focusing on pedestrians and vehicles
- FastAPI web service for real-time inference
- Docker containerization for easy deployment
- Supports local deployment with potential for AWS EC2 deployment
- Well-documented API with Swagger UI

## Requirements

- Python 3.8+
- Docker and Docker Compose (optional, for containerized deployment)

## Installation

### Option 1: Local Installation

1. Clone this repository:
```bash
git clone https://github.com/dhushyanth-h-m/cv-object-detection
cd object-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
uvicorn app.main:app --reload
```

### Option 2: Docker Deployment

1. Clone this repository:
```bash
git clone https://github.com/yourusername/object-detection.git
cd object-detection
```

2. Build and run with Docker Compose:
```bash
docker-compose up -d
```

## API Usage

Once the server is running, you can access:

- API Documentation: http://localhost:8000/docs
- API Endpoints:
  - `/detect`: POST endpoint for object detection
  - `/result/{filename}`: GET endpoint to retrieve result images

### Object Detection Endpoint

To detect objects in an image:

```bash
curl -X POST "http://localhost:8000/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg" \
  -F "conf=0.25"
```

Parameters:
- `file`: Image file to analyze
- `conf` (optional): Confidence threshold (0-1), default is 0.25
- `classes` (optional): List of class IDs to detect (0=person, 2=car, 5=bus, 7=truck)

Response example:
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

## Architecture

The project follows a modular architecture:

- `app/main.py`: FastAPI application setup
- `app/models/yolo_model.py`: YOLOv8 model implementation
- `app/routers/detection.py`: API endpoints for object detection
- `app/utils/`: Utility functions for file handling
- `app/static/`: Storage for uploaded and result images

## Deploying to AWS EC2 (Future)

For EC2 deployment:

1. Launch an EC2 instance (recommend t2.large or better with GPU for optimal performance)
2. Install Docker and Docker Compose
3. Clone this repository and deploy using Docker Compose
4. Configure security groups to allow inbound traffic on port 8000

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- FastAPI framework 