import os
import uuid
import time
from pathlib import Path
from typing import List, Optional, Union

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse
import numpy as np
from PIL import Image
import io

from app.models.yolo_model import YOLOModel
from app.utils.utils import save_uploaded_file
from app.utils.test_image_generator import generate_test_image

router = APIRouter(tags=["Detection"])

# Initialize the YOLO model (lazy loading - will be loaded on first detection)
model = YOLOModel()

@router.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify the API is working"""
    return {"status": "ok", "message": "Detection API is working"}

@router.get("/test-model")
async def test_model():
    """Test the YOLO model initialization"""
    try:
        # Try to initialize the model
        result = {
            "model_initialized": model is not None,
            "model_type": str(type(model)),
        }
        
        # Check if model has been loaded
        if hasattr(model, '_model') and model._model is not None:
            result["model_loaded"] = True
            result["model_instance_type"] = str(type(model._model))
        else:
            result["model_loaded"] = False
            # Try to load the model
            try:
                test_model = model.model  # This will trigger lazy loading
                result["model_load_success"] = True
                result["model_instance_type"] = str(type(test_model))
            except Exception as e:
                result["model_load_success"] = False
                result["model_load_error"] = str(e)
        
        return result
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": f"Model test failed: {str(e)}",
            "traceback": traceback.format_exc()
        }

@router.post("/detect")
async def detect_objects(
    request: Request,
    file: UploadFile = File(...),
    conf: Optional[float] = Form(0.25)
):
    """
    Detect pedestrians and vehicles in an uploaded image.
    
    - **file**: Image file to analyze
    - **conf**: Confidence threshold (0-1)
    - **classes**: List of class IDs to detect (0=person, 2=car, 5=bus, 7=truck)
                   If None, detects all classes
    """
    try:
        # Check if file is an image
        content_type = file.content_type or ""
        if not content_type.startswith("image/"):
            return JSONResponse(
                status_code=400,
                content={"error": "Uploaded file is not an image"}
            )
        
        # Read the image file
        image_content = await file.read()
        
        # Try to open the image
        try:
            image = Image.open(io.BytesIO(image_content))
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid image file: {str(e)}"}
            )
        
        # Save the uploaded file
        file_path = save_uploaded_file(file, image_content)
        
        # Extract classes from form data 
        final_classes = None
        try:
            form = await request.form()
            if "classes" in form:
                form_classes_data = form.getlist("classes")
                if form_classes_data:
                    try:
                        final_classes = [int(c) for c in form_classes_data]
                        print(f"Classes from form: {final_classes}")
                    except (ValueError, TypeError) as e:
                        print(f"Error parsing classes from form: {e}")
        except Exception as form_error:
            print(f"Error processing form data: {form_error}")
            
        # Perform detection
        start_time = time.time()
        results = model.detect(
            image, 
            conf_threshold=conf,
            classes=final_classes
        )
        inference_time = time.time() - start_time
        
        # Check for valid results
        if not results or "image_path" not in results:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Detection failed",
                    "message": "The model did not return valid results"
                }
            )
        
        # Get the result image path
        result_image_path = results["image_path"]
        
        # Return the results
        return {
            "message": "Detection completed successfully",
            "objects_detected": results["detections"],
            "inference_time": f"{inference_time:.4f}s",
            "result_image_url": f"/static/results/{os.path.basename(result_image_path)}",
            "original_image_url": f"/static/uploads/{os.path.basename(file_path)}"
        }
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error in detect_objects: {str(e)}")
        print(error_traceback)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Detection failed",
                "message": str(e),
                "traceback": error_traceback.split("\n")
            }
        )

@router.get("/result/{filename}")
async def get_result_image(filename: str):
    """Get a result image by filename"""
    file_path = Path(f"app/static/results/{filename}")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Result image not found")
    return FileResponse(file_path)

@router.api_route("/generate-test-image", methods=["GET", "POST"])
async def create_test_image(
    width: int = 640, 
    height: int = 480,
    num_shapes: int = 5
):
    """
    Generate a test image with random shapes for testing object detection
    
    - **width**: Width of the image (default: 640)
    - **height**: Height of the image (default: 480)
    - **num_shapes**: Number of shapes to generate (default: 5)
    """
    try:
        # Create a unique filename for the test image
        filename = f"test_image_{uuid.uuid4()}.jpg"
        output_path = f"app/static/test_images/{filename}"
        
        # Ensure the directory exists
        os.makedirs("app/static/test_images", exist_ok=True)
        
        # Generate the test image
        generate_test_image(
            width=width, 
            height=height, 
            num_shapes=num_shapes, 
            output_path=output_path
        )
        
        return {
            "message": "Test image generated successfully",
            "image_url": f"/static/test_images/{filename}",
            "width": width,
            "height": height,
            "num_shapes": num_shapes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating test image: {str(e)}")

@router.get("/test-image-save")
async def test_image_save():
    """Test image saving functionality"""
    try:
        from app.utils.test_image_generator import generate_test_image
        
        # Generate a test image
        test_path = "app/static/test_images/test_save.jpg"
        try:
            # Ensure the directory exists
            os.makedirs("app/static/test_images", exist_ok=True)
            
            # Generate a small test image
            generate_test_image(
                width=320, 
                height=240, 
                num_shapes=2, 
                output_path=test_path
            )
            
            # Try to open and save the image with PIL
            try:
                from PIL import Image
                img = Image.open(test_path)
                img_size = img.size
                
                # Save it to a results folder
                result_path = "app/static/results/test_save_result.jpg"
                os.makedirs("app/static/results", exist_ok=True)
                img.save(result_path)
                
                return {
                    "status": "success",
                    "message": "Test image saved successfully",
                    "image_size": img_size,
                    "original_path": test_path,
                    "result_path": result_path
                }
            except Exception as pil_error:
                return {
                    "status": "error",
                    "message": f"PIL test failed: {str(pil_error)}"
                }
                
        except Exception as gen_error:
            return {
                "status": "error",
                "message": f"Test image generation failed: {str(gen_error)}"
            }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": f"Image save test failed: {str(e)}",
            "traceback": traceback.format_exc()
        }

@router.get("/test-detect")
async def test_detect():
    """Test detection on a generated test image"""
    try:
        # 1. Generate a test image
        from app.utils.test_image_generator import generate_test_image
        test_path = "app/static/test_images/test_detect.jpg"
        os.makedirs("app/static/test_images", exist_ok=True)
        generate_test_image(
            width=640, 
            height=480, 
            num_shapes=5, 
            output_path=test_path
        )
        
        # 2. Load the image
        from PIL import Image
        img = Image.open(test_path)
        
        # 3. Run detection
        start_time = time.time()
        try:
            results = model.detect(
                img, 
                conf_threshold=0.25,
                classes=None  # Detect all supported classes
            )
            inference_time = time.time() - start_time
            
            return {
                "status": "success",
                "message": "Detection test completed successfully",
                "inference_time": f"{inference_time:.4f}s",
                "detections_count": len(results.get("detections", [])),
                "result_path": results.get("image_path", "")
            }
        except Exception as detect_err:
            import traceback
            return {
                "status": "error",
                "message": f"Detection failed: {str(detect_err)}",
                "traceback": traceback.format_exc()
            }
            
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": f"Test detection failed: {str(e)}",
            "traceback": traceback.format_exc()
        } 