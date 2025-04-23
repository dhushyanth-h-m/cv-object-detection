# Troubleshooting Guide

This guide helps you diagnose and fix common issues with the Object Detection API.

## Common Issues

### 1. API Returns 422 Unprocessable Entity Error

This usually indicates an issue with the request parameters:

#### Possible causes:
- The form data is missing required fields
- File upload format is incorrect
- Class IDs are not being properly sent

#### Solutions:
- Make sure the image file is properly selected and uploaded
- Ensure the confidence threshold is a number between 0.1 and 1.0
- Check that class selections are being properly sent
- Try directly uploading an image through the Swagger UI at `/docs`

### 2. Model Download Issues

#### Possible causes:
- Internet connection problems
- Insufficient disk space
- Permission issues
- PyTorch 2.6+ security changes

#### Solutions:
- Check your internet connection
- Ensure you have at least 200MB of free disk space
- Run the application with appropriate permissions
- If using PyTorch 2.6+, the system will attempt to use a fallback detector

### 3. "Weights only load failed" Error

This is a security feature in PyTorch 2.6+.

#### Possible causes:
- You're using PyTorch 2.6 or newer which has stricter security settings
- The YOLOv8 model isn't on the allowed list of safe globals

#### Solutions:
The system is configured to handle this automatically by:
1. Adding YOLOv8 classes to the safe globals list if possible
2. Temporarily patching the torch.load function to use weights_only=False
3. Falling back to a simple detector if YOLOv8 can't be loaded

You don't need to take any action as the system will automatically fall back to a simpler detector when this happens.

### 4. No Objects Detected

#### Possible causes:
- Image quality is poor
- Confidence threshold is too high
- No objects of the selected classes are in the image
- Using fallback detector (if YOLOv8 failed to load)

#### Solutions:
- Try a clearer image with obvious objects
- Lower the confidence threshold (e.g., to 0.2)
- Make sure you've selected the appropriate classes to detect
- Check the server logs to see if the fallback detector is being used

## Using the Diagnostic Page

For troubleshooting API issues, we've added a diagnostic page that provides detailed information about API requests and responses:

1. Visit http://localhost:8000/static/diagnostic.html
2. Upload an image file
3. Set detection parameters
4. Click "Run Test"
5. Check the log and request details for errors

This tool will help diagnose API issues by showing the raw request and response data.

## Testing with Simple Images

If you're having trouble with complex photos, try our test image generator:

```bash
python create_test_image.py
```

This creates a simple test image with basic shapes that can be used to verify that the API is functioning.

## Checking Logs

If you're running the server in development mode, you can see detailed logs in the terminal. Look for error messages that might indicate what's going wrong.

## Common Error Messages

### "No module named 'ultralytics'"
This means the dependencies are not properly installed. Run:
```bash
pip install -r requirements.txt
```

### "CUDA not available"
This is normal on machines without a GPU. The model will run on CPU, which is slower but will still work.

## Still Having Issues?

If you're still experiencing problems:

1. Try restarting the server
2. Check that all requirements are properly installed
3. Try using a different image file
4. Use the Docker container instead of local installation

If all else fails, the system will automatically use a fallback detection method that doesn't require YOLOv8. The fallback detector uses basic computer vision techniques and may not be as accurate, but it will allow the API to function. 