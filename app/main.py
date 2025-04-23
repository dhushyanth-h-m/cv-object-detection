from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.routers import detection
from app.utils.cleanup import setup_cleanup_task

app = FastAPI(
    title="Object Detection API",
    description="YOLOv8 object detection for pedestrians and vehicles",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers for JSON responses
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": str(exc.detail)}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"error": "Validation Error", "detail": str(exc)}
    )

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include routers
app.include_router(detection.router)

# Set up file cleanup task
setup_cleanup_task(app)

@app.get("/")
async def root():
    """Serve the HTML frontend"""
    return FileResponse('app/static/index.html')

@app.get("/api")
async def api_info():
    """Return API information"""
    return {
        "message": "Welcome to the Object Detection API",
        "docs": "/docs",
        "endpoints": {
            "detect": "/detect"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 