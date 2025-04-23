#!/usr/bin/env python3
"""
Object Detection API
Run this script to start the FastAPI application
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 