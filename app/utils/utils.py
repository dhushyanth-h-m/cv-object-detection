import os
import uuid
from pathlib import Path
from typing import Dict, Any, List, Union
from fastapi import UploadFile

def save_uploaded_file(file: UploadFile, file_content: bytes) -> str:
    """
    Save an uploaded file to the uploads directory
    
    Args:
        file: The uploaded file object
        file_content: The file content as bytes
    
    Returns:
        The path where the file was saved
    """
    # Create a unique filename to avoid collisions
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = f"app/static/uploads/{unique_filename}"
    
    # Save the file
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    return file_path

def clean_old_files(directory: str, max_files: int = 100):
    """
    Remove old files from a directory when the number of files exceeds max_files
    
    Args:
        directory: Directory to clean
        max_files: Maximum number of files to keep
    """
    # Get all files in the directory
    files = list(Path(directory).glob("*"))
    
    # Sort files by creation time (oldest first)
    files.sort(key=lambda x: x.stat().st_ctime)
    
    # Remove old files if there are too many
    if len(files) > max_files:
        for file in files[:-max_files]:
            try:
                file.unlink()
            except Exception as e:
                print(f"Error removing file {file}: {e}") 