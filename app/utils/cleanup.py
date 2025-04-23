"""
File cleanup utility for the Object Detection API
This module provides functionality to clean up old files
"""
import os
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def clean_old_files(directory: str, max_age_days: int = 1, max_files: int = 100):
    """
    Remove old files from a directory based on age and count
    
    Args:
        directory: Directory to clean
        max_age_days: Maximum age of files in days
        max_files: Maximum number of files to keep
    
    Returns:
        Number of files removed
    """
    if not os.path.exists(directory):
        logger.warning(f"Directory {directory} does not exist, skipping cleanup")
        return 0
    
    # Get all files in the directory
    files = list(Path(directory).glob("*"))
    
    # Exclude .gitkeep file
    files = [f for f in files if f.name != ".gitkeep"]
    
    if not files:
        logger.info(f"No files to clean in {directory}")
        return 0
    
    # Sort files by modification time (oldest first)
    files.sort(key=lambda x: x.stat().st_mtime)
    
    removed_count = 0
    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60
    
    # Remove old files by age
    for file in files[:]:
        file_age = current_time - file.stat().st_mtime
        if file_age > max_age_seconds:
            try:
                file.unlink()
                removed_count += 1
                files.remove(file)
                logger.debug(f"Removed old file: {file} (age: {file_age/86400:.1f} days)")
            except Exception as e:
                logger.error(f"Error removing file {file}: {e}")
    
    # If we still have too many files, remove the oldest ones
    if len(files) > max_files:
        for file in files[:-max_files]:
            try:
                file.unlink()
                removed_count += 1
                logger.debug(f"Removed excess file: {file}")
            except Exception as e:
                logger.error(f"Error removing file {file}: {e}")
    
    if removed_count > 0:
        logger.info(f"Cleaned up {removed_count} files from {directory}")
    
    return removed_count

def setup_cleanup_task(app=None):
    """
    Set up a background task to clean up old files
    
    Args:
        app: FastAPI application instance
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Clean uploads and results directories
    uploads_dir = "app/static/uploads"
    results_dir = "app/static/results"
    
    # Perform initial cleanup
    clean_old_files(uploads_dir)
    clean_old_files(results_dir)
    
    # If running with FastAPI, register cleanup on startup
    if app:
        @app.on_event("startup")
        async def startup_cleanup():
            logger.info("Performing initial file cleanup on startup")
            clean_old_files(uploads_dir)
            clean_old_files(results_dir)

if __name__ == "__main__":
    # Run standalone cleanup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info("Running standalone file cleanup")
    uploads_cleaned = clean_old_files("app/static/uploads")
    results_cleaned = clean_old_files("app/static/results")
    logger.info(f"Cleanup complete. Removed {uploads_cleaned + results_cleaned} files.") 