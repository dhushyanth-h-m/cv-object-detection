#!/bin/bash
# Start the object detection API

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Start the application
echo "Starting the API server..."
echo "Visit http://localhost:8000 in your browser to access the web interface"
echo "API documentation available at http://localhost:8000/docs"
echo "Press Ctrl+C to stop the server"
python run.py 