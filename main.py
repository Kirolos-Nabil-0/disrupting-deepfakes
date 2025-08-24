"""
Entry point for the Disrupting Deepfakes API.
This file serves as the main entry point for DigitalOcean App Platform deployment.
"""

import sys
import os

# Add the api directory to the Python path
api_dir = os.path.join(os.path.dirname(__file__), 'api')
sys.path.insert(0, api_dir)

# Import the FastAPI app from the api directory
from api.main import app

# Make the app available at module level for uvicorn
__all__ = ['app']

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable (DigitalOcean sets $PORT)
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )