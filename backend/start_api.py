#!/usr/bin/env python3
"""
Startup script for the MEGAN Inference API
"""

import sys
import os

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Add the Drugzello directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if __name__ == "__main__":
    import uvicorn
    from api.main import app
    
    # Explicitly include all routes from backend.main
    from backend.main import app as backend_app
    for route in backend_app.routes:
        app.router.routes.append(route)
    
    print("🚀 Starting MEGAN Inference API...")
    print("📖 API Documentation will be available at: http://localhost:8000/docs")
    print("🔍 Health check available at: http://localhost:8000/health")
    print("🧪 Test script: python test_api.py")
    print("-" * 60)
    
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
