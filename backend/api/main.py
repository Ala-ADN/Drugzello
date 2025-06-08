from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging
import sys
import os
from contextlib import asynccontextmanager

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from core.config import settings
from api.models import HealthCheckResponse, ErrorResponse
from api.endpoints.inference import router as inference_router
from api.endpoints.datasets import router as datasets_router

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Global variable to track startup time
startup_time = time.time()

# Global inference service instance
inference_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for the FastAPI application."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.version}")
    global startup_time, inference_service
    startup_time = time.time()
      # Initialize global inference service
    from services.inference_service import InferenceService
    inference_service = InferenceService()
    # Pre-load the model to ensure it's available
    model = inference_service.model_loader.get_model()  # This triggers model loading
    if model:
        logger.info(f"Model loaded successfully: {inference_service.get_model_version()}")
    else:
        logger.warning("Failed to load model during startup")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Simple FastAPI backend for single molecule solubility inference using MEGAN model",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(inference_router, prefix="/api/v1", tags=["inference"])
app.include_router(datasets_router, prefix="/api/v1", tags=["datasets"])

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with basic information."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.version,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    try:
        global inference_service
        
        # Use the global inference service if available, otherwise create new one
        if inference_service is None:
            from services.inference_service import InferenceService
            inference_service = InferenceService()
        
        model_loaded = inference_service.is_model_loaded()
        model_version = inference_service.get_model_version() if model_loaded else None
        
        uptime = time.time() - startup_time
        
        return HealthCheckResponse(
            status="healthy",
            model_loaded=model_loaded,
            model_version=model_version,
            uptime_seconds=uptime
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_server_error",
            message="An internal server error occurred"
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
