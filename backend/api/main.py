from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging
import sys
import os
from contextlib import asynccontextmanager
import asyncio

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
    logger.info("Starting application initialization")
    try:
        global startup_time, inference_service
        startup_time = time.time()
        
        # Initialize service and preload models
        logger.info("Initializing inference service and preloading models...")
        from services.inference_service import InferenceService
        inference_service = InferenceService()
        
        # Preload models in background
        async def preload_models():
            try:
                inference_service._load_molt5_model()
                inference_service._load_molt5_tokenizer()
                logger.info("Models preloaded successfully")
            except Exception as e:
                logger.error(f"Error preloading models: {e}")
                
        asyncio.create_task(preload_models())
        
        yield
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise
    finally:
        logger.info("Startup complete")

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
app.include_router(inference_router, tags=["inference"])
app.include_router(datasets_router, prefix="/api/v1", tags=["datasets"])
# Include datasets router again without prefix for frontend compatibility
app.include_router(datasets_router, tags=["frontend"])

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with basic information."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.version,
        "docs": "/docs",
        "health": "/health"
    }

# Update health check to be more resilient
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    try:
        uptime = time.time() - startup_time
        
        # Create immediate response
        response = HealthCheckResponse(
            status="healthy",
            model_loaded=False,
            model_version=None,
            uptime_seconds=uptime
        )
        
        # Don't block on model checks
        if inference_service is None:
            response.status = "degraded"
            logger.warning("Health check: Inference service not initialized")
        
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="degraded",
            model_loaded=False,
            model_version=None,
            uptime_seconds=0
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
