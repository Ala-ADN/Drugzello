from fastapi import APIRouter, HTTPException, status
import time
import logging
import sys
import os

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from api.models import (
    MoleculeInferenceRequest, 
    MoleculeInferenceResponse, 
    EnhancedInferenceRequest,
    EnhancedInferenceResponse,
    ErrorResponse,
    SolubilityPredictionRequest,
    SolubilityPredictionResponse
)
from services.inference_service import InferenceService

logger = logging.getLogger(__name__)

router = APIRouter()

# Cache the inference service
_inference_service = None

def get_inference_service():
    """Get or create inference service instance."""
    global _inference_service
    
    if _inference_service is not None:
        return _inference_service
        
    try:
        # Try to get the global instance from main.py
        import api.main as main_module
        if hasattr(main_module, 'inference_service') and main_module.inference_service is not None:
            _inference_service = main_module.inference_service
            return _inference_service
    except:
        pass
    
    # Fallback: create new instance
    _inference_service = InferenceService()
    return _inference_service

@router.post(
    "/predict",
    response_model=MoleculeInferenceResponse,
    summary="Predict molecule solubility",
    description="Predict the solubility of a molecule given its SMILES representation"
)
async def predict_molecule_solubility(request: MoleculeInferenceRequest):
    """
    Predict the solubility of a single molecule.
    
    - **smiles**: SMILES string representing the molecule
    
    Returns solubility prediction with confidence score.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing inference request for SMILES: {request.smiles}")
          # Get inference service and make prediction
        service = get_inference_service()
        prediction = service.predict_solubility_simple(request.smiles)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Create response
        response = MoleculeInferenceResponse(
            smiles=request.smiles,
            prediction=prediction,
            model_version=service.get_model_version(),
            processing_time_ms=processing_time_ms
        )
        
        logger.info(f"Successfully processed inference request in {processing_time_ms:.2f}ms")
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error for SMILES '{request.smiles}': {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=ErrorResponse(
                error="validation_error",
                message=str(e),
                details={"smiles": request.smiles}
            ).dict()
        )
    
    except RuntimeError as e:
        logger.error(f"Runtime error during inference: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorResponse(
                error="service_unavailable",
                message="Model service unavailable",
                details={"original_error": str(e)}
            ).dict()
        )
    
    except Exception as e:
        logger.error(f"Unexpected error during inference: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="internal_server_error",
                message="An unexpected error occurred during prediction"
            ).dict()
        )

@router.get(
    "/model/info",
    summary="Get model information",
    description="Get information about the loaded model"
)
async def get_model_info():
    """Get information about the currently loaded model."""
    try:
        service = get_inference_service()
        return {
            "model_loaded": service.is_model_loaded(),
            "model_version": service.get_model_version(),
            "model_type": "MEGAN",
            "supported_tasks": ["solubility_prediction"]
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model information"
        )

@router.post(
    "/predict/enhanced",
    response_model=EnhancedInferenceResponse,
    summary="Predict molecule solubility with interpretability",
    description="Predict the solubility of a molecule with optional uncertainty quantification, interpretability, and MolT5 analysis"
)
async def predict_molecule_solubility_enhanced(request: EnhancedInferenceRequest):
    """
    Predict the solubility of a single molecule with advanced features.
    
    - **smiles**: SMILES string representing the molecule
    - **include_uncertainty**: Whether to include uncertainty quantification
    - **include_explanations**: Whether to include molecular interpretability
    - **include_molt5**: Whether to include MolT5-based molecular analysis
    - **uncertainty_samples**: Number of Monte Carlo samples for uncertainty estimation
    
    Returns enhanced solubility prediction with optional uncertainty, explanations, and MolT5 analysis.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing enhanced inference request for SMILES: {request.smiles}")
        
        # Get inference service and make enhanced prediction
        service = get_inference_service()
        prediction = service.predict_solubility(
            request.smiles,
            include_uncertainty=request.include_uncertainty,
            include_explanations=request.include_explanations,
            include_molt5=request.include_molt5,
            uncertainty_samples=request.uncertainty_samples
        )
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Create response
        response = EnhancedInferenceResponse(
            smiles=request.smiles,
            prediction=prediction,
            model_version=service.get_model_version(),
            processing_time_ms=processing_time_ms
        )
        
        logger.info(f"Successfully processed enhanced inference request in {processing_time_ms:.2f}ms")
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error for SMILES '{request.smiles}': {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=ErrorResponse(
                error="validation_error",
                message=str(e),
                details={"smiles": request.smiles}
            ).dict()
        )
    
    except RuntimeError as e:
        logger.error(f"Runtime error during enhanced inference: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorResponse(
                error="service_unavailable",
                message="Model service unavailable",
                details={"original_error": str(e)}
            ).dict()
        )
    
    except Exception as e:
        logger.error(f"Unexpected error during enhanced inference: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="internal_server_error",
                message="An unexpected error occurred during enhanced prediction"
            ).dict()
        )
