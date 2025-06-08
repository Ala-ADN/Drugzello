from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
import re

class MoleculeInferenceRequest(BaseModel):
    """Request model for single molecule inference."""
    
    smiles: str = Field(
        ..., 
        description="SMILES string representing the molecule",
        example="CCO"
    )
    
    @validator('smiles')
    def validate_smiles(cls, v):
        """Basic SMILES validation."""
        if not v or not v.strip():
            raise ValueError("SMILES string cannot be empty")
        
        # Basic SMILES pattern validation (simplified)
        if not re.match(r'^[A-Za-z0-9@+\-\[\]()=#\.\/\\%:]+$', v.strip()):
            raise ValueError("Invalid SMILES format")
        
        return v.strip()

class SolubilityPrediction(BaseModel):
    """Solubility prediction result."""
    
    value: float = Field(
        ..., 
        description="Predicted solubility value",
        ge=-10.0,
        le=10.0
    )
    confidence: float = Field(
        ..., 
        description="Confidence score for the prediction",
        ge=0.0,
        le=1.0
    )
    unit: str = Field(
        default="log(mol/L)",
        description="Unit of the solubility prediction"
    )

class MoleculeInferenceResponse(BaseModel):
    """Response model for single molecule inference."""
    
    model_config = {"protected_namespaces": ()}
    
    smiles: str = Field(..., description="Input SMILES string")
    prediction: SolubilityPrediction
    model_version: str = Field(..., description="Version of the model used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    
class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Additional error details"
    )

class HealthCheckResponse(BaseModel):
    """Health check response model."""
    
    model_config = {"protected_namespaces": ()}
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: Optional[str] = Field(default=None, description="Model version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
