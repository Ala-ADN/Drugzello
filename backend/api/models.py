from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
import re

class MoleculeInferenceRequest(BaseModel):
    """Request model for single molecule inference."""
    
    smiles: str = Field(
        ..., 
        description="SMILES string representing the molecule",
        example="CCO"
    )
    
    @field_validator('smiles')
    @classmethod
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
        description="Predicted aqueous logS; not clamped to a training range",
        allow_inf_nan=False
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Calibrated confidence when available; null for this model",
        ge=0.0,
        le=1.0
    )
    unit: str = Field(
        default="logS",
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








class UncertaintyAnalysis(BaseModel):
    """Uncertainty analysis results."""
    
    prediction_std: float = Field(
        ..., 
        description="Standard deviation of prediction across Monte Carlo samples"
    )
    uaa_score: float = Field(
        ..., 
        description="Uncertainty in Atomic Attribution score"
    )
    aau_scores: List[float] = Field(
        ..., 
        description="Atomic Attribution of Uncertainty scores for each atom"
    )

class MolT5Interpretation(BaseModel):
    """MolT5-based molecular interpretation results."""
    
    prompt: str = Field(
        ..., 
        description="The prompt used for MolT5 analysis"
    )
    analysis: str = Field(
        ..., 
        description="MolT5's detailed molecular analysis"
    )
    confidence: float = Field(
        ..., 
        description="Confidence score for the interpretation",
        ge=0.0,
        le=1.0
    )
    model_version: str = Field(
        ..., 
        description="Version of the MolT5 model used"
    )
    channel_idx: int = Field(
        default=0,
        description="The explanation channel used for analysis"
    )

class ExplanationResults(BaseModel):
    """Learned masks aligned to RDKit atoms and bonds (no added self-loops)."""

    node_importances_by_channel: List[List[float]] = Field(default_factory=list)
    bond_importances_by_channel: List[List[float]] = Field(default_factory=list)
    channel_labels: List[str] = Field(default_factory=list)
    
    node_importances: List[float] = Field(
        ..., 
        description="Importance scores for each atom in the molecule"
    )
    edge_importances: List[float] = Field(
        ..., 
        description="Importance scores for each bond in the molecule"
    )
    interpretation: str = Field(
        ..., 
        description="Human-readable interpretation of the prediction"
    )
    rdkit_comparison: Dict[str, float] = Field(
        ..., 
        description="Comparison with RDKit baseline predictions"
    )
    molt5_interpretation: Optional[MolT5Interpretation] = Field(
        default=None,
        description="MolT5-based molecular interpretation"
    )

class EnhancedSolubilityPrediction(SolubilityPrediction):
    """Enhanced solubility prediction with uncertainty and interpretability."""
    
    model_config = {"protected_namespaces": ()}
    
    uncertainty: Optional[UncertaintyAnalysis] = Field(
        default=None,
        description="Uncertainty quantification results"
    )
    explanations: Optional[ExplanationResults] = Field(
        default=None,
        description="Interpretability and explanation results"
    )
    molt5_analysis: Optional[MolT5Interpretation] = Field(  # Add this field
        default=None,
        description="MolT5 analysis when requested without full explanations"
    )
    model_type: str = Field(
        default="megan",
        description="Type of model used for prediction"
    )

class EnhancedInferenceRequest(MoleculeInferenceRequest):
    """Enhanced inference request with interpretability options."""
    
    include_uncertainty: bool = Field(
        default=False,
        description="Whether to include uncertainty quantification"
    )
    include_explanations: bool = Field(
        default=False,
        description="Whether to include molecular interpretability"
    )
    include_molt5: bool = Field(
        default=False,
        description="Whether to include MolT5-based molecular analysis"
    )
    uncertainty_samples: int = Field(
        default=50,
        description="Number of Monte Carlo samples for uncertainty estimation",
        ge=10,
        le=200
    )

class EnhancedInferenceResponse(BaseModel):
    """Enhanced response model with interpretability features."""
    
    model_config = {"protected_namespaces": ()}
    
    smiles: str = Field(..., description="Input SMILES string")
    prediction: EnhancedSolubilityPrediction
    model_version: str = Field(..., description="Version of the model used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
