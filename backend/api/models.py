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

class DatasetConfig(BaseModel):
    """Configuration for a single dataset with solvent."""
    
    name: str = Field(..., description="Name of the MoleculeNet dataset", example="ESOL")
    solvent: str = Field(..., description="Solvent name for this dataset", example="water")

class MultiSolventDatasetRequest(BaseModel):
    """Request model for loading multi-solvent datasets."""
    
    dataset_configs: List[DatasetConfig] = Field(
        ..., 
        description="List of dataset configurations",
        example=[
            {"name": "ESOL", "solvent": "water"},
            {"name": "FreeSolv", "solvent": "water"},
            {"name": "Lipophilicity", "solvent": "octanol"}        ]
    )
    handle_duplicates: str = Field(
        default="remove",
        description="Strategy for handling duplicates",
        example="remove"
    )
    split_type: str = Field(
        default="kfold", 
        description="Type of data splitting",
        example="kfold"
    )
    n_folds: int = Field(default=5, description="Number of folds for k-fold", ge=2, le=20)
    normalize_by_solvent: bool = Field(
        default=True, 
        description="Whether to normalize targets separately by solvent"
    )
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    
    @field_validator('handle_duplicates')
    @classmethod
    def validate_duplicate_strategy(cls, v):
        valid_strategies = ['remove', 'keep_first', 'keep_last', 'average']
        if v not in valid_strategies:
            raise ValueError(f"handle_duplicates must be one of {valid_strategies}")
        return v
    
    @field_validator('split_type')
    @classmethod
    def validate_split_type(cls, v):
        valid_types = ['random', 'kfold', 'scaffold']
        if v not in valid_types:
            raise ValueError(f"split_type must be one of {valid_types}")
        return v

class CustomDatasetRequest(BaseModel):
    """Request model for loading custom datasets from CSV."""
    
    csv_data: str = Field(..., description="CSV content as string")
    smiles_col: str = Field(default="smiles", description="Name of SMILES column")
    target_col: str = Field(default="solubility", description="Name of target column")
    solvent_col: Optional[str] = Field(default=None, description="Name of solvent column")

class DatasetStatistics(BaseModel):
    """Dataset statistics response model."""
    
    num_samples: int = Field(..., description="Total number of samples")
    num_node_features: int = Field(..., description="Number of node features")
    num_edge_features: int = Field(..., description="Number of edge features")
    target_mean: float = Field(..., description="Mean target value")
    target_std: float = Field(..., description="Standard deviation of targets")
    target_min: float = Field(..., description="Minimum target value")
    target_max: float = Field(..., description="Maximum target value")
    avg_nodes_per_graph: float = Field(..., description="Average nodes per molecule")
    avg_edges_per_graph: float = Field(..., description="Average edges per molecule")
    max_nodes_per_graph: int = Field(..., description="Maximum nodes per molecule")
    max_edges_per_graph: int = Field(..., description="Maximum edges per molecule")

class SolventInfo(BaseModel):
    """Information about a solvent in the dataset."""
    
    name: str = Field(..., description="Solvent name")
    id: int = Field(..., description="Solvent ID in vocabulary")
    sample_count: int = Field(..., description="Number of samples for this solvent")

class DuplicateInfo(BaseModel):
    """Information about duplicate handling."""
    
    method: str = Field(..., description="Duplicate handling method used")
    duplicates_found: int = Field(..., description="Number of duplicates found")
    duplicates_removed: int = Field(..., description="Number of duplicates removed")

class MultiSolventDatasetResponse(BaseModel):
    """Response model for multi-solvent dataset operations."""
    
    dataset_id: str = Field(..., description="Unique identifier for this dataset")
    total_samples: int = Field(..., description="Total number of samples in dataset")
    solvents: List[SolventInfo] = Field(..., description="Information about solvents")
    statistics: DatasetStatistics = Field(..., description="Dataset statistics")
    duplicate_info: DuplicateInfo = Field(..., description="Duplicate handling information")
    normalization_applied: bool = Field(..., description="Whether normalization was applied")
    splits_available: bool = Field(..., description="Whether data splits are available")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

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

class ExplanationResults(BaseModel):
    """Molecular interpretation and explanation results."""
    
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
