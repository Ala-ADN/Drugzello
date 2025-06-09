"""
API endpoints for dataset loading and management.
"""

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
import time
import logging
import sys
import os
import hashlib
import json
from typing import Dict, Any
import pandas as pd
from io import StringIO

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from api.models import (
    MultiSolventDatasetRequest, 
    MultiSolventDatasetResponse, 
    CustomDatasetRequest,
    DatasetStatistics,
    SolventInfo,
    DuplicateInfo,
    ErrorResponse
)
from src.data.data_loader_enhanced import load_multi_solvent_molecular_data, MolecularDataLoader
from core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory dataset storage (for demo purposes - in production use proper storage)
_dataset_cache: Dict[str, Dict[str, Any]] = {}

def _generate_dataset_id(config: MultiSolventDatasetRequest) -> str:
    """Generate a unique ID for a dataset configuration."""
    config_str = json.dumps(config.dict(), sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:16]

@router.post("/datasets/multi-solvent", response_model=MultiSolventDatasetResponse)
async def load_multi_solvent_dataset(request: MultiSolventDatasetRequest):
    """
    Load and process a multi-solvent molecular dataset.
    
    This endpoint loads multiple MoleculeNet datasets, concatenates them by solvent,
    handles duplicates according to the specified strategy, and creates data splits.
    """
    start_time = time.time()
    
    try:
        # Generate dataset ID
        dataset_id = _generate_dataset_id(request)
        
        # Check if dataset is already cached
        if dataset_id in _dataset_cache:
            logger.info(f"Returning cached dataset {dataset_id}")
            cached_data = _dataset_cache[dataset_id]
            processing_time = (time.time() - start_time) * 1000
            
            return MultiSolventDatasetResponse(
                dataset_id=dataset_id,
                total_samples=cached_data['total_samples'],
                solvents=cached_data['solvents'],
                statistics=cached_data['statistics'],
                duplicate_info=cached_data['duplicate_info'],
                normalization_applied=cached_data['normalization_applied'],
                splits_available=cached_data['splits_available'],
                processing_time_ms=processing_time
            )
        
        # Convert request to the format expected by the dataloader
        dataset_configs = [config.dict() for config in request.dataset_configs]
        
        logger.info(f"Loading multi-solvent dataset with configs: {dataset_configs}")
        
        # Load the dataset
        result = load_multi_solvent_molecular_data(
            dataset_configs=dataset_configs,
            data_root=str(settings.data_path),
            handle_duplicates=request.handle_duplicates,
            split_type=request.split_type,
            n_folds=request.n_folds,
            normalize_by_solvent=request.normalize_by_solvent,
            random_state=request.random_state
        )
        
        # Extract information for response
        dataset = result['dataset']
        stats = result['stats']
        duplicate_info = result['duplicate_info']
        solvent_vocab = result['solvent_vocab']
        
        # Build solvent information
        solvents = []
        for solvent_name, solvent_id in solvent_vocab.items():
            # Count samples for this solvent
            sample_count = sum(1 for data in dataset if data.solvent_name == solvent_name)
            solvents.append(SolventInfo(
                name=solvent_name,
                id=solvent_id,
                sample_count=sample_count
            ))
        
        # Create response data
        response_data = {
            'total_samples': len(dataset),
            'solvents': solvents,
            'statistics': DatasetStatistics(**stats),
            'duplicate_info': DuplicateInfo(**duplicate_info),
            'normalization_applied': request.normalize_by_solvent or result.get('norm_params') is not None,
            'splits_available': 'splits' in result and result['splits'] is not None
        }
        
        # Cache the dataset and metadata
        _dataset_cache[dataset_id] = {
            'dataset': result,
            'request': request.dict(),
            **response_data
        }
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Successfully loaded dataset {dataset_id} with {len(dataset)} samples in {processing_time:.1f}ms")
        
        return MultiSolventDatasetResponse(
            dataset_id=dataset_id,
            processing_time_ms=processing_time,
            **response_data
        )
        
    except Exception as e:
        logger.error(f"Error loading multi-solvent dataset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load dataset: {str(e)}"
        )

@router.post("/datasets/custom", response_model=MultiSolventDatasetResponse)
async def load_custom_dataset(request: CustomDatasetRequest):
    """
    Load a custom molecular dataset from CSV data.
    
    This endpoint accepts CSV data as a string and converts it to a molecular dataset
    with SMILES-to-graph conversion.
    """
    start_time = time.time()
    
    try:
        # Parse CSV data
        csv_io = StringIO(request.csv_data)
        df = pd.read_csv(csv_io)
        
        logger.info(f"Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")
        
        # Validate required columns
        if request.smiles_col not in df.columns:
            raise ValueError(f"SMILES column '{request.smiles_col}' not found in CSV")
        if request.target_col not in df.columns:
            raise ValueError(f"Target column '{request.target_col}' not found in CSV")
        
        # Create temporary CSV file for the loader
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            csv_path = tmp_file.name
        
        try:
            # Load using the enhanced dataloader
            loader = MolecularDataLoader(str(settings.data_path))
            dataset = loader.load_custom_dataset(
                csv_path=csv_path,
                smiles_col=request.smiles_col,
                target_col=request.target_col,
                solvent_col=request.solvent_col
            )
            
            # Get statistics
            stats = loader.get_dataset_statistics(dataset)
            
            # Create solvent info if solvent column provided
            solvents = []
            if request.solvent_col and request.solvent_col in df.columns:
                unique_solvents = df[request.solvent_col].unique()
                for i, solvent in enumerate(unique_solvents):
                    count = len(df[df[request.solvent_col] == solvent])
                    solvents.append(SolventInfo(name=solvent, id=i, sample_count=count))
            else:
                solvents.append(SolventInfo(name="unknown", id=0, sample_count=len(dataset)))
            
            # Generate dataset ID
            dataset_id = hashlib.md5(request.csv_data.encode()).hexdigest()[:16]
            
            # No duplicates handling for custom datasets by default
            duplicate_info = DuplicateInfo(
                method="none",
                duplicates_found=0,
                duplicates_removed=0
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            logger.info(f"Successfully loaded custom dataset {dataset_id} with {len(dataset)} samples")
            
            return MultiSolventDatasetResponse(
                dataset_id=dataset_id,
                total_samples=len(dataset),
                solvents=solvents,
                statistics=DatasetStatistics(**stats),
                duplicate_info=duplicate_info,
                normalization_applied=False,
                splits_available=False,
                processing_time_ms=processing_time
            )
            
        finally:
            # Clean up temporary file
            os.unlink(csv_path)
        
    except Exception as e:
        logger.error(f"Error loading custom dataset: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to load custom dataset: {str(e)}"
        )

@router.get("/datasets/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """
    Get information about a previously loaded dataset.
    """
    if dataset_id not in _dataset_cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )
    
    cached_data = _dataset_cache[dataset_id]
    
    return {
        'dataset_id': dataset_id,
        'total_samples': cached_data['total_samples'],
        'solvents': cached_data['solvents'],
        'statistics': cached_data['statistics'],
        'duplicate_info': cached_data['duplicate_info'],
        'normalization_applied': cached_data['normalization_applied'],
        'splits_available': cached_data['splits_available'],
        'request_config': cached_data['request']
    }

@router.get("/datasets/{dataset_id}/splits")
async def get_dataset_splits(dataset_id: str):
    """
    Get data splits for a dataset.
    """
    if dataset_id not in _dataset_cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )
    
    cached_data = _dataset_cache[dataset_id]
    dataset_result = cached_data['dataset']
    
    if 'splits' not in dataset_result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No splits available for this dataset"
        )
    
    splits = dataset_result['splits']
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_splits(obj):
        if isinstance(obj, dict):
            return {k: convert_splits(v) for k, v in obj.items()}
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        else:
            return obj
    
    return {
        'dataset_id': dataset_id,
        'splits': convert_splits(splits)
    }

@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """
    Delete a dataset from cache.
    """
    if dataset_id not in _dataset_cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )
    
    del _dataset_cache[dataset_id]
    
    return {"message": f"Dataset {dataset_id} deleted successfully"}

@router.get("/datasets")
async def list_datasets():
    """
    List all cached datasets.
    """
    dataset_list = []
    for dataset_id, cached_data in _dataset_cache.items():
        dataset_list.append({
            'dataset_id': dataset_id,
            'total_samples': cached_data['total_samples'],
            'num_solvents': len(cached_data['solvents']),
            'normalization_applied': cached_data['normalization_applied'],
            'splits_available': cached_data['splits_available']
        })
    
    return {
        'datasets': dataset_list,
        'total_cached': len(dataset_list)
    }

@router.get("/available-datasets")
async def get_available_datasets():
    """
    Get list of available MoleculeNet datasets that can be loaded.
    """
    # This could be enhanced to dynamically check available datasets
    available_datasets = [
        {
            'name': 'ESOL',
            'description': 'Estimated Solubility Dataset',
            'target': 'log solubility in mols per litre',
            'samples': '~1128'
        },
        {
            'name': 'FreeSolv',
            'description': 'Free Solvation Database',
            'target': 'hydration free energy',
            'samples': '~642'
        },
        {
            'name': 'Lipophilicity',
            'description': 'Lipophilicity Dataset',
            'target': 'octanol/water distribution coefficient',
            'samples': '~4200'
        }
    ]
    
    return {
        'available_datasets': available_datasets
    }

# Mock data for frontend compatibility
MOCK_MOLECULES = [
    {"id": 1, "name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
    {"id": 2, "name": "Caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"},
    {"id": 3, "name": "Paracetamol", "smiles": "CC(=O)NC1=CC=C(C=C1)O"},
    {"id": 4, "name": "Ibuprofen", "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"},
    {"id": 5, "name": "Ethanol", "smiles": "CCO"}
]

MOCK_SOLVENTS = ["water", "ethanol", "acetone", "dmso", "chloroform"]

@router.get("/molecules")
async def list_molecules():
    """Get list of available molecules for the frontend."""
    logger.info("Listing all molecules")
    return MOCK_MOLECULES

@router.get("/solvents")
async def list_solvents():
    """Get list of available solvents for the frontend."""
    logger.info("Listing all solvents")
    return MOCK_SOLVENTS
