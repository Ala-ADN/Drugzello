import time
import logging
import sys
import os
from typing import Dict, Any
from rdkit import Chem
from rdkit.Chem import Descriptors

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from core.model_loader import model_loader
from api.models import SolubilityPrediction

logger = logging.getLogger(__name__)

class InferenceService:
    """Service for handling molecule inference requests."""
    
    def __init__(self):
        self.model_loader = model_loader
    
    def validate_smiles(self, smiles: str) -> bool:
        """Validate SMILES string using RDKit."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except Exception as e:
            logger.error(f"SMILES validation error: {e}")
            return False
    
    def preprocess_molecule(self, smiles: str) -> Dict[str, Any]:
        """Preprocess molecule for inference."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Invalid SMILES string")
            
            # Calculate basic molecular descriptors
            molecular_weight = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            num_atoms = mol.GetNumAtoms()
            
            return {
                "smiles": smiles,
                "molecular_weight": molecular_weight,
                "logp": logp,
                "num_atoms": num_atoms,
                "mol": mol
            }
        except Exception as e:
            logger.error(f"Molecule preprocessing error: {e}")
            raise ValueError(f"Failed to preprocess molecule: {e}")
    
    def predict_solubility(self, smiles: str) -> SolubilityPrediction:
        """Predict solubility for a single molecule."""
        start_time = time.time()
        
        try:
            # Validate SMILES
            if not self.validate_smiles(smiles):
                raise ValueError("Invalid SMILES string")
            
            # Preprocess molecule
            mol_data = self.preprocess_molecule(smiles)
            
            # Get model
            model = self.model_loader.get_model()
            if model is None:
                raise RuntimeError("Model not loaded")
            
            # Make prediction
            if hasattr(model, 'predict'):
                prediction = model.predict(smiles)
            else:
                # Fallback for different model interfaces
                prediction = self._fallback_prediction(mol_data)
            
            # Extract solubility and confidence
            solubility = prediction.get('solubility', 0.0)
            confidence = prediction.get('confidence', 0.5)
            
            # Ensure values are within expected ranges
            solubility = max(-10.0, min(10.0, float(solubility)))
            confidence = max(0.0, min(1.0, float(confidence)))
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            logger.info(f"Prediction completed in {processing_time:.2f}ms for SMILES: {smiles}")
            
            return SolubilityPrediction(
                value=solubility,
                confidence=confidence,
                unit="log(mol/L)"
            )
            
        except Exception as e:
            logger.error(f"Prediction error for SMILES '{smiles}': {e}")
            raise ValueError(f"Prediction failed: {e}")
    
    def _fallback_prediction(self, mol_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback prediction method using simple heuristics."""
        # Simple heuristic based on molecular properties
        mw = mol_data.get('molecular_weight', 200)
        logp = mol_data.get('logp', 0)
        
        # Simple solubility estimation (not scientifically accurate)
        # Generally: higher molecular weight and logP reduce solubility
        estimated_solubility = 2.0 - (mw / 100) - logp
        estimated_confidence = 0.6  # Low confidence for heuristic
        
        return {
            "solubility": estimated_solubility,
            "confidence": estimated_confidence
        }
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model_loader.is_loaded()
    
    def get_model_version(self) -> str:
        """Get model version."""
        return self.model_loader.get_version() or "unknown"
