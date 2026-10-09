"""SMILES inference service with trained MEGAN outputs and honest availability."""

from rdkit import Chem

from core.model_loader import model_loader
from api.models import (SolubilityPrediction, EnhancedSolubilityPrediction,
                        UncertaintyAnalysis, ExplanationResults)


class InferenceService:
    def __init__(self, loader=None):
        self._model_loader = loader if loader is not None else model_loader

    @staticmethod
    def validate_smiles(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None and mol.GetNumAtoms() > 0

    def is_model_loaded(self):
        return self._model_loader.is_loaded()

    def get_model_version(self):
        return self._model_loader.get_version() or "unavailable"

    def predict_solubility_simple(self, smiles):
        prediction = self.predict_solubility(smiles)
        return SolubilityPrediction(value=prediction.value, confidence=None, unit="logS")

    def predict_solubility(self, smiles, include_uncertainty=False, include_explanations=False,
                           include_molt5=False, uncertainty_samples=20):
        if not self.validate_smiles(smiles):
            raise ValueError("Invalid SMILES or empty molecule")
        if include_molt5:
            raise ValueError("MolT5 interpretation is not supported by the trained MEGAN service")
        model = self._model_loader.get_model()
        result = model.predict_with_uncertainty(
            smiles, n_samples=uncertainty_samples if include_uncertainty else 1,
            include_explanations=include_explanations
        )
        response = EnhancedSolubilityPrediction(value=result["solubility"], confidence=None,
                                                unit="logS", model_type=result["model_type"])
        if include_uncertainty:
            response.uncertainty = UncertaintyAnalysis(**result["uncertainty"])
        if include_explanations:
            response.explanations = ExplanationResults(
                node_importances=result["node_importances"], edge_importances=result["edge_importances"],
                node_importances_by_channel=result["node_importances_by_channel"],
                bond_importances_by_channel=result["bond_importances_by_channel"],
                channel_labels=result["channel_labels"],
                interpretation="Learned MEGAN masks; channel polarity requires measured fidelity validation.",
                rdkit_comparison={},
            )
        return response
