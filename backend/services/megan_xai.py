# megan_xai.py
"""
Service module for MEGAN model XAI logic, refactored from the original notebook.
"""
import os
import pickle
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from .fp_xai import get_weights_for_visualization, get_contour_image, mol2fp, AAU_weights, UAA_weights
from .megan_pytorch_new import MEGANCore, MEGANConfig, SearchSpace
import sys
sys.modules['__main__'].MEGANConfig = MEGANConfig
sys.modules['__main__'].SearchSpace = SearchSpace

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'megan_pytorch_model.pth')
TRAINING_RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'training_results.pkl')

# Load model and config at module import (singleton)
def _load_model():
    with open(TRAINING_RESULTS_PATH, "rb") as f:
        training_results = pickle.load(f)
    print("training_results keys:", training_results.keys())  # DEBUG: print keys
    # Try to get model_kwargs from best_config or results
    if "model_kwargs" in training_results:
        model_kwargs = training_results["model_kwargs"]
    elif "best_config" in training_results and training_results["best_config"] is not None:
        model_kwargs = training_results["best_config"]
    elif "results" in training_results and len(training_results["results"]) > 0 and "config" in training_results["results"][0]:
        model_kwargs = training_results["results"][0]["config"]
    else:
        raise KeyError("Could not find model configuration in training_results.pkl. Available keys: {}".format(training_results.keys()))
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model_state_dict = checkpoint["model_state_dict"]
    model_kwargs = checkpoint.get("model_kwargs", model_kwargs)
    model = MEGANCore(**model_kwargs)
    model.load_state_dict(model_state_dict)
    model.eval()
    def predict(self, data):
        with torch.no_grad():
            output, _, _ = self.forward(data.x, data.edge_index, data.batch, data.edge_attr)
            return output.cpu().numpy()
    import types
    model.predict = types.MethodType(predict, model)
    return model

MODEL = _load_model()

# Helper for Monte Carlo Dropout uncertainty
def monte_carlo_predictions(model, data, num_samples=10):
    model.train()
    predictions = []
    for _ in range(num_samples):
        with torch.no_grad():
            predictions.append(model.predict(data)[0])
    model.eval()
    return np.array(predictions)

def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    AllChem.Compute2DCoords(mol)
    return mol

def get_all_attributions(smiles):
    mol = smiles_to_mol(smiles)
    ml_weights, atom_weights, fpa_weights = get_weights_for_visualization(mol, MODEL, radius=2, n_bits=9)
    # Prepare input for prediction
    ecfp2_fingerprint = mol2fp(mol, radius=2, n_bits=9)
    from torch_geometric.data import Data
    data = Data(
        x=torch.tensor(ecfp2_fingerprint, dtype=torch.float).unsqueeze(0),
        edge_index=torch.tensor([[0], [0]], dtype=torch.long),
        edge_attr=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float),
        batch=torch.tensor([0], dtype=torch.long)
    )
    prediction = float(MODEL.predict(data)[0])
    # Uncertainty
    mc_predictions = monte_carlo_predictions(MODEL, data, num_samples=10)
    uncertainty = float(np.std(mc_predictions, axis=0))
    # UAA
    uaa_weights = UAA_weights(mol, MODEL, radius=2, n_bits=9)
    uaa_weights = np.array(uaa_weights).flatten() - 0.3
    # AAU
    aau_weights = AAU_weights(mol, MODEL, radius=2, n_bits=9)
    aau_weights = aau_weights - min(aau_weights)
    aau_weights = aau_weights * uncertainty / np.sum(aau_weights)
    # SVGs
    fig_ml = get_contour_image(mol, ml_weights)
    fig_atomic = get_contour_image(mol, atom_weights)
    fig_fpa = get_contour_image(mol, fpa_weights)
    fig_uaa = get_contour_image(mol, uaa_weights)
    fig_aau = get_contour_image(mol, aau_weights)
    return {
        "prediction": prediction,
        "uncertainty": uncertainty,
        "ml_weights": ml_weights.flatten().tolist(),
        "atom_weights": atom_weights.flatten().tolist(),
        "fpa_weights": fpa_weights.flatten().tolist(),
        "uaa_weights": uaa_weights.flatten().tolist(),
        "aau_weights": aau_weights.flatten().tolist(),
        "svg_ml": fig_ml,
        "svg_atomic": fig_atomic,
        "svg_fpa": fig_fpa,
        "svg_uaa": fig_uaa,
        "svg_aau": fig_aau
    }
