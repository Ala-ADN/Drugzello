"""Check the actual all-data refit through the in-process API; no server needed."""

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "backend/models/trained/scaffold-tuned/model.pth"
os.environ["MEGAN_MODEL_PATH"] = str(WEIGHTS)
sys.path.insert(0, str(ROOT / "backend"))

import torch
from fastapi.testclient import TestClient
from rdkit import Chem
from api.main import app
from core.model_loader import model_loader


def main():
    torch.manual_seed(42)
    records = []
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
        assert client.get("/health").json()["model_loaded"] is True
        for smiles in ("CCO", "CC(=O)Oc1ccccc1C(=O)O"):
            response = client.post("/predict/enhanced", json={"smiles": smiles,
                                   "include_explanations": True, "include_uncertainty": True,
                                   "uncertainty_samples": 10})
            assert response.status_code == 200, response.text
            body = response.json()
            prediction = body["prediction"]
            assert prediction["confidence"] is None
            assert prediction["unit"] == "logS"
            assert prediction["model_type"] == "megan_pytorch"
            assert body["model_version"].startswith("megan-regression-embedded-v3")
            mol = Chem.MolFromSmiles(smiles)
            explanations = prediction["explanations"]
            assert len(explanations["node_importances"]) == mol.GetNumAtoms()
            assert len(explanations["edge_importances"]) == mol.GetNumBonds()
            nodes, bonds = explanations["node_importances_by_channel"], explanations["bond_importances_by_channel"]
            assert len(nodes) == mol.GetNumAtoms() and all(len(row) == 2 for row in nodes)
            assert len(bonds) == mol.GetNumBonds() and all(len(row) == 2 for row in bonds)
            assert all(0 <= value <= 1 for row in nodes + bonds for value in row)
            assert prediction["uncertainty"]["prediction_std"] >= 0
            assert not model_loader.get_model().model.training
            simple = client.post("/solubility", json={"smiles": smiles, "solvent": "Water"})
            assert simple.status_code == 200
            value = simple.json()["solubility"]
            assert value == prediction["value"]
            repeated = client.post("/solubility", json={"smiles": smiles, "solvent": "Water"})
            assert repeated.json()["solubility"] == value
            records.append({"smiles": smiles, "deterministic_prediction_logS": value,
                            "mc_dropout_std_logS": prediction["uncertainty"]["prediction_std"],
                            "mc_samples": 10, "atoms": mol.GetNumAtoms(), "bonds": mol.GetNumBonds(),
                            "model_version": body["model_version"], "confidence": None})
        assert client.post("/predict", json={"smiles": "garbage"}).status_code == 422
        assert client.post("/solubility", json={"smiles": "CCO", "solvent": "DMSO"}).status_code == 422
        assert client.post("/predict/enhanced", json={"smiles": "CCO", "include_molt5": True}).status_code == 422
    evidence = {"date": "2026-10-09", "status": "passed",
                "checkpoint_sha256": hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
                "ready_http_status": 200, "learned_mask_alignment_checked": True,
                "deterministic_inference_restored_after_mc_dropout": True,
                "invalid_inputs_and_unsupported_options_return_422": True, "examples": records,
                "scope": "API integration check, not an accuracy or uncertainty calibration benchmark"}
    Path(__file__).with_name("api_verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
