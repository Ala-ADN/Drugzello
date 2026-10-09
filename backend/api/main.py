"""Canonical API: versioned trained MEGAN checkpoints, no heuristic fallback."""

from contextlib import asynccontextmanager
from pathlib import Path
import sys
import time

backend_dir = str(Path(__file__).resolve().parents[1])
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rdkit import Chem

from core.config import settings
from core.model_loader import model_loader
from services.inference_service import InferenceService
from api.models import HealthCheckResponse
from api.endpoints.inference import router as inference_router

startup_time = time.monotonic()
inference_service = InferenceService()


@asynccontextmanager
async def lifespan(app):
    global startup_time
    startup_time = time.monotonic()
    model_loader.load_model()
    yield
    model_loader.unload_model()


app = FastAPI(title=settings.app_name, version="2.0.0", lifespan=lifespan,
              description="Aqueous logS prediction with trained MEGAN and learned explanation masks")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(inference_router)


@app.get("/health", response_model=HealthCheckResponse)
def health():
    loaded = model_loader.is_loaded()
    return HealthCheckResponse(status="healthy" if loaded else "degraded", model_loaded=loaded,
                               model_version=model_loader.get_version(),
                               uptime_seconds=time.monotonic() - startup_time)


@app.get("/ready")
def ready():
    if not model_loader.is_loaded():
        raise HTTPException(503, "Trained checkpoint unavailable")
    return {"status": "ready", "model_version": model_loader.get_version()}


# Example molecule catalog, separate from model predictions. In-memory additions
# are session-only; no solvent-dependent benchmark or database is implied.
_molecules = [
    {"id": 1, "name": "Aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O"},
    {"id": 2, "name": "Caffeine", "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O"},
]


class MoleculeCreate(BaseModel):
    name: str
    smiles: str


@app.get("/molecules")
def molecules():
    return _molecules


@app.post("/molecules")
def create_molecule(request: MoleculeCreate):
    mol = Chem.MolFromSmiles(request.smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise HTTPException(422, "Invalid molecule")
    record = {"id": len(_molecules) + 1, **request.model_dump()}
    _molecules.append(record)
    return record


@app.get("/solvents")
def solvents():
    return ["Water"]


class SolubilityRequest(BaseModel):
    smiles: str | None = None
    molecule_id: int | None = None
    solvent: str = "Water"


@app.post("/solubility")
def solubility(request: SolubilityRequest):
    if request.solvent.lower() != "water":
        raise HTTPException(422, "The ESOL model supports aqueous solubility only")
    smiles = request.smiles
    if request.molecule_id is not None:
        record = next((m for m in _molecules if m["id"] == request.molecule_id), None)
        if record is None:
            raise HTTPException(404, "Molecule not found")
        smiles = record["smiles"]
    if not smiles:
        raise HTTPException(422, "Provide SMILES or a molecule ID")
    try:
        prediction = inference_service.predict_solubility_simple(smiles)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"solubility": prediction.value, "unit": "logS",
            "model_version": inference_service.get_model_version()}


@app.get("/")
def root():
    return {"message": settings.app_name, "docs": "/docs", "health": "/health"}
