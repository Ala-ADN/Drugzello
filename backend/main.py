from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import logging
import time
from datetime import datetime

from src.utils.config import load_config
from src.models.trainer import ModelTrainer
from src.utils.mlflow_integration import MLFlowLogger

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()

# Initialize FastAPI app
app = FastAPI(
    title="Drugzello ML API",
    description="API for molecular solubility prediction using MEGAN architecture",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get("cors_origins", ["http://localhost:5173"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class MoleculeCreate(BaseModel):
    name: str
    smiles: str

class Molecule(BaseModel):
    id: int
    name: str
    smiles: str

class SolubilityRequest(BaseModel):
    molecule_id: Optional[int] = None
    smiles: Optional[str] = None
    solvent: str

class SolubilityResult(BaseModel):
    solubility: float
    explanation: str

# Mock database (replace with actual database in production)
MOCK_MOLECULES = [
    {"id": 1, "name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
    {"id": 2, "name": "Caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"}
]
MOCK_SOLVENTS = [
    "Water", "Ethanol", "DMSO"
]

# --- Health check endpoints ---
@app.get("/health")
def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.get("/ready")
def readiness_check():
    """Readiness check endpoint."""
    # Add any checks for external dependencies here
    # e.g., database connectivity, MLflow availability
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {
            "mlflow": "ok",  # Add actual MLflow connectivity check
            "storage": "ok"   # Add actual storage connectivity check
        }
    }

# --- Endpoints ---
@app.get("/molecules", response_model=List[Molecule])
def list_molecules():
    logger.info("Listing all molecules")
    return MOCK_MOLECULES

@app.post("/molecules", response_model=Molecule)
def create_molecule(mol: MoleculeCreate):
    logger.info(f"Creating new molecule: {mol.name}")
    # Mock rdkit verification
    if not mol.smiles or "@" in mol.smiles:
        logger.warning(f"Invalid molecule SMILES: {mol.smiles}")
        raise HTTPException(status_code=400, detail="Invalid molecule (mocked rdkit check)")
    new_id = len(MOCK_MOLECULES) + 1
    molecule = {"id": new_id, **mol.dict()}
    MOCK_MOLECULES.append(molecule)
    return molecule

@app.get("/solvents", response_model=List[str])
def list_solvents():
    logger.info("Listing all solvents")
    return MOCK_SOLVENTS

@app.post("/solubility", response_model=SolubilityResult)
def check_solubility(req: SolubilityRequest):
    logger.info(f"Processing solubility request for solvent: {req.solvent}")
    
    # Mock preprocessing, MEGAN model, and postprocessing
    if req.molecule_id:
        mol = next((m for m in MOCK_MOLECULES if m["id"] == req.molecule_id), None)
        if not mol:
            logger.warning(f"Molecule ID not found: {req.molecule_id}")
            raise HTTPException(status_code=404, detail="Molecule not found")
        smiles = mol["smiles"]
    elif req.smiles:
        # Mock rdkit verification
        if not req.smiles or "@" in req.smiles:
            logger.warning(f"Invalid molecule SMILES: {req.smiles}")
            raise HTTPException(status_code=400, detail="Invalid molecule (mocked rdkit check)")
        smiles = req.smiles
    else:
        logger.warning("No molecule provided in request")
        raise HTTPException(status_code=400, detail="No molecule provided")
    
    # Mock MEGAN model and calculation
    solubility = 0.42  # Mocked value
    explanation = f"Solubility prediction for {smiles} in {req.solvent}."
    logger.info(f"Generated solubility prediction: {solubility}")
    
    return SolubilityResult(solubility=solubility, explanation=explanation)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Drugzello Solubility API!"}

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please contact support."}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
