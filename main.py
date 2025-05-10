from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mock database ---
MOCK_MOLECULES = [
    {"id": 1, "name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
    {"id": 2, "name": "Caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"}
]
MOCK_SOLVENTS = [
    "Water", "Ethanol", "DMSO"
]

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

# --- Endpoints ---
@app.get("/molecules", response_model=List[Molecule])
def list_molecules():
    return MOCK_MOLECULES

@app.post("/molecules", response_model=Molecule)
def create_molecule(mol: MoleculeCreate):
    # Mock rdkit verification
    if not mol.smiles or "@" in mol.smiles:
        raise HTTPException(status_code=400, detail="Invalid molecule (mocked rdkit check)")
    new_id = len(MOCK_MOLECULES) + 1
    molecule = {"id": new_id, **mol.dict()}
    MOCK_MOLECULES.append(molecule)
    return molecule

@app.get("/solvents", response_model=List[str])
def list_solvents():
    return MOCK_SOLVENTS

@app.post("/solubility", response_model=SolubilityResult)
def check_solubility(req: SolubilityRequest):
    # Mock preprocessing, MEGAN model, and postprocessing
    if req.molecule_id:
        mol = next((m for m in MOCK_MOLECULES if m["id"] == req.molecule_id), None)
        if not mol:
            raise HTTPException(status_code=404, detail="Molecule not found")
        smiles = mol["smiles"]
    elif req.smiles:
        # Mock rdkit verification
        if not req.smiles or "@" in req.smiles:
            raise HTTPException(status_code=400, detail="Invalid molecule (mocked rdkit check)")
        smiles = req.smiles
    else:
        raise HTTPException(status_code=400, detail="No molecule provided")
    # Mock MEGAN model and calculation
    solubility = 0.42  # Mocked value
    explanation = f"Mocked solubility prediction for {smiles} in {req.solvent}. (MEGAN/ML pipeline not implemented)"
    return SolubilityResult(solubility=solubility, explanation=explanation)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Drugzello Solubility API!"}
