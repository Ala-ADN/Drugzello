# Drugzello: Solubility Prediction Web App

This project is a full-stack web application for molecular solubility prediction. It features a FastAPI backend and a React (Vite) frontend.

## Project Structure

- `main.py` — FastAPI backend (API for molecules, solvents, and solubility prediction)
- `MEGAN.ipynb` — Notebook for MEGAN model and calculations (not yet integrated)
- `frontend/` — React Vite frontend

## Backend (FastAPI)

### Requirements
- Python 3.8+
- pip

### Setup & Run
1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install fastapi uvicorn pydantic
   ```
3. Start the API server:
   ```bash
   uvicorn main:app --reload
   ```
- API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Frontend (React + Vite)

### Requirements
- Node.js (v18+ recommended)
- npm

### Setup & Run
1. Open a new terminal and navigate to the frontend folder:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
- App runs at: [http://localhost:5173](http://localhost:5173)

## Usage Flow
1. User visits the landing page (React frontend)
2. User can create/select a molecule (API: `/molecules`)
3. If creating, molecular integrity is checked (mocked, will use rdkit)
4. User selects a solvent (API: `/solvents`)
5. User initiates solubility check (API: `/solubility`)
6. Backend mocks ML/MEGAN logic and returns a result
7. Frontend visualizes and explains the result

## Notes
- The backend currently mocks all chemistry/model logic.
- The MEGAN model and RDKit integration are not yet implemented.
- The frontend is a placeholder template.

---

Feel free to contribute or extend the project!
