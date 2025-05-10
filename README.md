# FastAPI Starter Project

This project includes a minimal FastAPI setup.

## Requirements
- Python 3.8+
- pip

## Installation
1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install FastAPI and Uvicorn:
   ```bash
   pip install fastapi uvicorn
   ```

## Running the Server
Start the FastAPI server with Uvicorn:
```bash
uvicorn main:app --reload
```

- Open your browser and go to: http://127.0.0.1:8000
- Interactive API docs: http://127.0.0.1:8000/docs
