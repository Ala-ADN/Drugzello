# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements/ requirements/

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# Preinstall PyTorch and PyG wheels to avoid source builds for torch-scatter
RUN pip install torch==2.4.1 && \
    pip install torch-scatter==2.1.2 torch-geometric==2.6.1 -f https://data.pyg.org/whl/torch-2.4.1+cpu.html

# Install Python dependencies (try to use binary wheels)
RUN pip install --only-binary=all --no-cache-dir -r requirements/development.txt || \
    pip install --no-cache-dir -r requirements/development.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p logs models/trained models/experiments data/processed data/features

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose port for FastAPI
EXPOSE 8000

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
