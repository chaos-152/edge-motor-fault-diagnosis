# Multi-architecture Dockerfile for Edge-Based Fault Diagnosis of Induction Motors
# Compatible with x86_64 (PC/Servers) and ARM64 (Raspberry Pi 4)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Install system dependencies (build-essential, libhdf5 for h5py)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY config.py run_pipeline.py ./
COPY data/ ./data/
COPY dsp/ ./dsp/
COPY models/ ./models/
COPY edge/ ./edge/
COPY visualization/ ./visualization/
COPY tests/ ./tests/

# Default command: runs the complete pipeline
CMD ["python", "run_pipeline.py"]
