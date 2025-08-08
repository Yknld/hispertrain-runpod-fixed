# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /

# Early build diagnostics
RUN echo "🏗️ Starting Docker build..." && \
    echo "Base image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel" && \
    python -V && pip --version

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the handler and training module
COPY handler.py /handler.py
COPY runpod_handler.py /runpod_handler.py

# RunPod serverless entry point
CMD ["python", "/handler.py"]