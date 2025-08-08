FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a simple handler.py that RunPod will auto-detect
RUN echo 'import runpod' > /handler.py && \
    echo 'def handler(event):' >> /handler.py && \
    echo '    print(f"📋 Received event: {event}")' >> /handler.py && \
    echo '    return {"message": "Hello from RunPod!"}' >> /handler.py && \
    echo 'runpod.serverless.start({"handler": handler})' >> /handler.py