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

# Make handler executable
RUN chmod +x handler_simple.py

# RunPod expects the handler at the root with this name
COPY handler_simple.py handler.py

# Add startup logging
RUN echo "#!/bin/bash" > /start.sh && \
    echo "echo '🚀 Container starting...'" >> /start.sh && \
    echo "echo '📋 Files in root:'" >> /start.sh && \
    echo "ls -la /" >> /start.sh && \
    echo "echo '📋 Starting Python handler...'" >> /start.sh && \
    echo "python /handler.py" >> /start.sh && \
    chmod +x /start.sh