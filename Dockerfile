FROM runpod/pytorch:2.2.0-py3.11-cuda12.1.1-devel

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set the handler
CMD ["python", "handler.py"]