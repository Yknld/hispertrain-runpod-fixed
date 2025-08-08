#!/usr/bin/env python3
"""
Simple RunPod handler for testing
"""

import os
import sys
import time

print("🚀 Starting Simple Handler...", flush=True)
print(f"📋 Python version: {sys.version}", flush=True)
print(f"📋 Working directory: {os.getcwd()}", flush=True)
print(f"📋 Python path: {sys.path}", flush=True)
print(f"📋 Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

def handler(event):
    """Simple test handler"""
    print(f"📋 Received event: {event}")
    
    try:
        # Test basic imports
        print("📋 Testing imports...")
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")
        
        # Get job input
        job_input = event.get("input", {})
        print(f"📋 Job input: {job_input}")
        
        return {
            "message": "Handler working successfully!",
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "transformers_version": transformers.__version__
        }
    except Exception as e:
        print(f"❌ Error in handler: {e}")
        import traceback
        traceback.print_exc()
        raise e

# RunPod serverless entry point
print("📋 Importing runpod...", flush=True)
import runpod
print("📋 Starting runpod serverless...", flush=True)
runpod.serverless.start({"handler": handler})
print("📋 RunPod serverless started!", flush=True)