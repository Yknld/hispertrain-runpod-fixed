#!/usr/bin/env python3
"""
Simple RunPod handler for testing
"""

print("🚀 Starting Simple Handler...")

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
        
        return {
            "status": "COMPLETED",
            "output": {
                "message": "Handler working successfully!",
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available()
            }
        }
    except Exception as e:
        print(f"❌ Error in handler: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "FAILED",
            "error": str(e)
        }

if __name__ == "__main__":
    print("📋 Starting RunPod serverless...")
    import runpod
    runpod.serverless.start({"handler": handler})