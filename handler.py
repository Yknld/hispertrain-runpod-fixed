#!/usr/bin/env python3
"""
RunPod Serverless Handler for Whisper Training
"""

import runpod
from datetime import datetime

def handler(event):
    """Main RunPod serverless handler function"""
    print(f"🚀 Handler started at {datetime.now()}", flush=True)
    print(f"📋 Received event: {event}", flush=True)
    
    # Just return success for now to test if handler is working
    return {"success": True, "message": "Handler is working!", "timestamp": str(datetime.now())}

# Start the RunPod serverless worker
if __name__ == "__main__":
    print("🚀 Starting RunPod Serverless Worker...", flush=True)
    runpod.serverless.start({"handler": handler})