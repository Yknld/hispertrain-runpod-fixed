#!/usr/bin/env python3
"""
RunPod Serverless Handler for Whisper Training
"""

import os
import sys
import logging
import runpod
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def handler(event):
    """Main RunPod serverless handler function"""
    try:
        print(f"🚀 Handler started at {datetime.now()}", flush=True)
        print(f"📋 Received event: {event}", flush=True)
        
        # Extract input parameters
        input_data = event.get('input', {})
        model = input_data.get('model', 'openai/whisper-tiny')
        epochs = input_data.get('epochs', 1)
        project = input_data.get('project', 'unknown')
        backend_url = input_data.get('backend_url', '')
        auth_token = input_data.get('auth_token', '')
        worker_api_key = input_data.get('worker_api_key', '')
        
        print(f"📋 Training parameters: model={model}, epochs={epochs}, project={project}", flush=True)
        
        # Import training modules here to avoid early import issues
        from runpod_handler import train_whisper_model, ProgressTracker
        
        # Create progress tracker
        progress_tracker = ProgressTracker(
            backend_url=backend_url,
            job_id=f"train_{project}_{int(datetime.now().timestamp())}",
            auth_token=auth_token,
            worker_api_key=worker_api_key
        )
        
        # Start training
        result = train_whisper_model(
            model_name=model,
            epochs=epochs,
            project_id=project,
            progress_tracker=progress_tracker
        )
        
        print(f"✅ Training completed successfully", flush=True)
        return {"success": True, "result": result}
        
    except Exception as e:
        error_msg = f"❌ Training failed: {str(e)}"
        print(error_msg, flush=True)
        logging.error(error_msg, exc_info=True)
        return {"success": False, "error": str(e)}

# Start the RunPod serverless worker
if __name__ == "__main__":
    print("🚀 Starting RunPod Serverless Worker...", flush=True)
    runpod.serverless.start({"handler": handler})