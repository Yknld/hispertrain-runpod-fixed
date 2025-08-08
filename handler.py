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
    
    try:
        # Extract input parameters
        input_data = event.get('input', {})
        model_id = input_data.get('model_id', 'openai/whisper-tiny')
        epochs = input_data.get('epochs', 1)
        project_id = input_data.get('project_id', 'unknown')
        backend_url = input_data.get('backend_base_url', '')
        auth_token = input_data.get('auth_token', '')
        worker_api_key = input_data.get('worker_api_key', '')
        batch_size = input_data.get('batch_size', 2)
        learning_rate = input_data.get('learning_rate', 2e-05)
        files_manifest = input_data.get('files_manifest', [])
        
        print(f"📋 Training parameters: model={model_id}, epochs={epochs}, project={project_id}", flush=True)
        print(f"📋 Files manifest: {len(files_manifest)} files", flush=True)
        
        # Import training modules here to avoid early import issues
        from runpod_handler import train_whisper_model, ProgressTracker
        
        # Create progress tracker
        progress_tracker = ProgressTracker(
            backend_url=backend_url,
            job_id=f"train_{project_id}_{int(datetime.now().timestamp())}",
            auth_token=auth_token,
            worker_api_key=worker_api_key
        )
        
        # Start training
        result = train_whisper_model(
            model_name=model_id,
            epochs=epochs,
            project_id=project_id,
            progress_tracker=progress_tracker,
            batch_size=batch_size,
            learning_rate=learning_rate,
            files_manifest=files_manifest
        )
        
        print(f"✅ Training completed successfully", flush=True)
        return {"success": True, "result": result}
        
    except Exception as e:
        error_msg = f"❌ Training failed: {str(e)}"
        print(error_msg, flush=True)
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

# Start the RunPod serverless worker
if __name__ == "__main__":
    print("🚀 Starting RunPod Serverless Worker...", flush=True)
    runpod.serverless.start({"handler": handler})