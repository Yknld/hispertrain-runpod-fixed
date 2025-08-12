#!/usr/bin/env python3
"""
RunPod Serverless Handler for Whisper Training
"""

import runpod
from datetime import datetime
import asyncio

async def handler(event):
    """Main RunPod serverless handler function (async)."""
    print(f"🚀 Handler started at {datetime.now()}", flush=True)
    print(f"📋 Received event: {event}", flush=True)

    try:
        # Extract input parameters
        input_data = event.get("input", {})
        model_id = input_data.get("model_id", "openai/whisper-tiny")
        epochs = int(input_data.get("epochs", 1))
        project_id = input_data.get("project_id", "unknown")
        user_id = input_data.get("user_id", "")
        backend_base_url = input_data.get("backend_base_url", "")
        auth_token = input_data.get("auth_token", "")
        worker_api_key = input_data.get("worker_api_key", "")
        batch_size = input_data.get('batch_size', 2)
        learning_rate = input_data.get('learning_rate', 2e-05)
        files_manifest = input_data.get('files_manifest', [])
        job_id = input_data.get('job_id')

        print(
            f"📋 Training parameters: model={model_id}, epochs={epochs}, project={project_id}",
            flush=True,
        )
        print(f"📋 Files manifest: {len(files_manifest)} files", flush=True)

        # Import training coroutine and await it
        from runpod_handler import run_training

        result = await run_training(
            base_model=model_id,
            epochs=epochs,
            project_id=project_id,
            user_id=user_id,
            backend_base_url=backend_base_url,
            auth_token=auth_token,
            worker_api_key=worker_api_key,
            batch_size=batch_size,
            learning_rate=learning_rate,
            files_manifest=files_manifest,
            job_id=job_id
        )

        print("✅ Training completed successfully", flush=True)
        return {"success": True, "result": result}

    except Exception as e:
        import traceback

        error_msg = f"❌ Training failed: {str(e)}"
        print(error_msg, flush=True)
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# Start the RunPod serverless worker
if __name__ == "__main__":
    print("🚀 Starting RunPod Serverless Worker...", flush=True)
    runpod.serverless.start({"handler": handler})
