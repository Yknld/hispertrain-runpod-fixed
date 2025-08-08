#!/usr/bin/env python3
"""
RunPod serverless entrypoint at repository root.
Imports the actual handler from runpod_handler.py (repo has flat structure).
"""

import runpod
from runpod_handler import handler

def main():
    print("🚀 Starting RunPod Whisper Training Handler...")
    try:
        print("📋 Importing handler...")
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()