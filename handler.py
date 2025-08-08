#!/usr/bin/env python3
"""
RunPod serverless entrypoint at repository root.
Imports the actual handler from runpod_handler.py (repo has flat structure).
"""

print("🚀 Starting RunPod Whisper Training Handler...")

try:
    print("📋 Step 1: Importing runpod...")
    import runpod
    print("✅ RunPod imported successfully")
    
    print("📋 Step 2: Importing handler...")
    from runpod_handler import handler
    print("✅ Handler imported successfully")
    
    print("📋 Step 3: Starting serverless...")
    runpod.serverless.start({"handler": handler})
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("📋 Checking available modules...")
    import os
    print(f"Working directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    import sys
    print(f"Python path: {sys.path}")
    raise
except Exception as e:
    print(f"❌ General Error: {e}")
    import traceback
    traceback.print_exc()
    raise