#!/usr/bin/env python3
"""
RunPod serverless entrypoint at repository root.
Imports the actual handler from runpod_handler.py (repo has flat structure).
"""

from runpod_handler import handler

if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})