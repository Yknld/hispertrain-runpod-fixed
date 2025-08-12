# WhisperTrain RunPod Worker

Professional Whisper model training pipeline for RunPod serverless deployment with real data loading and comprehensive progress tracking.

## Features

- ✅ **Real Data Loading**: Downloads and processes actual audio/VTT files from backend
- ✅ **Robust Error Handling**: Comprehensive fallbacks for malformed data
- ✅ **Progress Tracking**: Real-time updates sent back to WhisperTrain backend
- ✅ **Dual Authentication**: Supports both Bearer tokens and X-Worker-Key
- ✅ **Resource Monitoring**: GPU memory, CPU, and system stats tracking
- ✅ **Checkpoint Management**: Automatic save/resume functionality
- ✅ **Production Ready**: Optimized for RunPod serverless environment

## Deployment

This repository is designed to be deployed as a RunPod serverless endpoint:

1. **GitHub Integration**: RunPod automatically builds from this repository
2. **Auto-scaling**: Serverless workers spin up on demand
3. **GPU Optimized**: Supports CUDA for accelerated training

## Input Format

The worker expects this input structure:

```json
{
  "model_id": "openai/whisper-tiny",
  "epochs": 1,
  "project_id": "uuid-string",
  "user_id": "uuid-string", 
  "backend_base_url": "https://your-backend.com",
  "auth_token": "jwt-token-or-empty",
  "worker_api_key": "shared-secret",
  "batch_size": 2,
  "learning_rate": 2e-05,
  "files_manifest": [
    {"file_type": "audio", "filename": "recording1.wav"},
    {"file_type": "transcript", "filename": "recording1.vtt"}
  ]
}
```

## Progress Authentication

The worker reports progress using either:
- **Bearer Token**: `Authorization: Bearer <jwt>` (preferred)
- **Worker Key**: `X-Worker-Key: <shared-secret>` (fallback)

## Analytics & Monitoring

Comprehensive metrics tracking:
- Training loss progression
- GPU memory usage
- Processing times
- System resource utilization
- Checkpoint creation

## Architecture

```
handler.py -> runpod_handler.py -> WhisperFileDataset -> Backend Files
    ↓              ↓                      ↓
RunPod Entry   Training Logic      Real Data Loading
    ↓              ↓                      ↓  
Progress      Model Training       Audio/VTT Processing
Updates       Checkpointing        Error Handling
```

## Dependencies

See `requirements.txt` for pinned versions ensuring compatibility:

- PyTorch ecosystem (transformers, datasets, accelerate)
- Audio processing (soundfile)
- HTTP client (httpx)
- System monitoring (psutil)
- RunPod SDK

## Error Handling

Robust fallbacks for common issues:
- **Missing Audio**: Falls back to silence samples
- **Empty Transcripts**: Uses minimal tokens
- **Malformed Data**: Skips with logging
- **Network Errors**: Retries with exponential backoff
- **Memory Issues**: Automatic cleanup and monitoring

## Local Testing

```bash
# Test locally (optional)
python handler.py
```

Note: Designed for RunPod environment - local testing has limitations.

---

**WhisperTrain**: The fastest way to train custom Whisper models with YouTube data.
