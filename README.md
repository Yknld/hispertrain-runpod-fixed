# 🚀 WhisperTrain RunPod Training Pipeline

Professional Whisper model training with comprehensive analytics and monitoring.

## 🎯 Features

- **Real-time Progress Tracking**: Live training metrics and ETA
- **Smart Checkpointing**: Auto-save every 50 steps + recovery
- **Comprehensive Analytics**: Loss progression, system monitoring  
- **Production Ready**: Error handling, logging, performance optimization

## 🛠️ RunPod Deployment

### Quick Setup
1. Use this repo URL in RunPod serverless endpoint
2. Set handler: `handler.py`
3. Configure environment variables
4. Deploy!

### Environment Variables
```bash
BACKEND_BASE_URL=https://your-backend.com
HF_TOKEN=your_huggingface_token  # Optional
```

### Input Format
```json
{
  "model_id": "openai/whisper-small",
  "epochs": 3,
  "project_id": "project_123", 
  "user_id": "user_456",
  "backend_base_url": "https://your-backend.com",
  "auth_token": "your_auth_token",
  "worker_api_key": "optional_shared_secret"
}
```

### Progress Authentication
- If `auth_token` is provided, the worker posts with `Authorization: Bearer <token>`
- If no token, and `worker_api_key` is present, it posts with `X-Worker-Key: <worker_api_key>`

Backend must accept one of the above on `/api/training/progress/{job_id}`. You can set `WORKER_API_KEY` in backend env and pass it via `extra.worker_api_key` from the launcher.

## 📊 Training Analytics

The system tracks:
- Training/validation loss progression
- Learning rate scheduling
- System resource usage (CPU, memory, GPU)
- Checkpoint creation and management
- Real-time progress updates to backend

## 🔧 Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend       │    │   RunPod        │
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │ Training UI │ │───▶│ │ Job Launcher │ │───▶│ │   Handler   │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ └─────────────┘ │
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │Progress View│ │◀───│ │Progress API  │ │◀───│ │Progress Hook│ │
│ └─────────────┘ │    │ └──────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Getting Started

1. **Setup RunPod Endpoint**:
   - Point to this GitHub repo
   - Set handler: `handler.py`
   - Configure env vars

2. **Backend Integration**:
   - Implement `/api/training/progress/{job_id}` endpoint
   - Support X-Worker-Key authentication
   - Handle job status updates

3. **Launch Training**:
   ```python
   job_input = {
       "model_id": "openai/whisper-small",
       "epochs": 3,
       "project_id": "my-project",
       "backend_base_url": "https://my-backend.com",
       "worker_api_key": "my-secret-key"
   }
   ```

## 📈 Progress Updates

The handler sends real-time updates during training:

- `initializing`: Setting up environment
- `loading_model`: Downloading base model
- `preparing_dataset`: Processing training data
- `training_started`: Beginning training loop
- `epoch_started`: Starting new epoch
- `epoch_completed`: Epoch finished with metrics
- `checkpoint_saved`: Model checkpoint created
- `training_completed`: Training finished
- `saving_model`: Finalizing model artifacts
- `completed`: Job completed successfully
- `failed`: Error occurred

## 🛡️ Error Handling

- Comprehensive logging to `/tmp/training.log`
- Automatic fallback for older transformers versions
- Progress tracking even during failures
- System resource monitoring
- Graceful error reporting to backend

## 📦 Dependencies

All required packages are listed in `requirements.txt` with pinned versions for consistency:

- `transformers>=4.41.0`: Core Whisper functionality
- `datasets>=2.18.0`: Dataset processing
- `torch` + `torchaudio`: PyTorch backend
- `runpod==1.7.13`: Serverless platform integration
- `httpx>=0.27.0`: HTTP client for progress updates
- Additional ML libraries for comprehensive training

## 🔍 Monitoring

Training provides detailed metrics:
- Loss curves (training/validation)
- Learning rate scheduling
- System resource usage
- GPU memory utilization
- Training speed (samples/second)
- Estimated completion time

## 🎯 Production Features

- **Automatic Checkpointing**: Save every 50 steps
- **Resume Training**: Automatic checkpoint recovery
- **Resource Monitoring**: Track CPU, memory, GPU usage
- **Error Recovery**: Graceful failure handling
- **Real-time Updates**: Live progress to backend
- **Comprehensive Logging**: Detailed training logs
- **Metadata Tracking**: Complete training history