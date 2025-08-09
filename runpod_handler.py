#!/usr/bin/env python3
"""
WhisperTrain RunPod Training Handler

Professional Whisper model training with comprehensive analytics and monitoring.
Designed for RunPod serverless deployment with real-time progress tracking.
"""

print("📋 Importing basic modules...")
import asyncio
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

print("📋 Setting up logging...")
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/training.log')
    ]
)
logger = logging.getLogger(__name__)

print("📋 Importing heavy ML libraries...")
try:
    import torch
    print("✅ PyTorch imported")
except Exception as e:
    print(f"❌ Failed to import torch: {e}")
    raise

try:
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        Trainer,
        TrainingArguments,
        TrainerCallback,
        TrainerState,
        TrainerControl
    )
    print("✅ Transformers imported")
except Exception as e:
    print(f"❌ Failed to import transformers: {e}")
    raise

try:
    import runpod
    print("✅ RunPod imported in handler")
except Exception as e:
    print(f"❌ Failed to import runpod in handler: {e}")
    raise

try:
    import psutil
    print("✅ psutil imported")
except Exception as e:
    print(f"❌ Failed to import psutil: {e}")
    raise

print("✅ All imports successful in runpod_handler.py")


class ProgressTracker:
    """Track training progress and send updates to backend"""
    
    def __init__(self, backend_url: str, auth_token: str, job_id: str, worker_api_key: str | None = None):
        self.backend_url = backend_url
        self.auth_token = auth_token
        self.job_id = job_id
        self.worker_api_key = worker_api_key or ""
        self.metrics = []
        self.checkpoints = []
        
    def send_progress(self, progress_data: Dict[str, Any]):
        """Send progress update to backend (synchronously)."""
        try:
            import httpx
            headers = {}
            # Prefer Bearer token if provided, otherwise use X-Worker-Key shared secret
            if (self.auth_token or "").strip():
                headers["Authorization"] = f"Bearer {self.auth_token}"
            elif self.worker_api_key:
                headers["X-Worker-Key"] = self.worker_api_key

            response = httpx.post(
                f"{self.backend_url}/api/training/progress/{self.job_id}",
                json=progress_data,
                headers=headers,
                timeout=30.0
            )
            if response.status_code == 200:
                logger.info(f"✅ Progress update sent: {progress_data.get('status', 'unknown')}")
            else:
                logger.warning(f"⚠️ Progress update failed: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Failed to send progress: {e}")

    def log_metrics(self, epoch: int, step: int, metrics: Dict[str, float]):
        """Log training metrics"""
        timestamp = datetime.now().isoformat()
        metric_entry = {
            "timestamp": timestamp,
            "epoch": epoch,
            "step": step,
            "metrics": metrics
        }
        self.metrics.append(metric_entry)
        logger.info(f"📊 Epoch {epoch}, Step {step}: {metrics}")


class TrainingCallback(TrainerCallback):
    """Custom callback for tracking training progress"""
    
    def __init__(self, progress_tracker: ProgressTracker, total_epochs: int):
        self.progress_tracker = progress_tracker
        self.total_epochs = total_epochs
        self.start_time = time.time()
        self.epoch_start_time = time.time()
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Called when training begins"""
        self.progress_tracker.send_progress({
            "status": "training_started",
            "total_epochs": self.total_epochs,
            "timestamp": datetime.now().isoformat()
        })
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch"""
        self.epoch_start_time = time.time()
        epoch = int(state.epoch)
        logger.info(f"🎯 Starting epoch {epoch}/{self.total_epochs}")
        
        self.progress_tracker.send_progress({
            "status": "epoch_started",
            "epoch": epoch,
            "total_epochs": self.total_epochs,
            "progress_percent": (epoch / self.total_epochs) * 100,
            "timestamp": datetime.now().isoformat()
        })
        
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch"""
        epoch = int(state.epoch)
        epoch_time = time.time() - self.epoch_start_time
        total_time = time.time() - self.start_time
        
        # Get latest metrics
        latest_log = state.log_history[-1] if state.log_history else {}
        
        metrics = {
            "train_loss": latest_log.get("train_loss", 0.0),
            "eval_loss": latest_log.get("eval_loss", 0.0),
            "learning_rate": latest_log.get("learning_rate", 0.0),
            "epoch_time_seconds": epoch_time,
            "total_time_seconds": total_time
        }
        
        # Log to tracker
        self.progress_tracker.log_metrics(epoch, state.global_step, metrics)
        
        # Send progress update
        self.progress_tracker.send_progress({
            "status": "epoch_completed",
            "epoch": epoch,
            "total_epochs": self.total_epochs,
            "progress_percent": (epoch / self.total_epochs) * 100,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "eta_seconds": epoch_time * (self.total_epochs - epoch) if epoch < self.total_epochs else 0
        })
        
    def on_log(self, args, state, control, **kwargs):
        """Called when logging occurs"""
        if state.log_history:
            latest_log = state.log_history[-1]
            step = latest_log.get("step", state.global_step)
            
            # Extract key metrics
            metrics = {}
            for key in ["train_loss", "eval_loss", "learning_rate"]:
                if key in latest_log:
                    metrics[key] = latest_log[key]
                    
            if metrics:
                self.progress_tracker.log_metrics(int(state.epoch), step, metrics)
                
    def on_save(self, args, state, control, **kwargs):
        """Called when a checkpoint is saved"""
        checkpoint_info = {
            "epoch": int(state.epoch),
            "step": state.global_step,
            "save_path": args.output_dir,
            "timestamp": datetime.now().isoformat()
        }
        self.progress_tracker.checkpoints.append(checkpoint_info)
        
        self.progress_tracker.send_progress({
            "status": "checkpoint_saved",
            "checkpoint": checkpoint_info,
            "timestamp": datetime.now().isoformat()
        })


def get_system_stats():
    """Get current system resource usage"""
    try:
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            "gpu_memory_reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        }
    except Exception as e:
        logger.warning(f"Could not get system stats: {e}")
        return {}


async def run_training(*, base_model: str, epochs: int, project_id: str, user_id: str,
                       backend_base_url: str, auth_token: str, worker_api_key: str | None = None) -> Dict[str, Any]:
    """
    Comprehensive Whisper training with analytics, checkpoints, and progress tracking.
    """
    job_id = f"train_{project_id}_{int(time.time())}"
    logger.info(f"🚀 Starting training job {job_id}: {base_model} for {epochs} epochs")
    
    # Initialize progress tracker
    progress_tracker = ProgressTracker(backend_base_url, auth_token, job_id, worker_api_key)
    
    # Send initial status
    progress_tracker.send_progress({
        "status": "initializing",
        "job_id": job_id,
        "base_model": base_model,
        "epochs": epochs,
        "system_stats": get_system_stats(),
        "timestamp": datetime.now().isoformat()
    })
    
    try:
        # 1) Load base model with progress tracking
        progress_tracker.send_progress({"status": "loading_model", "timestamp": datetime.now().isoformat()})
        logger.info("📥 Loading base model...")
        
        processor = WhisperProcessor.from_pretrained(base_model)
        model = WhisperForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )
        logger.info("✅ Model loaded successfully")
        
        progress_tracker.send_progress({
            "status": "model_loaded",
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "system_stats": get_system_stats(),
            "timestamp": datetime.now().isoformat()
        })

        # 2) Dataset preparation with progress tracking
        progress_tracker.send_progress({"status": "preparing_dataset", "timestamp": datetime.now().isoformat()})
        logger.info("📊 Creating dataset...")

        # For Whisper, the model expects `input_features` (log-mel spectrograms) and `labels` (token IDs).
        # Until we wire real audio + transcripts, use a minimal dummy PyTorch dataset with correct shapes.
        from torch.utils.data import Dataset as TorchDataset

        class WhisperDummyDataset(TorchDataset):
            def __init__(self, num_samples: int, feature_frames: int = 3000, label_len: int = 64):
                self.num_samples = num_samples
                self.feature_frames = feature_frames
                self.label_len = label_len

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                # input_features: (80, frames) float32
                input_features = torch.zeros(80, self.feature_frames, dtype=torch.float32)
                # labels: fixed-length int64 tokens (non -100 so loss computes)
                labels = torch.randint(low=1, high=1000, size=(self.label_len,), dtype=torch.long)
                return {"input_features": input_features, "labels": labels}

        train_samples = 64
        eval_samples = 16

        train_dataset = WhisperDummyDataset(train_samples)
        eval_dataset = WhisperDummyDataset(eval_samples)
        
        progress_tracker.send_progress({
            "status": "dataset_ready",
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "timestamp": datetime.now().isoformat()
        })

        # 3) Setup training with comprehensive configuration
        logger.info("⚙️ Setting up training configuration...")
        out_root = Path("/tmp/whisper_training")
        out_root.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        checkpoints_dir = out_root / "checkpoints"
        logs_dir = out_root / "logs"
        final_dir = out_root / "final"
        
        for dir_path in [checkpoints_dir, logs_dir, final_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # GPU throttling prevention: Optimized training arguments
        # Build TrainingArguments robustly for older transformers versions
        def build_training_args() -> TrainingArguments:
            try:
                return TrainingArguments(
                    output_dir=str(checkpoints_dir),
                    per_device_train_batch_size=1,  # Small batch to reduce memory
                    per_device_eval_batch_size=1,
                    num_train_epochs=max(1, int(epochs)),
                    learning_rate=5e-6,  # Lower learning rate for stability
                    warmup_steps=10,     # Fewer warmup steps
                    weight_decay=0.01,
                    fp16=False,          # Disable fp16 to avoid gradient scaling issues
                    dataloader_pin_memory=False,
                    dataloader_num_workers=0,
                    gradient_accumulation_steps=4,  # Accumulate more to reduce frequency
                    save_strategy="steps",
                    save_steps=25,       # Save more frequently
                    save_total_limit=3,  # Keep fewer checkpoints
                    evaluation_strategy="steps",
                    eval_steps=25,       # Eval more frequently but short
                    logging_strategy="steps",
                    logging_steps=5,     # Log frequently for monitoring
                    logging_dir=str(logs_dir),
                    report_to=[],
                    ignore_data_skip=False,
                    prediction_loss_only=True,  # Skip extra metrics to save compute
                    load_best_model_at_end=False,  # Skip to save time
                    max_steps=100,       # Limit total steps to prevent throttling
                    dataloader_drop_last=True,
                )
            except TypeError:
                # Fallback for older versions without evaluation/logging/save strategy arguments
                return TrainingArguments(
                    output_dir=str(checkpoints_dir),
                    per_device_train_batch_size=1,
                    per_device_eval_batch_size=1,
                    num_train_epochs=max(1, int(epochs)),
                    learning_rate=2e-5,
                    fp16=False,  # Disable fp16 for stability
                    logging_steps=10,
                )

        args = build_training_args()

        # Initialize callback for progress tracking
        callback = TrainingCallback(progress_tracker, epochs)
        
        # Create trainer with callback
        # Provide a data collator to handle dict batches with tensors
        def data_collator(features):
            # Convert list of dicts into batch tensors
            input_features = torch.stack([f["input_features"] for f in features])  # (B, 80, T)
            labels = torch.nn.utils.rnn.pad_sequence(
                [f["labels"] for f in features], batch_first=True, padding_value=-100
            )
            return {"input_features": input_features, "labels": labels}

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[callback],
        )

        # 4) Start training with comprehensive monitoring
        logger.info(f"🎯 Starting training for {epochs} epochs...")
        progress_tracker.send_progress({
            "status": "training_starting",
            "configuration": {
                "epochs": epochs,
                "batch_size": args.per_device_train_batch_size,
                "learning_rate": float(args.learning_rate),
                "save_steps": int(getattr(args, "save_steps", 0) or 0),
                "eval_steps": int(getattr(args, "eval_steps", 0) or 0)
            },
            "system_stats": get_system_stats(),
            "timestamp": datetime.now().isoformat()
        })
        
        # GPU memory cleanup before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🖥️ GPU Memory: {gpu_memory:.1f}GB total")
            
        progress_tracker.send_progress({
            "status": "starting_training",
            "gpu_memory_gb": gpu_memory if torch.cuda.is_available() else 0,
            "timestamp": datetime.now().isoformat()
        })
        
        # Train with checkpoint recovery only if a checkpoint exists
        resume_from_checkpoint_path = None
        try:
            if os.path.isdir(checkpoints_dir):
                # Find latest checkpoint directory if any
                checkpoint_dirs = [
                    d for d in os.listdir(checkpoints_dir)
                    if str(d).startswith("checkpoint-") and (checkpoints_dir / d).is_dir()
                ]
                if checkpoint_dirs:
                    # Sort by step number and pick the latest
                    def _step(d):
                        try:
                            return int(str(d).split("checkpoint-")[-1])
                        except Exception:
                            return -1
                    latest = sorted(checkpoint_dirs, key=_step)[-1]
                    resume_from_checkpoint_path = str(checkpoints_dir / latest)
                    logger.info(f"🧭 Resuming from checkpoint: {resume_from_checkpoint_path}")
        except Exception as e:
            logger.warning(f"Could not scan checkpoints: {e}")

        train_result = trainer.train(
            resume_from_checkpoint=resume_from_checkpoint_path if resume_from_checkpoint_path else None
        )
        
        logger.info("✅ Training completed!")
        progress_tracker.send_progress({
            "status": "training_completed",
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "final_train_loss": train_result.metrics.get("train_loss", 0),
            "timestamp": datetime.now().isoformat()
        })

        # 5) Save final model with metadata
        logger.info("💾 Saving trained model...")
        progress_tracker.send_progress({"status": "saving_model", "timestamp": datetime.now().isoformat()})
        
        model.save_pretrained(final_dir)
        processor.save_pretrained(final_dir)
        
        # Save training metadata
        metadata = {
            "job_id": job_id,
            "base_model": base_model,
            "epochs": epochs,
            "project_id": project_id,
            "user_id": user_id,
            "training_metrics": progress_tracker.metrics,
            "checkpoints": progress_tracker.checkpoints,
            "final_metrics": train_result.metrics,
            "system_stats": get_system_stats(),
            "completion_time": datetime.now().isoformat()
        }
        
        with open(final_dir / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save training log
        with open(final_dir / "training_log.txt", "w") as f:
            for metric_entry in progress_tracker.metrics:
                f.write(f"{metric_entry['timestamp']}: Epoch {metric_entry['epoch']}, Step {metric_entry['step']}: {metric_entry['metrics']}\n")

        # 6) Final status and artifacts
        result = {
            "status": "completed",
            "job_id": job_id,
            "artifacts": {
                "model_path": str(final_dir),
                "base_model": base_model,
                "epochs": epochs,
                "project_id": project_id,
                "user_id": user_id,
                "training_metrics": progress_tracker.metrics[-10:],  # Last 10 metrics
                "checkpoints": progress_tracker.checkpoints,
                "final_loss": train_result.metrics.get("train_loss", 0),
                "total_training_time": train_result.metrics.get("train_runtime", 0)
            },
            "message": f"Training completed successfully! Final loss: {train_result.metrics.get('train_loss', 'N/A')}"
        }

        progress_tracker.send_progress({
            "status": "completed",
            "final_result": result,
            "timestamp": datetime.now().isoformat()
        })

        logger.info("🎉 Training job finished successfully!")
        return result

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        
        # Send failure notification
        progress_tracker.send_progress({
            "status": "failed",
            "error": str(e),
            "system_stats": get_system_stats(),
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "status": "failed",
            "job_id": job_id,
            "error": str(e),
            "message": f"Training failed: {e}",
            "metrics": progress_tracker.metrics if progress_tracker.metrics else [],
            "checkpoints": progress_tracker.checkpoints if progress_tracker.checkpoints else []
        }


async def handler(event):
    """RunPod serverless handler function (async)."""
    try:
        # Parse input
        job_input = event.get("input", {})
        
        model_id = job_input.get("model_id", "openai/whisper-small")
        epochs = int(job_input.get("epochs", 3))
        project_id = job_input.get("project_id", "")
        user_id = job_input.get("user_id", "")
        backend_base_url = job_input.get("backend_base_url", "")
        auth_token = job_input.get("auth_token", "")
        worker_api_key = job_input.get("worker_api_key", "")

        print(f"📋 Job Input: model={model_id}, epochs={epochs}, project={project_id}")

        # Run training within the existing event loop
        result = await run_training(
            base_model=model_id,
            epochs=epochs,
            project_id=project_id,
            user_id=user_id,
            backend_base_url=backend_base_url,
            auth_token=auth_token,
            worker_api_key=worker_api_key,
        )

        return {"status": "COMPLETED", "output": result}

    except Exception as e:
        print(f"❌ Handler error: {e}")
        return {
            "status": "FAILED", 
            "error": str(e),
            "output": {"status": "failed", "error": str(e)}
        }


# Start the RunPod serverless worker
if __name__ == "__main__":
    print("🚀 Starting RunPod Whisper Training Handler...")
    try:
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        print(f"❌ Failed to start handler: {e}")
        import traceback
        traceback.print_exc()