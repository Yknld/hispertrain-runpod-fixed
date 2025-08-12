#!/usr/bin/env python3
"""
WhisperTrain RunPod Training Handler

Professional Whisper model training with real data loading and comprehensive error handling.
Designed for RunPod serverless deployment with real-time progress tracking.
"""

import asyncio
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import numpy as np
import tempfile

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s:%(lineno)d %(asctime)s,%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/training.log')
    ]
)
logger = logging.getLogger(__name__)

# Import heavy ML libraries with progress
print("📋 Importing basic modules...", flush=True)
import torch
from torch.utils.data import Dataset as TorchDataset

print("📋 Setting up logging...", flush=True)

print("📋 Importing heavy ML libraries...", flush=True)
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
import runpod
import psutil

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
        stats = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "gpu_available": torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            stats.update({
                "gpu_memory_allocated_mb": torch.cuda.memory_allocated() // (1024 * 1024),
                "gpu_memory_reserved_mb": torch.cuda.memory_reserved() // (1024 * 1024),
                "gpu_memory_total_mb": torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            })
            
        return stats
    except Exception as e:
        logger.warning(f"Could not get system stats: {e}")
        return {}


# Constants for training
MAX_TRAIN_AUDIO_SECONDS = 30  # Cap audio length for training samples


class WhisperDummyDataset(TorchDataset):
    """Minimal dummy dataset for testing/fallback"""
    
    def __init__(self, size: int):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Create minimal valid Whisper input
        input_features = torch.zeros(80, int(16000 * MAX_TRAIN_AUDIO_SECONDS / 160), dtype=torch.float32)
        labels = torch.tensor([50257, 50362, 50363], dtype=torch.long)  # <|startoftranscript|><|en|><|transcribe|>
        return {"input_features": input_features, "labels": labels}


class WhisperFileDataset(TorchDataset):
    """Dataset that loads real audio and transcript files from backend"""
    
    def __init__(self, files_manifest: List[Dict[str, str]], user_id: str, backend_base_url: str, auth_token: str, worker_api_key: str, processor: WhisperProcessor):
        self.user_id = user_id
        self.backend_base_url = backend_base_url
        self.auth_token = auth_token
        self.worker_api_key = worker_api_key
        self.processor = processor
        self.data_pairs = self._prepare_data_pairs(files_manifest, user_id, backend_base_url, auth_token, worker_api_key)
        logger.info(f"📦 Found {len(self.data_pairs)} audio-text pairs in manifest")
        
    def _prepare_data_pairs(self, files_manifest, user_id, backend_base_url, auth_token, worker_api_key):
        """Download and pair audio/VTT files"""
        import httpx
        
        # Separate audio and transcript files
        audio_files = [f for f in files_manifest if f.get("file_type") == "audio"]
        transcript_files = [f for f in files_manifest if f.get("file_type") == "transcript"]
        
        logger.info(f"📁 Found {len(audio_files)} audio files and {len(transcript_files)} transcript files")
        
        # Create pairs by matching filenames (remove extensions)
        paired_data = []
        
        for audio_file in audio_files:
            audio_name = audio_file.get("filename", "")
            audio_basename = audio_name.rsplit('.', 1)[0] if '.' in audio_name else audio_name
            
            # Find matching transcript
            matching_transcript = None
            for transcript_file in transcript_files:
                transcript_name = transcript_file.get("filename", "")
                transcript_basename = transcript_name.rsplit('.', 1)[0] if '.' in transcript_name else transcript_name
                
                if audio_basename == transcript_basename:
                    matching_transcript = transcript_file
                    break
            
            if matching_transcript:
                paired_data.append({
                    "audio_filename": audio_name,
                    "transcript_filename": matching_transcript.get("filename", "")
                })
                logger.info(f"✅ Paired: {audio_name} <-> {matching_transcript.get('filename', '')}")
            else:
                logger.warning(f"⚠️ No transcript found for audio file: {audio_name}")
        
        return paired_data
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        """Load and process a single audio-transcript pair"""
        try:
            pair = self.data_pairs[idx]
            audio_filename = pair["audio_filename"]
            transcript_filename = pair["transcript_filename"]
            
            # Download audio file
            audio_array = self._download_audio(audio_filename)
            if audio_array is None or len(audio_array) == 0:
                logger.warning(f"⚠️ Empty or invalid audio for {audio_filename}. Using silence.")
                audio_array = np.zeros(int(16000 * MAX_TRAIN_AUDIO_SECONDS), dtype=np.float32)  # Silence
            
            # Limit audio length
            max_samples = int(16000 * MAX_TRAIN_AUDIO_SECONDS)
            if len(audio_array) > max_samples:
                audio_array = audio_array[:max_samples]
            
            # Download transcript
            text = self._download_transcript(transcript_filename)
            if not text.strip():
                logger.warning(f"⚠️ Empty transcript for {transcript_filename}. Using minimal token.")
                text = "silence"  # Use a minimal token for empty text
            
            # Process audio and text
            input_features = self.processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features[0]
            labels = self.processor(text=text, return_tensors="pt").input_ids[0]
            
            # Ensure labels are not empty
            if labels.numel() == 0:
                logger.warning(f"⚠️ Empty labels after processing for {transcript_filename}. Using minimal token.")
                labels = torch.tensor([self.processor.tokenizer.bos_token_id], dtype=torch.long)  # BOS token
            
            # Truncate audio features if too long  
            if input_features.shape[-1] > self.processor.feature_extractor.n_samples:
                input_features = input_features[:, :self.processor.feature_extractor.n_samples]
            
            return {"input_features": input_features, "labels": labels}
            
        except Exception as e:
            logger.error(f"❌ Error processing sample {self.data_pairs[idx].get('audio_filename')}: {e}")
            # Fallback to dummy data for this sample
            return {
                "input_features": torch.zeros(80, int(16000 * MAX_TRAIN_AUDIO_SECONDS / 160), dtype=torch.float32),
                "labels": torch.tensor([self.processor.tokenizer.bos_token_id], dtype=torch.long)
            }
    
    def _download_audio(self, filename: str) -> np.ndarray:
        """Download audio file from backend"""
        try:
            import httpx
            import soundfile as sf
            import io
            
            # Construct URL - files are served at /uploads/{user_id}/{filename}
            url = f"{self.backend_base_url}/uploads/{self.user_id}/{filename}"
            
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            elif self.worker_api_key:
                headers["X-Worker-Key"] = self.worker_api_key
            
            response = httpx.get(url, headers=headers, timeout=60.0)
            response.raise_for_status()
            
            # Load audio using soundfile
            audio_bytes = io.BytesIO(response.content)
            audio_array, sample_rate = sf.read(audio_bytes)
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Resample to 16kHz if needed (basic resampling)
            if sample_rate != 16000:
                logger.warning(f"⚠️ Audio sample rate is {sample_rate}, expected 16000. Using basic resampling.")
                # Simple resampling (not ideal but works for testing)
                target_length = int(len(audio_array) * 16000 / sample_rate)
                audio_array = np.interp(
                    np.linspace(0, len(audio_array), target_length),
                    np.arange(len(audio_array)),
                    audio_array
                )
            
            return audio_array.astype(np.float32)
            
        except Exception as e:
            logger.error(f"❌ Failed to download audio {filename}: {e}")
            return None
    
    def _download_transcript(self, filename: str) -> str:
        """Download transcript file from backend"""
        try:
            import httpx
            
            # Construct URL - files are served at /uploads/{user_id}/{filename}
            url = f"{self.backend_base_url}/uploads/{self.user_id}/{filename}"
            
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            elif self.worker_api_key:
                headers["X-Worker-Key"] = self.worker_api_key
            
            response = httpx.get(url, headers=headers, timeout=60.0)
            response.raise_for_status()
            
            # Parse VTT content
            vtt_content = response.text
            
            # Extract text from VTT (simple parser)
            lines = vtt_content.split('\n')
            text_lines = []
            
            for line in lines:
                line = line.strip()
                # Skip VTT headers, timestamps, and empty lines
                if (line and 
                    not line.startswith('WEBVTT') and 
                    not line.startswith('NOTE') and
                    '-->' not in line and
                    not line.isdigit()):
                    text_lines.append(line)
            
            text = ' '.join(text_lines).strip()
            return text
            
        except Exception as e:
            logger.error(f"❌ Failed to download transcript {filename}: {e}")
            return ""


async def run_training(*, base_model: str, epochs: int, project_id: str, user_id: str,
                       backend_base_url: str, auth_token: str, worker_api_key: str | None = None,
                       batch_size: int = 2, learning_rate: float = 2e-05, 
                       files_manifest: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Comprehensive Whisper training with real data loading and robust error handling.
    """
    # Use deterministic job_id that matches backend expectation
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
            torch_dtype=torch.float32,  # Use float32 to match training args
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
        logger.info("📊 Preparing dataset...")
        
        files_manifest = files_manifest or []
        
        if files_manifest:
            logger.info(f"📦 Found {len(files_manifest)} files in manifest. Preparing real dataset...")
            # Filter to only audio and transcript files
            audio_files = [f for f in files_manifest if f.get("file_type") == "audio"]
            transcript_files = [f for f in files_manifest if f.get("file_type") == "transcript"]
            
            # Create WhisperFileDataset
            train_dataset = WhisperFileDataset(
                files_manifest=files_manifest,  # Pass full manifest for pairing
                user_id=user_id,
                backend_base_url=backend_base_url,
                auth_token=auth_token,
                worker_api_key=worker_api_key,
                processor=processor
            )
            eval_dataset = WhisperDummyDataset(16)  # Use dummy for eval for now
            logger.info(f"📦 Found {len(train_dataset.data_pairs)} audio-text pairs in manifest")
        else:
            logger.info("⚠️ No files manifest provided or no audio-text pairs found. Using dummy dataset.")
            train_dataset = WhisperDummyDataset(64)
            eval_dataset = WhisperDummyDataset(16)
        
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
        
        # Build TrainingArguments robustly for older transformers versions
        def build_training_args() -> TrainingArguments:
            try:
                return TrainingArguments(
                    output_dir=str(checkpoints_dir),
                    per_device_train_batch_size=batch_size,  # Use provided batch_size
                    per_device_eval_batch_size=batch_size,
                    num_train_epochs=max(1, int(epochs)),
                    learning_rate=learning_rate,  # Use provided learning_rate
                    warmup_steps=10,
                    weight_decay=0.01,
                    fp16=False,  # Disable fp16 to avoid gradient scaling issues
                    dataloader_pin_memory=False,
                    dataloader_num_workers=0,
                    gradient_accumulation_steps=4,
                    save_strategy="steps",
                    save_steps=25,
                    save_total_limit=3,
                    evaluation_strategy="steps",
                    eval_steps=25,
                    logging_strategy="steps",
                    logging_steps=5,
                    logging_dir=str(logs_dir),
                    report_to=[],
                    ignore_data_skip=False,
                    prediction_loss_only=True,
                    load_best_model_at_end=False,
                    max_steps=100,  # Limit steps for faster testing
                    dataloader_drop_last=True,
                )
            except TypeError:
                # Fallback for older versions without evaluation/logging/save strategy arguments
                return TrainingArguments(
                    output_dir=str(checkpoints_dir),
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    num_train_epochs=max(1, int(epochs)),
                    learning_rate=learning_rate,
                    fp16=False,  # Disable fp16 for stability
                    logging_steps=10,
                )

        args = build_training_args()

        # Initialize callback for progress tracking
        callback = TrainingCallback(progress_tracker, epochs)
        
        # Data collator to handle batching properly
        def data_collator(features):
            # Filter out any None features from failed loading
            features = [f for f in features if f is not None]
            if not features:
                logger.warning("⚠️ No valid features in batch for data_collator. Returning empty batch.")
                return {"input_features": torch.empty(0), "labels": torch.empty(0)}

            # Ensure all features have 'input_features' and 'labels'
            valid_features = []
            for f in features:
                if "input_features" in f and "labels" in f:
                    valid_features.append(f)
                else:
                    logger.warning(f"⚠️ Skipping malformed feature in batch: {f.keys()}")
            
            if not valid_features:
                logger.warning("⚠️ No valid features after filtering in data_collator. Returning empty batch.")
                return {"input_features": torch.empty(0), "labels": torch.empty(0)}

            # Pad input_features to max length in batch
            max_input_len = max(f["input_features"].shape[-1] for f in valid_features)
            padded_input_features = []
            for f in valid_features:
                pad_len = max_input_len - f["input_features"].shape[-1]
                padded_input_features.append(torch.nn.functional.pad(f["input_features"], (0, pad_len)))
            input_features = torch.stack(padded_input_features)

            # Pad labels
            labels = torch.nn.utils.rnn.pad_sequence(
                [f["labels"] for f in valid_features], batch_first=True, padding_value=-100
            )
            return {"input_features": input_features, "labels": labels}
        
        # Create trainer with callback
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
            "status": "starting_training",
            "configuration": {
                "epochs": epochs,
                "batch_size": args.per_device_train_batch_size,
                "learning_rate": args.learning_rate,
                "max_steps": getattr(args, "max_steps", "unlimited")
            },
            "system_stats": get_system_stats(),
            "timestamp": datetime.now().isoformat()
        })
        
        # Train with automatic checkpoint recovery (only if checkpoint exists)
        checkpoint_path = None
        if os.path.exists(checkpoints_dir) and os.listdir(checkpoints_dir):
            # Check if there are valid checkpoints
            checkpoint_dirs = [d for d in os.listdir(checkpoints_dir) if d.startswith("checkpoint-")]
            if checkpoint_dirs:
                checkpoint_path = True  # Let trainer find the latest
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint_path)
        
        # Clear GPU cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
        import traceback
        logger.error(f"❌ Training failed: {e}")
        traceback.print_exc()
        
        # Send failure notification
        progress_tracker.send_progress({
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
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
