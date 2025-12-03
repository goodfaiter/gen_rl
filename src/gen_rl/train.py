"""
Run mjlab's train logic from a Python script instead of the terminal.

Usage:
  - Edit `task_id` below to the task you want to train (e.g. "Mjlab-Cartpole").
  - Make sure your task package is installed (pip install -e .) so mjlab discovers it.
  - Run: python run_train_programmatically.py
"""

import os
from pathlib import Path
from dataclasses import replace

# Ensure CUDA selection etc. if needed (optional).
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import the training API from mjlab
from mjlab.scripts.train import TrainConfig, run_train

# Choose the registered task id
task_id = "CREATELAB-Cartpole"  # <-- change to your task id

# Build a TrainConfig from the registered task.
# This uses the same logic that the CLI uses.
cfg = TrainConfig.from_task(task_id)

# If you need to set registry_name (for tracking tasks) or override fields,
# use dataclasses.replace() to create a modified TrainConfig instance.
# Example:
# cfg = replace(cfg, registry_name="my_wandb_artifact:latest")

# Choose a directory for logs/checkpoints (same purpose as the CLI's log dir)
log_dir = Path.cwd() / "runs" / task_id
log_dir.mkdir(parents=True, exist_ok=True)

# Run the training function directly.
# This calls the same code path as the CLI command.
run_train(task_id, cfg, log_dir)