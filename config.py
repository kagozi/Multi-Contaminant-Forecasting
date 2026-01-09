# ============================================================================
# Module 1: config.py - Configuration and utilities
# ============================================================================

import os
import random
import numpy as np
import torch
from dataclasses import dataclass

@dataclass
class TrainConfig:
    """Training and model configuration"""
    # Data
    contaminants: list = None
    history_len: int = 30
    horizon: int = 7
    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15
    
    # Sparsity
    p_space: float = 0.2
    p_time: float = 0.5
    seed: int = 42
    
    # Model architecture
    d_basin: int = 32
    d_static: int = 32
    d_coord: int = 16
    d_model: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    
    # Training
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Synthetic data
    synth_basins: int = 300
    synth_days: int = 365 * 4
    synth_dyn_features: int = 6
    synth_static_features: int = 8

    def __post_init__(self):
        if self.contaminants is None:
            self.contaminants = ["no3", "phosphorus", "discharge"]


def seed_everything(seed: int):
    """Set all random seeds for reproducibility"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False