"""
Memory estimation and profiling utilities.
"""

import logging
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


def estimate_memory_usage(
    model: torch.nn.Module,
    batch_size: int = 1,
    sequence_length: int = 512,
    dtype: str = "fp16",
    gradient_checkpointing: bool = False,
) -> Dict[str, float]:
    """
    Estimate memory usage for training an expanded model.

    Args:
        model: The model to estimate for
        batch_size: Training batch size
        sequence_length: Input sequence length
        dtype: Data type ('fp32', 'fp16', 'bf16')
        gradient_checkpointing: Whether gradient checkpointing is enabled

    Returns:
        Dictionary with memory estimates in GB
    """
    # Bytes per parameter by dtype
    dtype_bytes = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
    }

    bytes_per_param = dtype_bytes.get(dtype, 2)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Model weights memory
    model_memory = total_params * bytes_per_param / (1024**3)

    # Activations (rough estimate)
    # For a transformer: ~batch_size * seq_len * hidden_dim * num_layers * 4 bytes
    if hasattr(model, "config"):
        hidden_size = getattr(model.config, "hidden_size", 2048)
        num_layers = getattr(model.config, "num_hidden_layers", 24)
    else:
        hidden_size = 2048
        num_layers = 24

    activation_memory = batch_size * sequence_length * hidden_size * num_layers * 4 / (1024**3)

    if gradient_checkpointing:
        activation_memory *= 0.3  # Roughly 70% savings

    # Gradients (same size as trainable params)
    gradient_memory = trainable_params * bytes_per_param / (1024**3)

    # Optimizer states (2x params for Adam)
    optimizer_memory = trainable_params * bytes_per_param * 2 / (1024**3)

    # Total
    total_memory = model_memory + activation_memory + gradient_memory + optimizer_memory

    return {
        "model_weights_gb": round(model_memory, 2),
        "activations_gb": round(activation_memory, 2),
        "gradients_gb": round(gradient_memory, 2),
        "optimizer_states_gb": round(optimizer_memory, 2),
        "total_gb": round(total_memory, 2),
        "recommended_gb": round(total_memory * 1.2, 2),  # 20% buffer
    }


def get_memory_profile(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Get current GPU memory profile.

    Args:
        device: Device to profile (default: current device)

    Returns:
        Memory statistics dictionary
    """
    if not torch.cuda.is_available():
        return {"cuda_available": False}

    if device is None:
        device = torch.cuda.current_device()

    torch.cuda.synchronize(device)

    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
    max_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)

    return {
        "cuda_available": True,
        "device": device,
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "max_allocated_gb": round(max_allocated, 2),
        "max_reserved_gb": round(max_reserved, 2),
    }


def print_memory_profile(device: Optional[torch.device] = None) -> None:
    """Print memory profile to console."""
    profile = get_memory_profile(device)

    if not profile["cuda_available"]:
        print("CUDA not available")
        return

    print("=" * 50)
    print("GPU Memory Profile")
    print("=" * 50)
    print(f"Device: {profile['device']}")
    print(f"Allocated: {profile['allocated_gb']:.2f} GB")
    print(f"Reserved:  {profile['reserved_gb']:.2f} GB")
    print(f"Max Allocated: {profile['max_allocated_gb']:.2f} GB")
    print(f"Max Reserved:  {profile['max_reserved_gb']:.2f} GB")
    print("=" * 50)
