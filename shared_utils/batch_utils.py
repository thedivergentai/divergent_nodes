"""
Batch processing utilities for handling tensors and lists in ComfyUI nodes.
Provides helpers for splitting, combining, and processing batches efficiently.
"""
import torch
from typing import List, Callable, Any, TypeVar
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


def split_tensor_batch(tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    Splits a batched tensor [B, H, W, C] into a list of individual tensors [1, H, W, C].
    
    Args:
        tensor: Input tensor with shape [B, H, W, C]
    
    Returns:
        List of tensors, each with shape [1, H, W, C]
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    
    if tensor.dim() != 4:
        raise ValueError(f"Expected 4D tensor [B, H, W, C], got shape {tensor.shape}")
    
    batch_size = tensor.shape[0]
    return [tensor[i:i+1] for i in range(batch_size)]


def combine_tensor_batch(tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Combines a list of tensors into a single batched tensor.
    
    Args:
        tensors: List of tensors, each with compatible shapes
    
    Returns:
        Combined tensor with batch dimension
    """
    if not tensors:
        raise ValueError("Cannot combine empty list of tensors")
    
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("All items must be torch.Tensor")
    
    # Check shapes are compatible (all dims except batch should match)
    first_shape = tensors[0].shape[1:]
    if not all(t.shape[1:] == first_shape for t in tensors):
        shapes = [t.shape for t in tensors]
        raise ValueError(f"Incompatible tensor shapes for batching: {shapes}")
    
    return torch.cat(tensors, dim=0)


def process_in_batches(
    items: List[T],
    batch_size: int,
    process_fn: Callable[[List[T]], Any],
    show_progress: bool = True
) -> List[Any]:
    """
    Process a large list of items in smaller batches.
    
    Args:
        items: List of items to process
        batch_size: Number of items per batch
        process_fn: Function that processes a batch and returns results
        show_progress: Whether to log progress
    
    Returns:
        List of results from processing each batch
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    
    if not items:
        return []
    
    results = []
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        if show_progress:
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)...")
        
        try:
            batch_results = process_fn(batch)
            results.append(batch_results)
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {e}", exc_info=True)
            raise
    
    return results


def pad_list_to_length(items: List[T], target_length: int, pad_value: T = None) -> List[T]:
    """
    Pad a list to a target length by repeating the last value or using a pad value.
    
    Args:
        items: List to pad
        target_length: Desired length
        pad_value: Value to use for padding (if None, repeats last item)
    
    Returns:
        Padded list
    """
    if not items:
        raise ValueError("Cannot pad empty list")
    
    if len(items) >= target_length:
        return items[:target_length]
    
    if pad_value is None:
        pad_value = items[-1]
    
    return items + [pad_value] * (target_length - len(items))
