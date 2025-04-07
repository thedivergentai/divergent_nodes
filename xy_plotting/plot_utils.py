"""
Utility functions for XY Plot nodes, focusing on setup, validation,
and axis determination logic extracted from node implementations.
"""
import os
import re
import logging
import numpy as np
import torch
import folder_paths
import comfy.utils
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List, Union, Sequence, TypeAlias

# Define type hints reused from the main node
LoadedLoraT: TypeAlias = Optional[Dict[str, torch.Tensor]]

# Setup logger for this utility module
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def validate_lora_path(lora_folder_path: str) -> str:
    """
    Validates the LoRA folder path input, resolving relative paths if possible.

    Args:
        lora_folder_path (str): The input path string.

    Returns:
        str: The validated, absolute path to the LoRA folder.

    Raises:
        ValueError: If the path is invalid or not a directory.
    """
    logger.debug(f"Validating LoRA folder path input: '{lora_folder_path}'")
    clean_path = lora_folder_path.strip('\'" ')
    if os.path.isdir(clean_path):
        return os.path.abspath(clean_path)
    try:
        # Use folder_paths to resolve relative to standard loras folder
        abs_lora_path = folder_paths.get_full_path("loras", clean_path)
        if abs_lora_path and os.path.isdir(abs_lora_path):
            logger.info(f"Resolved relative LoRA path '{clean_path}' to: {abs_lora_path}")
            return abs_lora_path
    except Exception as e:
        logger.warning(f"Error resolving path relative to loras folder: {e}", exc_info=True)
    # Check if it's an absolute path that exists
    if os.path.isabs(clean_path) and os.path.isdir(clean_path):
         return clean_path
    # If all checks fail
    error_msg = f"LoRA folder path is not a valid directory: '{lora_folder_path}' (Checked: '{clean_path}')"
    logger.error(error_msg)
    raise ValueError(error_msg)

def get_lora_files(lora_folder_path: str) -> List[str]:
    """
    Scans the specified directory for valid LoRA filenames.

    Args:
        lora_folder_path (str): The absolute path to the validated LoRA directory.

    Returns:
        List[str]: A sorted list of valid LoRA filenames found.

    Raises:
        ValueError: If the directory cannot be read.
    """
    logger.info(f"Scanning for LoRA files in: {lora_folder_path}")
    try:
        all_files = os.listdir(lora_folder_path)
    except OSError as e:
        logger.error(f"Could not read LoRA folder path: {lora_folder_path}. Check permissions.", exc_info=True)
        raise ValueError(f"Could not read LoRA folder path: {lora_folder_path}. Error: {e}") from e

    valid_extensions = ('.safetensors', '.pt', '.ckpt', '.pth')
    lora_files = sorted([
        f for f in all_files
        if os.path.isfile(os.path.join(lora_folder_path, f)) and f.lower().endswith(valid_extensions)
    ])

    if not lora_files:
        logger.warning(f"No valid LoRA files found in {lora_folder_path}")
    else:
        logger.info(f"Found {len(lora_files)} potential LoRA files.")
        logger.debug(f"LoRA files found: {lora_files}")
    return lora_files

def determine_plot_axes(
    lora_files: List[str],
    x_lora_steps: int,
    y_strength_steps: int,
    max_strength: float
) -> Tuple[List[str], List[float]]:
    """
    Determines the items (LoRAs and strengths) for the plot axes based on step inputs.

    Args:
        lora_files (List[str]): List of available LoRA filenames.
        x_lora_steps (int): Number of LoRAs for X-axis (0=all, 1=last).
        y_strength_steps (int): Number of strength steps for Y-axis.
        max_strength (float): Maximum LoRA strength.

    Returns:
        Tuple[List[str], List[float]]: plot_loras (X-axis), plot_strengths (Y-axis).
    """
    # --- X-Axis (LoRAs) ---
    plot_loras: List[str] = ["No LoRA"] # Baseline column
    num_available_loras: int = len(lora_files)
    logger.debug(f"Determining X-axis: {num_available_loras} available LoRAs, requested steps: {x_lora_steps}")

    if num_available_loras > 0:
        if x_lora_steps == 0: # Use all
            plot_loras.extend(lora_files)
        elif x_lora_steps == 1: # Use last
             plot_loras.append(lora_files[-1])
        elif x_lora_steps > 1: # Sample evenly
            if x_lora_steps >= num_available_loras: # Use all if steps >= available
                 plot_loras.extend(lora_files)
            else:
                indices = np.linspace(0, num_available_loras - 1, num=x_lora_steps, dtype=int)
                unique_indices = sorted(list(set(indices)))
                logger.debug(f"Calculated indices for LoRA selection: {unique_indices}")
                for i in unique_indices:
                    if 0 <= i < num_available_loras:
                        plot_loras.append(lora_files[i])

    # --- Y-Axis (Strengths) ---
    num_strength_points = max(1, y_strength_steps)
    logger.debug(f"Determining Y-axis: {num_strength_points} strength steps up to max {max_strength}")

    if num_strength_points == 1:
        plot_strengths: List[float] = [max_strength]
    else:
        # Evenly spaced points up to max_strength
        plot_strengths = [ (i / num_strength_points) * max_strength for i in range(1, num_strength_points + 1) ]

    logger.info(f"Determined Grid Dimensions: {len(plot_strengths)} rows (Strengths), {len(plot_loras)} columns (LoRAs)")
    logger.debug(f"LoRAs to plot (X-axis): {plot_loras}")
    logger.debug(f"Strengths to plot (Y-axis): {[f'{s:.4f}' for s in plot_strengths]}")
    return plot_loras, plot_strengths

def preload_loras(lora_names: List[str], lora_folder_path: str) -> Dict[str, LoadedLoraT]:
    """
    Loads specified LoRA tensors from disk into memory.

    Args:
        lora_names (List[str]): List of LoRA filenames to load (can include "No LoRA").
        lora_folder_path (str): Validated absolute path to the LoRA directory.

    Returns:
        Dict[str, LoadedLoraT]: Dictionary mapping LoRA name to the loaded tensor
                                or None if loading failed or name is "No LoRA".
    """
    loaded_loras: Dict[str, LoadedLoraT] = {}
    logger.info("Pre-loading LoRA files...")
    for name in lora_names:
        if name == "No LoRA":
            loaded_loras[name] = None # Placeholder for baseline
            continue
        lora_path = os.path.join(lora_folder_path, name)
        if not os.path.exists(lora_path):
            logger.warning(f"  LoRA file not found during pre-load: {lora_path}. Will skip.")
            loaded_loras[name] = None
            continue
        try:
            logger.debug(f"  Loading LoRA: {name}")
            # safe_load=True is important for security
            lora_tensor = comfy.utils.load_torch_file(lora_path, safe_load=True)
            loaded_loras[name] = lora_tensor
            logger.debug(f"  Successfully loaded LoRA: {name}")
        except Exception as e:
            logger.warning(f"  Failed to pre-load LoRA '{name}'. Will skip. Error: {e}", exc_info=True)
            loaded_loras[name] = None # Mark as failed
    logger.info("LoRA pre-loading complete.")
    return loaded_loras

def setup_output_directory(output_folder_name: str) -> Optional[str]:
    """
    Creates the output directory structure for saving individual plot images.

    Args:
        output_folder_name (str): The base name for the output folder (will be sanitized).

    Returns:
        Optional[str]: The absolute path to the created run-specific folder, or None if fails.
    """
    try:
        output_path = folder_paths.get_output_directory()
        # Sanitize folder name
        safe_folder_name = re.sub(r'[\\/*?:"<>|]', '_', output_folder_name).strip()
        if not safe_folder_name:
            safe_folder_name = "XYPlot_Output"
            logger.warning(f"Output folder name invalid, using default: '{safe_folder_name}'")

        base_output_folder = os.path.join(output_path, safe_folder_name)
        # Unique subfolder per run
        run_folder = os.path.join(base_output_folder, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        os.makedirs(run_folder, exist_ok=True)
        logger.info(f"Output directory for individual images prepared: {run_folder}")
        return run_folder
    except Exception as e:
         logger.error(f"Error setting up output directory '{output_folder_name}': {e}", exc_info=True)
         return None
