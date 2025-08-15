import os
import torch
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import json
import random
import logging
import re # Import re for regex operations
from typing import Tuple, Dict, Any, Optional, List

# Import ComfyUI types for better type hinting and autocomplete
from comfy.comfy_types import ComfyNodeABC, IO, InputTypeDict

# Assuming ComfyUI's folder_paths and cli_args are available in the node environment
import folder_paths
from comfy.cli_args import args

# Import custom log level
from ..shared_utils.logging_utils import SUCCESS_HIGHLIGHT
from ..shared_utils.text_encoding_utils import ensure_utf8_friendly
from ..shared_utils.console_io import sanitize_filename # Import from shared_utils

# Setup logger for this module
logger = logging.getLogger(__name__)

class SaveImageEnhancedNode(ComfyNodeABC): # Inherit from ComfyNodeABC
    """
    A ComfyUI node for saving images with enhanced options, including
    custom output folders, flexible filename prefixes, and optional caption saving.
    """
    def __init__(self):
        # Default output directory relative to ComfyUI's output
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.compress_level = 4 # Default PNG compression level

    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict: # Use InputTypeDict for type hinting
        return {
            "required": {
                "images": (IO.IMAGE,), # Use IO.IMAGE
                "filename_prefix": (IO.STRING, {"default": "ComfyUI_DN_%date:yyyy-MM-dd%", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes. Supports %batch_num%."}),
                "output_folder": (IO.STRING, {"default": "output", "tooltip": "Subfolder within ComfyUI's output directory, or an absolute path."}),
                "add_counter_suffix": (IO.BOOLEAN, {"default": True, "tooltip": "If True, adds an incrementing numerical suffix (_00001_) to the filename to prevent overwriting."}),
            },
            "optional": {
                "caption_file_extension": (IO.STRING, {"default": ".txt", "tooltip": "The extension for the caption file."}),
                "caption": (IO.STRING, {"forceInput": True, "multiline": True, "default": "", "tooltip": "Optional: String to save as a caption file alongside the image."}),
            },
            "hidden": {
                "prompt": "PROMPT", # Used for saving metadata in PNG
                "extra_pnginfo": "EXTRA_PNGINFO" # Used for saving metadata in PNG
            },
        }

    RETURN_TYPES: Tuple[str] = (IO.STRING,) # Use IO.STRING
    RETURN_NAMES: Tuple[str] = ("last_filename_saved",)
    FUNCTION: str = "save_images" # Use type hint for FUNCTION

    OUTPUT_NODE: bool = True # Use type hint for OUTPUT_NODE

    CATEGORY: str = "Divergent Nodes 👽/Image" # New category for Divergent Nodes
    DESCRIPTION: str = "Saves the input images to a specified directory with optional caption and filename counter."

    def save_images(self, images: torch.Tensor, output_folder: str, filename_prefix: str="ComfyUI_DN_%date:yyyy-MM-dd%", add_counter_suffix: bool=True, prompt: Optional[Dict[str, Any]]=None, extra_pnginfo: Optional[Dict[str, Any]]=None, caption: Optional[str]=None, caption_file_extension: str=".txt") -> Tuple[str]:
        """
        Saves images with enhanced options.
        """
        full_output_folder = self._get_full_output_folder(output_folder)
        os.makedirs(full_output_folder, exist_ok=True)
        logger.info(f"Saving images to: {full_output_folder}")

        last_filename = ""
        
        # Use ComfyUI's path helper to get base filename and initial counter
        # We will modify the filename later based on add_counter_suffix
        # The counter returned here is based on existing files matching the prefix pattern
        # Strip leading/trailing whitespace from filename_prefix
        cleaned_filename_prefix = filename_prefix.strip()
        
        # Ensure images is not empty to avoid index error
        if not images.shape[0] > 0:
            logger.error("No images provided to Save Image Enhanced node.")
            return ("ERROR: No images provided.",)

        _, base_filename_without_counter, initial_counter, subfolder, _ = folder_paths.get_save_image_path(
            cleaned_filename_prefix, full_output_folder, images[0].shape[1], images[0].shape[0]
        )

        current_counter = initial_counter

        for batch_number, image in enumerate(images):
            # Convert tensor to PIL Image
            pil_image = tensor_to_pil(image)

            # Prepare filename
            filename = self._get_image_filename(filename_prefix, current_counter, pil_image, prompt, extra_pnginfo, add_counter_suffix)
            
            # Sanitize filename to prevent OSError
            sanitized_filename = sanitize_filename(filename)
            
            full_path_no_ext = os.path.join(full_output_folder, sanitized_filename)
            full_image_path = f"{full_path_no_ext}.png" # Always save as PNG

            # Save image
            try:
                info = PngInfo() if not args.disable_metadata else None
                if info:
                    if prompt is not None:
                        # Ensure prompt metadata is UTF-8 friendly
                        info.add_text("prompt", json.dumps(ensure_utf8_friendly(prompt)))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            # Ensure all extra_pnginfo values are UTF-8 friendly
                            info.add_text(x, json.dumps(ensure_utf8_friendly(extra_pnginfo[x])))
                
                pil_image.save(full_image_path, pnginfo=info, compress_level=self.compress_level)
                logger.log(SUCCESS_HIGHLIGHT, f"Image saved: {full_image_path}") # Use SUCCESS_HIGHLIGHT
                last_filename = full_image_path
            except Exception as e:
                logger.error(f"Error saving image {full_image_path}: {e}", exc_info=True)
                last_filename = f"ERROR: Could not save image {full_image_path} - {e}"
                continue # Try to save next image

            # Save caption if provided
            if caption and caption_file_extension:
                caption_filename = f"{full_path_no_ext}{caption_file_extension}"
                try:
                    # Ensure caption is UTF-8 friendly
                    safe_caption = ensure_utf8_friendly(caption)
                    with open(caption_filename, "w", encoding="utf-8") as f:
                        f.write(safe_caption)
                    logger.info(f"Caption saved: {caption_filename}")
                except Exception as e:
                    logger.error(f"Error saving caption {caption_filename}: {e}", exc_info=True)

            # Increment counter only if suffix is added
            if add_counter_suffix:
                current_counter += 1

        return (last_filename,)

    def _get_full_output_folder(self, output_folder_input: str) -> str:
        """
        Determines the full path for the output folder.
        If output_folder_input is an absolute path, use it directly.
        Otherwise, treat it as a subfolder relative to ComfyUI's output directory.
        """
        if os.path.isabs(output_folder_input):
            return output_folder_input
        else:
            return os.path.join(self.output_dir, output_folder_input)

    def _get_image_filename(self, filename_prefix: str, index: int, pil_image: Image.Image,
                            prompt: Optional[Dict[str, Any]], extra_pnginfo: Optional[Dict[str, Any]],
                            add_counter_suffix: bool) -> str:
        """
        Generates a dynamic filename based on the prefix, index, image properties, and prompt info.
        """
        # Replace date placeholders
        filename = filename_prefix.replace("%date:yyyy-MM-dd%", time.strftime("%Y-%m-%d"))
        filename = filename.replace("%date:yyyyMMdd_HHmmss%", time.strftime("%Y%m%d_%H%M%S"))

        # Replace image dimension placeholders
        filename = filename.replace("%width%", str(pil_image.width))
        filename = filename.replace("%height%", str(pil_image.height))

        # Replace batch number placeholder
        # Note: %batch_num% is typically 0-indexed for the current batch,
        # while 'index' here is the global counter.
        # If user expects %batch_num% to be 0-indexed for current batch,
        # this needs adjustment. For now, using global 'index'.
        filename = filename.replace("%batch_num%", str(index).zfill(5)) # Pad with zeros

        # Replace placeholders from other nodes (e.g., %Empty Latent Image.width%)
        if prompt:
            for node_id, node_data in prompt.items():
                if isinstance(node_data, dict) and "inputs" in node_data:
                    for input_name, input_value in node_data["inputs"].items():
                        placeholder = f"%{node_id}.{input_name}%"
                        if placeholder in filename:
                            filename = filename.replace(placeholder, str(input_value))
        
        # The counter suffix is now handled by the main loop's `current_counter`
        # and the `add_counter_suffix` logic.
        # This function now just prepares the base filename part.
        return filename
