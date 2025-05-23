import os
import torch
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import json
import random
import logging

# Assuming ComfyUI's folder_paths and cli_args are available in the node environment
import folder_paths
from comfy.cli_args import args

# --- Utility function copied from source/shared_utils/text_encoding_utils.py ---
# This is copied here to ensure it's available within the custom node environment
# without complex import path issues.
logger = logging.getLogger(__name__)

def ensure_utf8_friendly(text_input: str) -> str:
    """
    Ensures the input string is UTF-8 friendly by encoding and then
    decoding with error replacement.
    Args:
        text_input: The string to process.
    Returns:
        A UTF-8 friendly version of the string.
    """
    if not isinstance(text_input, str):
        logger.warning(f"ensure_utf8_friendly received non-string input: {type(text_input)}. Converting to string.")
        text_input = str(text_input)
    try:
        # Encode to bytes using UTF-8, replacing errors, then decode back to string
        return text_input.encode('utf-8', errors='replace').decode('utf-8')
    except Exception as e:
        logger.error(f"Error during UTF-8 conversion for input '{text_input[:100]}...': {e}", exc_info=True)
        # Fallback: return original string if conversion fails catastrophically (should be rare with 'replace')
        return text_input
# --- End of Utility function ---


class SaveImageEnhancedNode:
    def __init__(self):
        # Default output directory relative to ComfyUI's output
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        # Use a random prefix for temporary files if needed, but not for final output
        # self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 4 # Default PNG compression level

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI_DN_%date:yyyy-MM-dd%", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes. Supports %batch_num%."}),
                "output_folder": ("STRING", {"default": "output", "tooltip": "The folder to save the images to. Can be relative to ComfyUI's output or an absolute path."}),
                "add_counter_suffix": ("BOOLEAN", {"default": True, "tooltip": "If True, adds an incrementing numerical suffix (_00001_) to the filename to prevent overwriting."}),
            },
            "optional": {
                "caption_file_extension": ("STRING", {"default": ".txt", "tooltip": "The extension for the caption file."}),
                "caption": ("STRING", {"forceInput": True, "tooltip": "string to save as .txt file"}),
            },
            "hidden": {
                "prompt": "PROMPT", # Used for saving metadata in PNG
                "extra_pnginfo": "EXTRA_PNGINFO" # Used for saving metadata in PNG
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("last_filename_saved",)
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "👽 Divergent Nodes/Image" # New category for Divergent Nodes
    DESCRIPTION = "Saves the input images to a specified directory with optional caption and filename counter."

    def save_images(self, images, output_folder, filename_prefix="ComfyUI_DN_%date:yyyy-MM-dd%", add_counter_suffix=True, prompt=None, extra_pnginfo=None, caption=None, caption_file_extension=".txt"):

        # Resolve the full output folder path
        if os.path.isabs(output_folder):
            full_output_folder = output_folder
        else:
            # Assume relative to ComfyUI's output directory
            self.output_dir = folder_paths.get_output_directory()
            full_output_folder = os.path.join(self.output_dir, output_folder)

        # Ensure the output directory exists
        if not os.path.exists(full_output_folder):
            os.makedirs(full_output_folder, exist_ok=True)

        results = list()
        last_saved_filename = ""

        # Use ComfyUI's path helper to get base filename and initial counter
        # We will modify the filename later based on add_counter_suffix
        # The counter returned here is based on existing files matching the prefix pattern
        # Strip leading/trailing whitespace from filename_prefix
        cleaned_filename_prefix = filename_prefix.strip()
        _, base_filename_without_counter, initial_counter, subfolder, _ = folder_paths.get_save_image_path(
            cleaned_filename_prefix, full_output_folder, images[0].shape[1], images[0].shape[0]
        )

        current_counter = initial_counter

        for batch_number, image in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    # Ensure prompt metadata is UTF-8 friendly
                    metadata.add_text("prompt", ensure_utf8_friendly(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                         # Ensure extra_pnginfo metadata is UTF-8 friendly
                        metadata.add_text(x, ensure_utf8_friendly(extra_pnginfo[x]))

            # Construct the filename based on prefix, batch number, and optional counter
            # Replace %batch_num% placeholder
            filename_part = base_filename_without_counter.replace("%batch_num%", str(batch_number))

            if add_counter_suffix:
                # Include the counter suffix
                file = f"{filename_part}_{current_counter:05}.png"
            else:
                # Omit the counter suffix
                file = f"{filename_part}.png"

            file_path = os.path.join(full_output_folder, file)

            # Save the image
            img.save(file_path, pnginfo=metadata, compress_level=self.compress_level)

            # Save the caption if provided
            if caption is not None:
                # Construct caption filename
                if add_counter_suffix:
                     # Include the counter suffix in caption filename
                    txt_file = f"{filename_part}_{current_counter:05}{caption_file_extension}"
                else:
                    # Omit the counter suffix in caption filename
                    txt_file = f"{filename_part}{caption_file_extension}"

                txt_file_path = os.path.join(full_output_folder, txt_file)

                # Save the caption with UTF-8 encoding and sanitization
                with open(txt_file_path, 'w', encoding='utf-8') as f:
                    f.write(ensure_utf8_friendly(caption))

            results.append({
                "filename": file,
                "subfolder": subfolder, # subfolder is relative to ComfyUI's output_dir
                "type": self.type
            })
            last_saved_filename = file_path # Store the full path of the last saved file

            # Increment counter only if suffix is added, otherwise it's effectively static per prefix+batch_num
            if add_counter_suffix:
                current_counter += 1

        # The UI expects a list of results, but the node output is a single string
        # We return the full path of the last saved file as the node output
        return (last_saved_filename,)

# Node mappings are now in __init__.py
