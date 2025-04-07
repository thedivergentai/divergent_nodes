"""
Node implementation for generating an XY plot comparing LoRA models vs. strength.
"""
import torch
import numpy as np
import os
import comfy.sd
import comfy.utils
import comfy.samplers
import folder_paths
from PIL import Image # Needed for saving individual images
import math # Not used currently, consider removing if unused after refactor
from datetime import datetime
import re # For cleaning paths
import logging
from typing import Dict, Any, Tuple, Optional, List, Union, Sequence, TypeAlias

# Import grid assembly functions
try:
    from .grid_assembly import assemble_image_grid, draw_labels_on_grid
except ImportError:
    # Define dummy functions if import fails, though this indicates a setup issue
    logging.basicConfig(level=logging.INFO) # Ensure basicConfig is called if logger used early
    logging.error("Failed to import grid_assembly functions. Grid generation will fail.")
    def assemble_image_grid(*args: Any, **kwargs: Any) -> torch.Tensor: raise RuntimeError("grid_assembly not found")
    def draw_labels_on_grid(*args: Any, **kwargs: Any) -> torch.Tensor: raise RuntimeError("grid_assembly not found")

# Setup logger for this module
# Use __name__ for hierarchical logging (e.g., 'xy_plotting.lora_strength_plot_node')
logger = logging.getLogger(__name__)
# Ensure handler is configured if running standalone or if root logger isn't set up
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Define type hints for complex ComfyUI types for better readability
# These are placeholders; more specific types might be possible but complex
ComfyConditioningT: TypeAlias = List[Tuple[torch.Tensor, Dict[str, Any]]]
ComfyCLIPObjectT: TypeAlias = Any # Placeholder for the CLIP object structure
ComfyVAEObjectT: TypeAlias = Any # Placeholder for the VAE object structure
ComfyModelObjectT: TypeAlias = Any # Placeholder for the ModelPatcher object structure
ComfyLatentT: TypeAlias = Dict[str, torch.Tensor] # Standard latent format { "samples": tensor }

class LoraStrengthXYPlot:
    """
    Generates an XY plot grid comparing different LoRAs (X-axis)
    against varying model strengths (Y-axis).

    Iterates through selected LoRAs and strength values, generates an image
    for each combination using a base workflow, and assembles the results
    into a single grid image with optional labels. This node helps visualize
    the impact of different LoRAs and their strengths on the final image.
    """
    CATEGORY = "👽 Divergent Nodes/XY Plots"
    OUTPUT_NODE = True # This node produces a final output image

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """
        Defines the input types for the ComfyUI node interface.

        Dynamically fetches available checkpoints, samplers, and schedulers.
        Provides inputs for model selection, LoRA path, conditioning, latent,
        sampling parameters, grid dimensions, and output options.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary defining required and optional inputs.
        """
        # --- Dynamic Lists ---
        try:
            checkpoint_list = folder_paths.get_filename_list("checkpoints")
            if not checkpoint_list:
                logger.warning("No checkpoints found in checkpoints directory.")
                checkpoint_list = ["ERROR: No Checkpoints Found"]
        except Exception as e:
            logger.error(f"Failed to get checkpoint list: {e}", exc_info=True)
            checkpoint_list = ["ERROR: Failed to Load"]

        try:
            sampler_list = comfy.samplers.KSampler.SAMPLERS
            if not sampler_list: sampler_list = ["ERROR: No Samplers Found"]
        except Exception as e:
            logger.error(f"Failed to get sampler list: {e}", exc_info=True)
            sampler_list = ["ERROR: Failed to Load"]

        try:
            scheduler_list = comfy.samplers.KSampler.SCHEDULERS
            if not scheduler_list: scheduler_list = ["ERROR: No Schedulers Found"]
        except Exception as e:
            logger.error(f"Failed to get scheduler list: {e}", exc_info=True)
            scheduler_list = ["ERROR: Failed to Load"]

        # --- Input Definitions ---
        return {
            "required": {
                "checkpoint_name": (checkpoint_list, {"tooltip": "Base model checkpoint to use."}),
                "lora_folder_path": ("STRING", {"default": "loras/", "multiline": False, "tooltip": "Path to the folder containing LoRA files (relative to ComfyUI/models/loras or absolute)."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning."}),
                "latent_image": ("LATENT", {"tooltip": "Initial latent image."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Base seed for generation. Each grid image increments this."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Number of sampling steps."}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "CFG scale."}),
                "sampler_name": (sampler_list, {"tooltip": "Sampler to use."}),
                "scheduler": (scheduler_list, {"tooltip": "Scheduler to use."}),
                "x_lora_steps": ("INT", {"default": 3, "min": 0, "max": 100, "tooltip": "Number of LoRAs for X-axis (0=all found, 1=last found)."}),
                "y_strength_steps": ("INT", {"default": 3, "min": 1, "max": 100, "tooltip": "Number of strength steps for Y-axis."}),
                "max_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "Maximum LoRA strength for Y-axis."}),
            },
            "optional": {
                "opt_clip": ("CLIP", {"tooltip": "Optional override CLIP model."}),
                "opt_vae": ("VAE", {"tooltip": "Optional override VAE model."}),
                "save_individual_images": ("BOOLEAN", {"default": False, "tooltip": "Save each generated grid cell image individually."}),
                "output_folder_name": ("STRING", {"default": "XYPlot_LoRA-Strength", "multiline": False, "tooltip": "Subfolder name within ComfyUI output directory for saved images."}),
                "row_gap": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1, "tooltip": "Gap between rows in the final grid (pixels)."}),
                "col_gap": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1, "tooltip": "Gap between columns in the final grid (pixels)."}),
                "draw_labels": ("BOOLEAN", {"default": True, "tooltip": "Draw LoRA names and strength values on the grid."}),
                "x_axis_label": ("STRING", {"default": "LoRA", "multiline": False, "tooltip": "Optional overall label for the X-axis."}),
                "y_axis_label": ("STRING", {"default": "Strength", "multiline": False, "tooltip": "Optional overall label for the Y-axis."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("xy_plot_image",)
    FUNCTION = "generate_plot"

    # --------------------------------------------------------------------------
    # Helper Methods for Setup and Validation
    # --------------------------------------------------------------------------
    def _validate_inputs(self, lora_folder_path: str) -> str:
        """
        Validates the LoRA folder path input, resolving relative paths if possible.

        Strips surrounding quotes/spaces, checks if it's a directory, tries to
        resolve relative to the standard 'loras' folder, and finally checks
        if it's an existing absolute path.

        Args:
            lora_folder_path (str): The input path string from the node interface.

        Returns:
            str: The validated, absolute path to the LoRA folder.

        Raises:
            ValueError: If the path is invalid or not a directory after checks.
        """
        logger.debug(f"Validating LoRA folder path input: '{lora_folder_path}'")
        clean_path = lora_folder_path.strip('\'" ')

        # 1. Check if the cleaned path is already a valid directory
        if os.path.isdir(clean_path):
            logger.debug(f"Path '{clean_path}' is already a valid directory.")
            # Ensure it's absolute for consistency, though isdir implies it likely is if not empty
            return os.path.abspath(clean_path)

        # 2. Try resolving relative to the standard 'loras' folder
        try:
            abs_lora_path = folder_paths.get_full_path("loras", clean_path)
            if abs_lora_path and os.path.isdir(abs_lora_path):
                logger.info(f"Resolved relative LoRA path '{clean_path}' to: {abs_lora_path}")
                return abs_lora_path
            else:
                logger.debug(f"Path '{clean_path}' not found relative to loras directory.")
        except Exception as e:
            logger.warning(f"Error resolving path relative to loras folder: {e}", exc_info=True)

        # 3. Check if it's an absolute path that exists (redundant if #1 passed, but safe)
        if os.path.isabs(clean_path) and os.path.isdir(clean_path):
             logger.debug(f"Path '{clean_path}' is an existing absolute directory.")
             return clean_path # Already absolute

        # 4. If all checks fail, raise an error
        error_msg = f"LoRA folder path is not a valid directory: '{lora_folder_path}' (Checked: '{clean_path}')"
        logger.error(error_msg)
        raise ValueError(error_msg)

    def _load_base_models(self,
                          checkpoint_name: str,
                          opt_clip: Optional[ComfyCLIPObjectT],
                          opt_vae: Optional[ComfyVAEObjectT]
                          ) -> Tuple[ComfyModelObjectT, ComfyCLIPObjectT, ComfyVAEObjectT]:
        """
        Loads the base checkpoint model, CLIP, and VAE.

        Handles optional overrides for CLIP and VAE provided via node inputs.
        Uses comfy.sd.load_checkpoint_guess_config for robust loading.

        Args:
            checkpoint_name (str): Name of the checkpoint file (e.g., "model.safetensors").
            opt_clip (Optional[ComfyCLIPObjectT]): Optional override CLIP model object.
            opt_vae (Optional[ComfyVAEObjectT]): Optional override VAE model object.

        Returns:
            Tuple[ComfyModelObjectT, ComfyCLIPObjectT, ComfyVAEObjectT]: Loaded model, CLIP, and VAE.

        Raises:
            FileNotFoundError: If the specified checkpoint file cannot be found.
            ValueError: If the final CLIP or VAE object is None after loading/overriding.
            Exception: For other unexpected errors during model loading.
        """
        logger.info(f"Loading base checkpoint: {checkpoint_name}")
        try:
            ckpt_path = folder_paths.get_full_path("checkpoints", checkpoint_name)
            if not ckpt_path or not os.path.exists(ckpt_path):
                logger.error(f"Checkpoint file not found at resolved path for: {checkpoint_name}")
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_name}")

            logger.debug(f"Loading from checkpoint path: {ckpt_path}")
            # Use guess_config for robustness against different checkpoint formats
            loaded_model, loaded_clip, loaded_vae = comfy.sd.load_checkpoint_guess_config(
                ckpt_path, output_vae=True, output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings")
            )

            # Apply overrides if provided
            model: ComfyModelObjectT = loaded_model
            clip: ComfyCLIPObjectT = opt_clip if opt_clip is not None else loaded_clip
            vae: ComfyVAEObjectT = opt_vae if opt_vae is not None else loaded_vae

            # Validate final models
            if model is None:
                 # This case should be unlikely if load_checkpoint_guess_config succeeds
                 raise ValueError("Base model failed to load from checkpoint.")
            if clip is None:
                raise ValueError("CLIP model could not be loaded from checkpoint or provided via input.")
            if vae is None:
                raise ValueError("VAE model could not be loaded from checkpoint or provided via input.")

            logger.info("Base models (Model, CLIP, VAE) loaded successfully.")
            return model, clip, vae

        except FileNotFoundError: # Re-raise specifically
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading base models for checkpoint '{checkpoint_name}': {e}", exc_info=True)
            # Re-raise to halt execution, providing context
            raise RuntimeError(f"Failed to load base models from checkpoint '{checkpoint_name}'. See logs for details.") from e

    def _get_lora_files(self, lora_folder_path: str) -> List[str]:
        """
        Scans the specified directory for valid LoRA filenames (.safetensors, .pt, .ckpt, .pth).

        Args:
            lora_folder_path (str): The absolute path to the validated LoRA directory.

        Returns:
            List[str]: A sorted list of valid LoRA filenames found. Returns empty list if none found.

        Raises:
            ValueError: If the directory cannot be read due to permissions or other OS errors.
        """
        logger.info(f"Scanning for LoRA files in: {lora_folder_path}")
        try:
            all_files = os.listdir(lora_folder_path)
        except OSError as e:
            logger.error(f"Could not read LoRA folder path: {lora_folder_path}. Check permissions.", exc_info=True)
            # Raise a more specific error for clarity
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

    def _determine_plot_axes(self,
                             lora_files: List[str],
                             x_lora_steps: int,
                             y_strength_steps: int,
                             max_strength: float
                             ) -> Tuple[List[str], List[float]]:
        """
        Determines the items (LoRAs and strengths) for the plot axes based on step inputs.

        Handles selection of LoRAs for the X-axis (all, last, or sampled) and
        calculates evenly spaced strength values for the Y-axis.

        Args:
            lora_files (List[str]): List of available LoRA filenames from _get_lora_files.
            x_lora_steps (int): Number of LoRAs to select for the X-axis (0 means all).
            y_strength_steps (int): Number of strength steps for the Y-axis (min 1).
            max_strength (float): The maximum strength value for the Y-axis.

        Returns:
            Tuple[List[str], List[float]]:
            - plot_loras: List of LoRA names (including "No LoRA") for the X-axis.
            - plot_strengths: List of strength values for the Y-axis.
        """
        # --- X-Axis (LoRAs) ---
        plot_loras: List[str] = ["No LoRA"] # Always include a baseline column without any LoRA
        num_available_loras: int = len(lora_files)
        logger.debug(f"Determining X-axis: {num_available_loras} available LoRAs, requested steps: {x_lora_steps}")

        if num_available_loras > 0:
            if x_lora_steps == 0: # Use all found LoRAs
                logger.debug("X-axis: Using all available LoRAs.")
                plot_loras.extend(lora_files)
            elif x_lora_steps == 1: # Use only the last LoRA found (alphabetically)
                 logger.debug("X-axis: Using only the last available LoRA.")
                 plot_loras.append(lora_files[-1]) # Use last element from sorted list
            elif x_lora_steps > 1: # Sample LoRAs evenly
                if x_lora_steps >= num_available_loras: # If steps >= files, use all
                     logger.debug(f"X-axis: Requested steps ({x_lora_steps}) >= available ({num_available_loras}), using all.")
                     plot_loras.extend(lora_files)
                else:
                    # Calculate indices for even spread, ensuring first and last are included if possible
                    # We want x_lora_steps points *total* on the axis *after* "No LoRA"
                    num_loras_to_select = x_lora_steps
                    logger.debug(f"X-axis: Sampling {num_loras_to_select} LoRAs evenly.")
                    # np.linspace generates evenly spaced points including endpoints
                    indices = np.linspace(0, num_available_loras - 1, num=num_loras_to_select, dtype=int)
                    # Ensure uniqueness and sort (linspace should already be sorted)
                    unique_indices = sorted(list(set(indices)))
                    logger.debug(f"Calculated indices for LoRA selection: {unique_indices}")
                    for i in unique_indices:
                        if 0 <= i < num_available_loras: # Double check bounds
                            plot_loras.append(lora_files[i])
                        else:
                             logger.warning(f"Calculated LoRA index {i} out of bounds (0-{num_available_loras-1}). Skipping.")

        # --- Y-Axis (Strengths) ---
        num_strength_points = max(1, y_strength_steps) # Ensure at least one strength value
        logger.debug(f"Determining Y-axis: {num_strength_points} strength steps up to max {max_strength}")

        if num_strength_points == 1:
            plot_strengths: List[float] = [max_strength]
        else:
            # Generate points evenly spaced from (max_strength / num_strength_points) up to max_strength
            # Example: steps=3, max=1.0 -> [0.333, 0.666, 1.0]
            plot_strengths = [ (i / num_strength_points) * max_strength for i in range(1, num_strength_points + 1) ]

        # Final log of determined axes
        logger.info(f"Determined Grid Dimensions: {len(plot_strengths)} rows (Strengths), {len(plot_loras)} columns (LoRAs)")
        logger.debug(f"LoRAs to plot (X-axis): {plot_loras}")
        logger.debug(f"Strengths to plot (Y-axis): {[f'{s:.4f}' for s in plot_strengths]}") # More precision for debug
        return plot_loras, plot_strengths

    def _setup_output_directory(self, output_folder_name: str) -> Optional[str]:
        """
        Creates the output directory structure for saving individual plot images.

        Generates a base folder and a run-specific subfolder with a timestamp.

        Args:
            output_folder_name (str): The base name for the output folder (will be sanitized).

        Returns:
            Optional[str]: The absolute path to the created run-specific folder,
                           or None if creation fails.
        """
        try:
            output_path = folder_paths.get_output_directory()
            # Sanitize the user-provided folder name to remove invalid characters
            safe_folder_name = re.sub(r'[\\/*?:"<>|]', '_', output_folder_name)
            safe_folder_name = safe_folder_name.strip() # Remove leading/trailing whitespace
            if not safe_folder_name: # Handle empty name after sanitization
                safe_folder_name = "XYPlot_Output"
                logger.warning(f"Output folder name was empty or invalid, using default: '{safe_folder_name}'")

            base_output_folder = os.path.join(output_path, safe_folder_name)
            # Create a unique subfolder for each run using a timestamp
            run_folder = os.path.join(base_output_folder, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            os.makedirs(run_folder, exist_ok=True) # Create base and run folder
            logger.info(f"Output directory for individual images prepared: {run_folder}")
            return run_folder
        except OSError as e:
            logger.warning(f"Could not create output directory '{run_folder}'. OS Error: {e}. Disabling saving.", exc_info=True)
            return None
        except Exception as e: # Catch other potential errors like permission issues
             logger.error(f"Unexpected error setting up output directory '{output_folder_name}'.", exc_info=True)
             return None

    # --------------------------------------------------------------------------
    # Core Image Generation Logic - Decomposed Helpers
    # --------------------------------------------------------------------------
    def _apply_lora_to_models(self,
                              model: ComfyModelObjectT,
                              clip: ComfyCLIPObjectT,
                              lora_name: str,
                              strength: float,
                              lora_folder_path: str
                              ) -> Tuple[ComfyModelObjectT, ComfyCLIPObjectT, str]:
        """
        Applies a specific LoRA to cloned model and CLIP objects.

        Args:
            model (ComfyModelObjectT): The base model object (will be cloned).
            clip (ComfyCLIPObjectT): The base CLIP object (will be cloned).
            lora_name (str): Filename of the LoRA to apply.
            strength (float): Strength to apply the LoRA.
            lora_folder_path (str): Validated path to the LoRA directory.

        Returns:
            Tuple[ComfyModelObjectT, ComfyCLIPObjectT, str]:
                - The model with LoRA applied (or original if failed).
                - The CLIP with LoRA applied (or original if failed).
                - A sanitized filename part derived from the LoRA name.
        """
        current_model = model.clone()
        current_clip = clip.clone()
        lora_filename_part = os.path.splitext(lora_name)[0] # Use filename without ext for saving

        lora_path = os.path.join(lora_folder_path, lora_name)
        if not os.path.exists(lora_path):
            logger.warning(f"  LoRA file not found: {lora_path}. Skipping application.")
            return current_model, current_clip, lora_filename_part # Return originals

        try:
            logger.info(f"  Applying LoRA: {lora_name} with strength {strength:.3f}")
            # safe_load=True is important for security
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            current_model, current_clip = comfy.sd.load_lora_for_models(
                current_model, current_clip, lora, strength, strength
            )
            del lora # Free memory immediately after application
            logger.debug(f"  Successfully applied LoRA: {lora_name}")
        except Exception as e:
            # Log clearly but don't halt the whole plot generation for one failed LoRA
            logger.warning(f"  Failed to load or apply LoRA '{lora_name}'. Skipping application. Error: {e}", exc_info=True)
            # Return the original cloned models

        return current_model, current_clip, lora_filename_part

    def _prepare_latent_for_sampling(self,
                                     base_latent: ComfyLatentT,
                                     positive_cond: ComfyConditioningT
                                     ) -> ComfyLatentT:
        """
        Prepares the latent dictionary for sampling.

        Ensures the latent is in the correct format (dict with "samples") and
        handles potential batch size mismatches between latent and conditioning
        by repeating the latent if necessary.

        Args:
            base_latent (ComfyLatentT): The initial latent dictionary.
            positive_cond (ComfyConditioningT): The positive conditioning list.

        Returns:
            ComfyLatentT: The prepared latent dictionary, potentially with repeated samples.

        Raises:
            TypeError: If the base_latent format is invalid.
        """
        current_latent = base_latent.copy() # Work on a copy

        if not isinstance(current_latent, dict) or "samples" not in current_latent:
            msg = f"Invalid latent_image format: {type(base_latent)}. Expected dict with 'samples'."
            logger.error(msg)
            raise TypeError(msg)

        latent_samples = current_latent['samples']
        latent_batch_size = latent_samples.shape[0]

        # Determine conditioning batch size (can be complex, this is a basic check)
        cond_batch_size = 1
        if isinstance(positive_cond, list) and positive_cond:
             # Assuming standard format: List[Tuple[Tensor, Dict]]
             # A more robust check might be needed depending on custom conditioning formats
             cond_batch_size = len(positive_cond)

        logger.debug(f"  Latent batch size: {latent_batch_size}, Conditioning batch size: {cond_batch_size}")

        if latent_batch_size != cond_batch_size:
            if latent_batch_size == 1 and cond_batch_size > 1:
                # Common case: single latent, multiple conds (e.g., from area composition)
                logger.warning(f"  Latent batch (1) != Cond batch ({cond_batch_size}). Repeating latent sample to match.")
                current_latent['samples'] = latent_samples.repeat(cond_batch_size, 1, 1, 1)
            else:
                # Less common, might indicate an issue upstream or require specific handling
                logger.warning(f"  Latent batch ({latent_batch_size}) != Cond batch ({cond_batch_size}). Proceeding, but mismatch might cause sampler errors.")
                # Sampler might handle this, or it might error. No change made here.

        return current_latent

    def _run_sampling_and_decode(self,
                                 model: ComfyModelObjectT,
                                 clip: ComfyCLIPObjectT,
                                 vae: ComfyVAEObjectT,
                                 positive: ComfyConditioningT,
                                 negative: ComfyConditioningT,
                                 latent: ComfyLatentT,
                                 seed: int,
                                 steps: int,
                                 cfg: float,
                                 sampler_name: str,
                                 scheduler: str
                                 ) -> torch.Tensor:
        """
        Performs the core sampling and VAE decoding.

        Args:
            model, clip, vae: The models to use for this step.
            positive, negative: Conditioning tensors.
            latent: Prepared latent dictionary.
            seed, steps, cfg, sampler_name, scheduler: Sampling parameters.

        Returns:
            torch.Tensor: The generated image tensor [B, H, W, C].

        Raises:
            ValueError: If sampler output is missing 'samples'.
            RuntimeError: If sampling or decoding fails unexpectedly.
        """
        logger.debug(f"  Starting sampling: {sampler_name}/{scheduler}, Steps: {steps}, CFG: {cfg}, Seed: {seed}")
        try:
            samples_latent = comfy.sample.sample(
                model, clip, vae, positive, negative, latent,
                seed=seed, steps=steps, cfg=cfg,
                sampler_name=sampler_name, scheduler=scheduler,
                denoise=1.0 # Standard full denoise for XY plot from latent
            )

            if not isinstance(samples_latent, dict) or "samples" not in samples_latent:
                raise ValueError("Sampler output was not a dict or missing 'samples' key.")

            logger.debug("  Sampling complete. Decoding samples...")
            # Decode latent samples to image tensor
            # vae.decode expects [B, C, H_latent, W_latent]
            img_tensor_chw = vae.decode(samples_latent["samples"]) # Output: [B, C, H, W]
            logger.debug(f"  Decoding complete. Output shape: {img_tensor_chw.shape}")

            # Convert to channels-last format [B, H, W, C] for consistency
            img_tensor_bhwc = img_tensor_chw.permute(0, 2, 3, 1)
            return img_tensor_bhwc

        except Exception as e:
            logger.error(f"  Error during sampling or decoding: {e}", exc_info=True)
            raise RuntimeError("Sampling or VAE Decoding failed.") from e


    def _save_image_if_enabled(self,
                               image_tensor_bhwc: torch.Tensor,
                               run_folder: Optional[str],
                               save_individual_images: bool,
                               img_index: int,
                               row_idx: int,
                               col_idx: int,
                               lora_filename_part: str,
                               strength: float):
        """
        Saves the generated image to disk if enabled and path is valid.

        Args:
            image_tensor_bhwc: The generated image tensor [B, H, W, C].
            run_folder: Validated output directory path.
            save_individual_images: Flag indicating if saving is enabled.
            img_index: Overall index for logging.
            row_idx, col_idx: Grid position indices.
            lora_filename_part: Sanitized LoRA name for filename.
            strength: Strength value for filename.
        """
        if not (save_individual_images and run_folder):
            return # Saving disabled or folder setup failed

        try:
            # Assume batch size 1 for saving individual grid cells
            img_tensor_hwc = image_tensor_bhwc[0]
            img_np = img_tensor_hwc.cpu().numpy()
            img_pil = Image.fromarray(np.clip(img_np * 255.0, 0, 255).astype(np.uint8))

            # Sanitize filename part (already done for LoRA, but good practice)
            safe_lora_name = re.sub(r'[\\/*?:"<>|]', '_', lora_filename_part)
            filename = f"row-{row_idx}_col-{col_idx}_lora-{safe_lora_name}_str-{strength:.3f}.png"
            filepath = os.path.join(run_folder, filename)

            logger.debug(f"  Saving individual image to: {filepath}")
            img_pil.save(filepath)
            logger.info(f"  Saved individual image: {filename}")
        except Exception as e_save:
            # Log warning but don't interrupt the whole process
            logger.warning(f"  Failed to save individual image {img_index}. Error: {e_save}", exc_info=True)


    def _create_placeholder_image(self, base_latent: ComfyLatentT) -> torch.Tensor:
        """Creates a black placeholder image based on latent dimensions."""
        try:
            latent_shape = base_latent['samples'].shape # [B, C, H_latent, W_latent]
            # Estimate image size (standard SD scaling factor)
            H_img = latent_shape[2] * 8
            W_img = latent_shape[3] * 8
            # Create black image [H, W, C=3]
            placeholder = torch.zeros((H_img, W_img, 3), dtype=torch.float32)
            logger.info("  Created black placeholder image due to generation error.")
            return placeholder
        except Exception as e_placeholder:
            logger.error(f"  Failed to create placeholder image after error: {e_placeholder}", exc_info=True)
            # If placeholder fails, we have a bigger problem. Re-raise.
            raise RuntimeError("Image generation failed and placeholder creation also failed.") from e_placeholder


    def _generate_single_image(
        self,
        base_model: ComfyModelObjectT,
        base_clip: ComfyCLIPObjectT,
        base_vae: ComfyVAEObjectT,
        positive: ComfyConditioningT,
        negative: ComfyConditioningT,
        base_latent: ComfyLatentT,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        lora_name: str,
        strength: float,
        lora_folder_path: str,
        run_folder: Optional[str],
        save_individual_images: bool,
        img_index: int,
        total_images: int,
        plot_loras: List[str] # Needed for saving filename logic
    ) -> torch.Tensor:
        """
        Generates a single image tile for the XY plot grid.

        Orchestrates LoRA application, latent prep, sampling, decoding,
        optional saving, and error handling for one grid cell.

        Args:
            base_model, base_clip, base_vae: Base models (will be cloned).
            positive, negative: Conditioning data.
            base_latent: Initial latent dictionary.
            seed, steps, cfg, sampler_name, scheduler: Sampling parameters.
            lora_name (str): Filename of the LoRA to apply ("No LoRA" for baseline).
            strength (float): Strength to apply the LoRA.
            lora_folder_path (str): Path to the LoRA directory.
            run_folder (Optional[str]): Directory to save individual images.
            save_individual_images (bool): Whether to save individual images.
            img_index (int): Current image index (1-based) for logging/seed offset.
            total_images (int): Total number of images in the grid for logging.
            plot_loras (List[str]): List of LoRAs being plotted (for filename).

        Returns:
            torch.Tensor: The generated image tensor [H, W, C]. Returns a black
                          placeholder tensor on error during generation.
        """
        logger.info(f"\nGenerating image {img_index}/{total_images} (LoRA: '{lora_name}', Strength: {strength:.3f})")
        img_tensor_hwc: Optional[torch.Tensor] = None
        current_model = None
        current_clip = None

        try:
            # 1. Apply LoRA (if specified)
            lora_filename_part = "NoLoRA"
            if lora_name != "No LoRA":
                current_model, current_clip, lora_filename_part = self._apply_lora_to_models(
                    base_model, base_clip, lora_name, strength, lora_folder_path
                )
            else:
                # Still need clones even if no LoRA is applied
                current_model = base_model.clone()
                current_clip = base_clip.clone()
                logger.debug("  Skipping LoRA application (baseline image).")

            # 2. Prepare Latent
            current_latent = self._prepare_latent_for_sampling(base_latent, positive)

            # 3. Run Sampling and Decode
            image_tensor_bhwc = self._run_sampling_and_decode(
                current_model, current_clip, base_vae, positive, negative, current_latent,
                seed + img_index - 1, # Increment seed per image
                steps, cfg, sampler_name, scheduler
            )
            # Extract single image (assuming batch size 1 for grid cell)
            # If batch size > 1 due to latent/cond mismatch, we still take the first.
            if image_tensor_bhwc.shape[0] > 1:
                 logger.warning(f"  Sampler returned batch size {image_tensor_bhwc.shape[0]}, using only the first image for the grid.")
            img_tensor_hwc = image_tensor_bhwc[0]


            # 4. Save Individual Image (Optional)
            row_idx = (img_index - 1) // len(plot_loras)
            col_idx = (img_index - 1) % len(plot_loras)
            self._save_image_if_enabled(
                image_tensor_bhwc, run_folder, save_individual_images,
                img_index, row_idx, col_idx, lora_filename_part, strength
            )

        except Exception as e_generate:
            logger.error(f"  ERROR generating image {img_index} (LoRA: '{lora_name}', Str: {strength:.3f}). Error: {e_generate}", exc_info=True)
            # Create placeholder on any generation error
            img_tensor_hwc = self._create_placeholder_image(base_latent)

        finally:
            # --- Clean up GPU Memory ---
            # Ensure models are deleted even if errors occurred mid-process
            del current_model
            del current_clip
            # Request garbage collection and cache clearing
            # This helps prevent memory buildup over many iterations
            comfy.model_management.soft_empty_cache()
            logger.debug("  GPU memory cleanup requested.")

        # Ensure we always return a valid tensor
        if img_tensor_hwc is None:
             # This path should ideally not be hit due to error handling, but acts as a final safeguard.
             logger.error("Image tensor was unexpectedly None after generation attempt. Creating final placeholder.")
             img_tensor_hwc = self._create_placeholder_image(base_latent)

        return img_tensor_hwc

    # --------------------------------------------------------------------------
    # Main Orchestration Method
    # --------------------------------------------------------------------------
    def generate_plot(self,
                      # Required inputs
                      checkpoint_name: str,
                      lora_folder_path: str,
                      positive: ComfyConditioningT,
                      negative: ComfyConditioningT,
                      latent_image: ComfyLatentT,
                      seed: int,
                      steps: int,
                      cfg: float,
                      sampler_name: str,
                      scheduler: str,
                      x_lora_steps: int,
                      y_strength_steps: int,
                      max_strength: float,
                      # Optional inputs
                      opt_clip: Optional[ComfyCLIPObjectT] = None,
                      opt_vae: Optional[ComfyVAEObjectT] = None,
                      save_individual_images: bool = False,
                      output_folder_name: str = "XYPlot_LoRA-Strength",
                      row_gap: int = 0,
                      col_gap: int = 0,
                      draw_labels: bool = True,
                      x_axis_label: str = "",
                      y_axis_label: str = ""
                      ) -> Tuple[torch.Tensor]:
        """
        Orchestrates the entire XY plot generation process.

        This is the main entry point called by ComfyUI when the node executes.
        It performs the following steps:
        1. Validates inputs (LoRA path).
        2. Loads base models (Checkpoint, CLIP, VAE).
        3. Determines the LoRAs and strengths for the plot axes.
        4. Sets up the output directory if saving individual images.
        5. Iterates through each grid cell (LoRA x Strength combination).
        6. Calls `_generate_single_image` for each cell.
        7. Assembles the generated images into a grid using `assemble_image_grid`.
        8. Optionally draws labels on the grid using `draw_labels_on_grid`.
        9. Returns the final grid image tensor (with batch dimension added).

        Args:
            checkpoint_name (str): Name of the base checkpoint file.
            lora_folder_path (str): Path to the folder containing LoRA files.
            positive (ComfyConditioningT): Positive conditioning data.
            negative (ComfyConditioningT): Negative conditioning data.
            latent_image (ComfyLatentT): Initial latent dictionary.
            seed (int): Base seed for generation.
            steps (int): Number of sampling steps.
            cfg (float): CFG scale.
            sampler_name (str): Name of the sampler.
            scheduler (str): Name of the scheduler.
            x_lora_steps (int): Number of LoRAs for X-axis.
            y_strength_steps (int): Number of strength steps for Y-axis.
            max_strength (float): Maximum LoRA strength for Y-axis.
            opt_clip (Optional[ComfyCLIPObjectT]): Optional override CLIP model.
            opt_vae (Optional[ComfyVAEObjectT]): Optional override VAE model.
            save_individual_images (bool): Whether to save individual grid images.
            output_folder_name (str): Subfolder name for saved images.
            row_gap (int): Gap between rows in the final grid.
            col_gap (int): Gap between columns in the final grid.
            draw_labels (bool): Whether to draw labels on the grid.
            x_axis_label (str): Optional overall label for the X-axis.
            y_axis_label (str): Optional overall label for the Y-axis.

        Returns:
            Tuple[torch.Tensor]: A tuple containing the final XY plot image tensor
                                 in ComfyUI's expected format [1, H, W, C].

        Raises:
            RuntimeError: If critical steps like model loading or image generation fail.
        """
        logger.info("--- Starting LoRA vs Strength XY Plot Generation ---")
        start_time = datetime.now()

        try:
            # --- Setup ---
            validated_lora_path = self._validate_inputs(lora_folder_path)
            model, clip, vae = self._load_base_models(checkpoint_name, opt_clip, opt_vae)
            lora_files = self._get_lora_files(validated_lora_path)
            plot_loras, plot_strengths = self._determine_plot_axes(
                lora_files, x_lora_steps, y_strength_steps, max_strength
            )

            num_rows = len(plot_strengths)
            num_cols = len(plot_loras)
            total_images = num_rows * num_cols
            if total_images == 0:
                 logger.warning("Plot axes determined to have zero images. Aborting.")
                 # Consider returning a blank image or raising a specific error
                 raise ValueError("Plot generation resulted in zero images based on inputs.")

            logger.info(f"Preparing to generate {total_images} images ({num_rows} rows x {num_cols} cols).")
            run_folder = self._setup_output_directory(output_folder_name) if save_individual_images else None

            # --- Generation Loop ---
            generated_images: List[torch.Tensor] = []
            generation_successful = True # Flag to track if all images were generated
            for y_idx, strength in enumerate(plot_strengths):
                for x_idx, lora_name in enumerate(plot_loras):
                    try:
                        img_tensor = self._generate_single_image(
                            base_model=model, base_clip=clip, base_vae=vae,
                            positive=positive, negative=negative, base_latent=latent_image,
                            seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
                            lora_name=lora_name, strength=strength, lora_folder_path=validated_lora_path,
                            run_folder=run_folder, save_individual_images=save_individual_images,
                            img_index=(y_idx * num_cols + x_idx + 1), total_images=total_images,
                            plot_loras=plot_loras
                        )
                        generated_images.append(img_tensor)
                    except Exception as e_inner:
                         # Log error from _generate_single_image but continue loop
                         # _generate_single_image should return a placeholder on error
                         logger.error(f"Error in _generate_single_image for cell ({y_idx},{x_idx}): {e_inner}", exc_info=True)
                         # We expect a placeholder was returned, so append it if possible
                         # If _create_placeholder_image also failed, _generate_single_image would re-raise
                         placeholder = self._create_placeholder_image(latent_image) # Recreate placeholder just in case
                         generated_images.append(placeholder)
                         generation_successful = False # Mark that at least one image failed

            # --- Grid Assembly ---
            if not generated_images:
                # This should only happen if the loop didn't run at all (total_images was 0)
                logger.error("No images available for grid assembly.")
                raise RuntimeError("No images were generated or collected. Cannot create grid.")

            logger.info(f"\nAssembling final image grid ({len(generated_images)} images)...")
            # Use the imported grid assembly function
            grid_tensor = assemble_image_grid(generated_images, num_rows, num_cols, row_gap, col_gap)
            logger.debug(f"Grid assembled. Tensor shape: {grid_tensor.shape}")

            # --- Label Drawing ---
            final_labeled_tensor = grid_tensor
            if draw_labels:
                logger.info("Drawing labels on grid...")
                # Prepare labels (strip extension from LoRA names)
                x_axis_labels = [os.path.splitext(name)[0] if name != "No LoRA" else "No LoRA" for name in plot_loras]
                y_axis_labels = [f"{s:.3f}" for s in plot_strengths] # Format strength values
                try:
                    final_labeled_tensor = draw_labels_on_grid(
                        grid_tensor, x_labels=x_axis_labels, y_labels=y_axis_labels,
                        x_axis_label=x_axis_label, y_axis_label=y_axis_label
                        # Pass font size, colors etc. if they become inputs later
                    )
                    logger.debug(f"Labels drawn. Final tensor shape: {final_labeled_tensor.shape}")
                except Exception as e_label:
                     logger.error(f"Failed to draw labels on grid: {e_label}", exc_info=True)
                     # Proceed with the unlabeled grid if labeling fails
                     final_labeled_tensor = grid_tensor
            else:
                 logger.info("Label drawing skipped as per input.")

            # --- Final Output ---
            # Add batch dimension for ComfyUI output format [1, H, W, C]
            final_output_tensor = final_labeled_tensor.unsqueeze(0)

            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"--- XY Plot: Generation Complete (Duration: {duration}) ---")
            if not generation_successful:
                 logger.warning("Plot generation finished, but one or more images failed and were replaced by placeholders.")

            return (final_output_tensor,)

        except Exception as e:
            # Catch errors from setup steps (validation, model load, axes determination)
            logger.critical(f"--- XY Plot: Generation FAILED due to critical error: {e} ---", exc_info=True)
            # Re-raise the exception to make the error visible in ComfyUI
            raise RuntimeError(f"XY Plot generation failed: {e}") from e


# Note: Mappings are handled in xy_plotting/__init__.py
