"""
Node implementation for generating an XY plot comparing LoRA models vs. strength.
Refactored for memory efficiency using temporary file storage, direct model/clip/vae inputs,
cancellation support, and optional preview image output.
"""
import torch
import numpy as np
import os
import comfy.sd
import comfy.utils
import comfy.samplers
import comfy.model_management
import folder_paths
from PIL import Image
from datetime import datetime
import re
import logging
import tempfile
import shutil
from typing import Dict, Any, Tuple, Optional, List, Union, Sequence, TypeAlias

# Import ComfyUI types for better type hinting and autocomplete
from comfy.comfy_types import ComfyNodeABC, IO, InputTypeDict

# Import custom log level
from ..shared_utils.logging_utils import SUCCESS_HIGHLIGHT

# Import utility functions
try:
    from .grid_assembly import assemble_image_grid, draw_labels_on_grid
    from .plot_utils import (
        validate_lora_path,
        get_lora_files,
        determine_plot_axes,
        preload_loras,
        setup_output_directory
    )
    from .sampling_helpers import (
        prepare_latent_for_sampling,
        run_sampling_and_save_latent,
        load_latent_and_decode,
        create_placeholder_latent,
        create_placeholder_image,
        generate_single_latent
    )
    from .image_helpers import save_tensor_to_file
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logging.error("❌ [XYPlot] Failed to import utility functions. This node will likely fail. Please ensure all Divergent Nodes files are correctly installed.", exc_info=True)
    # Define dummy functions to prevent NameErrors, though the node will be broken
    def assemble_image_grid(*args: Any, **kwargs: Any) -> torch.Tensor: raise RuntimeError("grid_assembly not found")
    def draw_labels_on_grid(*args: Any, **kwargs: Any) -> torch.Tensor: raise RuntimeError("grid_assembly not found")
    def validate_lora_path(*args: Any, **kwargs: Any) -> str: raise RuntimeError("plot_utils not found")
    def get_lora_files(*args: Any, **kwargs: Any) -> List[str]: raise RuntimeError("plot_utils not found")
    def determine_plot_axes(*args: Any, **kwargs: Any) -> Tuple[List[str], List[float]]: raise RuntimeError("plot_utils not found")
    def preload_loras(*args: Any, **kwargs: Any) -> Dict[str, Any]: raise RuntimeError("plot_utils not found")
    def setup_output_directory(*args: Any, **kwargs: Any) -> Optional[str]: raise RuntimeError("plot_utils not found")
    def prepare_latent_for_sampling(*args: Any, **kwargs: Any) -> Any: raise RuntimeError("sampling_helpers not found")
    def run_sampling_and_save_latent(*args: Any, **kwargs: Any) -> Any: raise RuntimeError("sampling_helpers not found")
    def load_latent_and_decode(*args: Any, **kwargs: Any) -> Any: raise RuntimeError("sampling_helpers not found")
    def create_placeholder_latent(*args: Any, **kwargs: Any) -> Any: raise RuntimeError("sampling_helpers not found")
    def create_placeholder_image(*args: Any, **kwargs: Any) -> Any: raise RuntimeError("sampling_helpers not found")
    def generate_single_latent(*args: Any, **kwargs: Any) -> Any: raise RuntimeError("sampling_helpers not found")
    def save_tensor_to_file(*args: Any, **kwargs: Any) -> Any: raise RuntimeError("image_helpers not found")


# Setup logger
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Define type hints
ComfyConditioningT: TypeAlias = List[Tuple[torch.Tensor, Dict[str, Any]]]
ComfyCLIPObjectT: TypeAlias = Any
ComfyVAEObjectT: TypeAlias = Any
ComfyModelObjectT: TypeAlias = Any
ComfyLatentT: TypeAlias = Dict[str, torch.Tensor]
LoadedLoraT: TypeAlias = Optional[Dict[str, torch.Tensor]]
TensorHWC: TypeAlias = torch.Tensor # Expected shape [H, W, C]

class LoraStrengthXYPlot(ComfyNodeABC): # Inherit from ComfyNodeABC
    """
    Generates an XY plot grid comparing LoRAs (X-axis) vs Strength (Y-axis).

    Uses provided Model, CLIP, VAE. Optimized for memory by generating images
    individually, saving them to a temporary directory, and then assembling
    the grid from these files. Supports cancellation and optional preview.
    """
    CATEGORY: str = "Divergent Nodes 👽/XY Plots" # Use type hint for CATEGORY
    OUTPUT_NODE: bool = True # Use type hint for OUTPUT_NODE

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict: # Use InputTypeDict for type hinting
        """Defines the input types for the ComfyUI node interface."""
        try:
            sampler_list = comfy.samplers.KSampler.SAMPLERS or ["ERROR: No Samplers Found"]
        except Exception as e:
            logger.error(f"❌ [XYPlot] Failed to get sampler list. This might affect available sampler options. Details: {e}", exc_info=True)
            sampler_list = ["ERROR: Failed to Load"]
        try:
            scheduler_list = comfy.samplers.KSampler.SCHEDULERS or ["ERROR: No Schedulers Found"]
        except Exception as e:
            logger.error(f"❌ [XYPlot] Failed to get scheduler list. This might affect available scheduler options. Details: {e}", exc_info=True)
            scheduler_list = ["ERROR: Failed to Load"]

        return {
            "required": {
                "model": (IO.MODEL, {"tooltip": "Input model."}), # Use IO.MODEL
                "clip": (IO.CLIP, {"tooltip": "Input CLIP model."}), # Use IO.CLIP
                "vae": (IO.VAE, {"tooltip": "Input VAE model."}), # Use IO.VAE
                "lora_folder_path": (IO.STRING, {"default": "loras/", "multiline": False, "tooltip": "Path to LoRA folder (relative to ComfyUI/models/loras or absolute)."}), # Use IO.STRING
                "positive": (IO.CONDITIONING, {"tooltip": "Positive conditioning."}), # Use IO.CONDITIONING
                "negative": (IO.CONDITIONING, {"tooltip": "Negative conditioning."}), # Use IO.CONDITIONING
                "latent_image": (IO.LATENT, {"tooltip": "Initial latent image."}), # Use IO.LATENT
                "seed": (IO.INT, {"default": 0, "min": 0, "max": 0xffffffffffffffff}), # Use IO.INT
                "steps": (IO.INT, {"default": 20, "min": 1, "max": 10000}), # Use IO.INT
                "cfg": (IO.FLOAT, {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}), # Use IO.FLOAT
                "sampler_name": (sampler_list,),
                "scheduler": (scheduler_list,),
                "x_lora_steps": (IO.INT, {"default": 3, "min": 0, "max": 100, "tooltip": "Number of LoRAs for X-axis (0=all, 1=last)."}), # Use IO.INT
                "y_strength_steps": (IO.INT, {"default": 3, "min": 1, "max": 100, "tooltip": "Number of strength steps for Y-axis."}), # Use IO.INT
                "max_strength": (IO.FLOAT, {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}), # Use IO.FLOAT
            },
            "optional": {
                "save_individual_images": (IO.BOOLEAN, {"default": False, "tooltip": "Save individual grid images to the output folder."}), # Use IO.BOOLEAN
                "display_last_image": (IO.BOOLEAN, {"default": False, "tooltip": "Output the last generated image as a preview."}), # New Input, Use IO.BOOLEAN
                "output_folder_name": (IO.STRING, {"default": "XYPlot_LoRA-Strength"}), # Use IO.STRING
                "row_gap": (IO.INT, {"default": 10, "min": 0, "max": 200, "step": 1}), # Use IO.INT
                "col_gap": (IO.INT, {"default": 10, "min": 0, "max": 200, "step": 1}), # Use IO.INT
                "draw_labels": (IO.BOOLEAN, {"default": True}), # Use IO.BOOLEAN
                "x_axis_label": (IO.STRING, {"default": "LoRA"}), # Use IO.STRING
                "y_axis_label": (IO.STRING, {"default": "Strength"}), # Use IO.STRING
            },
        }

    # Updated return types and names
    RETURN_TYPES: Tuple[str] = (IO.IMAGE, IO.IMAGE) # Use IO.IMAGE
    RETURN_NAMES: Tuple[str] = ("xy_plot_image", "last_generated_image")
    FUNCTION: str = "generate_plot" # Use type hint for FUNCTION

    def generate_plot(self,
                      model: ComfyModelObjectT,
                      clip: ComfyCLIPObjectT,
                      vae: ComfyVAEObjectT,
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
                      save_individual_images: bool = False,
                      display_last_image: bool = False,
                      output_folder_name: str = "XYPlot_LoRA-Strength",
                      row_gap: int = 0,
                      col_gap: int = 0,
                      draw_labels: bool = True,
                      x_axis_label: str = "",
                      y_axis_label: str = ""
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Orchestrates the XY plot generation using temporary file storage for latents.

        This method generates images for each combination of LoRA and strength,
        saves the latents to disk, then decodes them sequentially for grid assembly.
        This approach significantly reduces peak memory usage.

        Args:
            model (ComfyModelObjectT): The ComfyUI model object.
            clip (ComfyCLIPObjectT): The ComfyUI CLIP object.
            vae (ComfyVAEObjectT): The ComfyUI VAE model object.
            lora_folder_path (str): Path to the LoRA folder.
            positive (ComfyConditioningT): Positive conditioning.
            negative (ComfyConditioningT): Negative conditioning.
            latent_image (ComfyLatentT): Initial latent image dictionary.
            seed (int): Base seed for random number generation.
            steps (int): Number of sampling steps.
            cfg (float): Classifier-free guidance scale.
            sampler_name (str): Name of the sampler to use.
            scheduler (str): Name of the scheduler to use.
            x_lora_steps (int): Number of LoRAs for X-axis (0=all, 1=last).
            y_strength_steps (int): Number of strength steps for Y-axis.
            max_strength (float): Maximum LoRA strength.
            save_individual_images (bool): Whether to save individual grid images.
            display_last_image (bool): Whether to output the last generated image as a preview.
            output_folder_name (str): Name of the output folder.
            row_gap (int): Gap between rows in pixels.
            col_gap (int): Gap between columns in pixels.
            draw_labels (bool): Whether to draw labels on the grid.
            x_axis_label (str): Label for the X-axis.
            y_axis_label (str): Label for the Y-axis.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The assembled XY plot image tensor [1, H_grid, W_grid, C].
                - The last generated image tensor [1, H, W, C] (for preview).

        Raises:
            ValueError: If plot generation results in zero images.
            RuntimeError: If no latents are generated/saved or no images are decoded.
        """
        logger.info("🚀 [XYPlot] --- Starting LoRA vs Strength XY Plot Generation (Disk-Backed Latent Storage) ---")
        start_time = datetime.now()
        generation_successful = True
        loaded_loras_cache = {}
        temp_dir = None
        generated_latent_paths: List[str] = []
        last_decoded_image_tensor_cpu: Optional[TensorHWC] = None
        final_output_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32) # Default empty output
        preview_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32) # Default empty preview

        try:
            # --- Setup ---
            validated_lora_path = validate_lora_path(lora_folder_path)
            lora_files = get_lora_files(validated_lora_path)
            plot_loras, plot_strengths = determine_plot_axes(
                lora_files, x_lora_steps, y_strength_steps, max_strength
            )
            loaded_loras_cache = preload_loras(plot_loras, validated_lora_path)

            num_rows = len(plot_strengths)
            num_cols = len(plot_loras)
            total_images = num_rows * num_cols
            if total_images == 0:
                 raise ValueError("Plot generation resulted in zero images based on inputs.")

            # Input Validation: Check latent_image batch size
            if latent_image['samples'].shape[0] > 1:
                logger.warning(f"⚠️ [XYPlot] Input latent_image has batch size {latent_image['samples'].shape[0]}. Only the first sample will be used for plot generation to ensure consistent grid cells.")
                # Ensure the base_latent passed to prepare_latent_for_sampling is also just the first sample
                latent_image['samples'] = latent_image['samples'][0:1]

            # Create temporary directory for latents
            temp_dir = tempfile.mkdtemp(prefix="lora_strength_plot_latents_")
            logger.info(f"📁 [XYPlot] Created temporary directory for latents: {temp_dir}")

            logger.info(f"⚙️ [XYPlot] Preparing to generate {total_images} latents ({num_rows} rows x {num_cols} cols).")
            run_folder = setup_output_directory(output_folder_name) if save_individual_images else None

            # --- Latent Generation Loop (Saving Latents to Temp Files) ---
            logger.info("🔁 [XYPlot] Starting main latent generation loop...")
            img_idx = 0
            interrupted = False
            for y_idx, strength in enumerate(plot_strengths):
                if interrupted: break
                for x_idx, lora_name in enumerate(plot_loras):
                    img_idx += 1

                    loaded_lora = loaded_loras_cache.get(lora_name)
                    lora_filename_part = os.path.splitext(lora_name)[0] if lora_name != "No LoRA" else "NoLoRA"
                    safe_lora_name = re.sub(r'[\\/*?:"<>|]', '_', lora_filename_part)
                    # Use .pt extension for PyTorch tensors
                    temp_latent_filename = f"latent_{img_idx:04d}_row-{y_idx}_col-{x_idx}_lora-{safe_lora_name}_str-{strength:.3f}.pt"
                    temp_latent_filepath = os.path.join(temp_dir, temp_latent_filename)

                    try:
                        # Generate latent and save to temporary file
                        saved_latent_path = generate_single_latent(
                            base_model=model, base_clip=clip,
                            positive=positive, negative=negative, base_latent=latent_image,
                            seed=seed + img_idx - 1, # Increment seed per image
                            steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
                            loaded_lora=loaded_lora, strength=strength,
                            img_index=img_idx, lora_filename_part=lora_filename_part,
                            temp_filepath=temp_latent_filepath # Pass filepath for saving
                        )
                        generated_latent_paths.append(saved_latent_path)

                    except Exception as e_inner:
                         logger.error(f"❌ [XYPlot] Error processing latent {img_idx} for cell ({y_idx},{x_idx}). A placeholder will be used. Details: {e_inner}", exc_info=True)
                         generation_successful = False
                         # If generate_single_latent failed to save even a placeholder, this path won't be added.
                         # If it saved a placeholder, the path is added and will be handled later.
                    finally:
                         comfy.model_management.soft_empty_cache() # Ensure GPU memory is freed after each latent generation

            # --- Post-Loop: Latent Decoding and Grid Assembly ---
            if not generated_latent_paths:
                logger.error("❌ [XYPlot] No latents were generated or saved successfully. Cannot create plot.")
                raise RuntimeError("Failed to generate any latents for the plot.")

            logger.info(f"🖼️ [XYPlot] Starting VAE decoding and grid assembly from {len(generated_latent_paths)} latents.")
            decoded_image_tensors_for_grid: List[TensorHWC] = []

            # Determine device for VAE decoding (model's device)
            vae_device = vae.device if hasattr(vae, 'device') else torch.device('cpu')

            for i, latent_path in enumerate(generated_latent_paths):
                try:
                    # Load latent from disk and decode
                    decoded_img_tensor_hwc = load_latent_and_decode(vae, latent_path, device=vae_device)
                    decoded_image_tensors_for_grid.append(decoded_img_tensor_hwc.cpu()) # Move to CPU for grid assembly

                    # Save individual decoded images if requested
                    if save_individual_images and run_folder:
                        # Reconstruct original filename parts for saving
                        match = re.search(r'latent_(\d{4})_row-(\d+)_col-(\d+)_lora-(.+?)_str-([\d.]+)\.pt', os.path.basename(latent_path))
                        if match:
                            img_idx_str, y_idx_str, x_idx_str, safe_lora_name, strength_str = match.groups()
                            perm_filename = f"row-{y_idx_str}_col-{x_idx_str}_lora-{safe_lora_name}_str-{strength_str}.png"
                            perm_filepath = os.path.join(run_folder, perm_filename)
                            try:
                                save_tensor_to_file(decoded_img_tensor_hwc, perm_filepath)
                            except Exception as e_perm_save:
                                logger.warning(f"⚠️ [XYPlot] Failed to save individual image to permanent location {perm_filepath}. This image may be missing from your output folder. Error: {e_perm_save}")
                        else:
                            logger.warning(f"⚠️ [XYPlot] Could not parse latent filename for individual save: {os.path.basename(latent_path)}. Individual image may not be saved correctly.")

                    # Update last image for preview (already on CPU)
                    last_decoded_image_tensor_cpu = decoded_img_tensor_hwc.cpu().clone()

                except Exception as e_decode:
                    logger.error(f"❌ [XYPlot] Error decoding latent from {latent_path}. A placeholder image will be used for the grid. Details: {e_decode}", exc_info=True)
                    generation_successful = False
                    # Append a placeholder image to maintain grid structure
                    try:
                        latent_shape = latent_image['samples'].shape
                        H_img, W_img = latent_shape[2] * 8, latent_shape[3] * 8
                        C_img = 3
                        # Ensure placeholder image is created on CPU
                        placeholder_img = create_placeholder_image(H_img, W_img, C_img, torch.device('cpu'))
                        decoded_image_tensors_for_grid.append(placeholder_img)
                    except Exception as e_ph_create:
                        logger.critical(f"🚨 [XYPlot] CRITICAL: Failed to create placeholder image for grid assembly after decode error. Plot assembly may fail. Details: {e_ph_create}", exc_info=True)
                        # If placeholder creation fails, the grid assembly might fail or be malformed.
                        # This is a critical state, but we try to continue to provide some output.
                finally:
                    # Ensure GPU memory is freed after each decode
                    comfy.model_management.soft_empty_cache()

            if not decoded_image_tensors_for_grid:
                 logger.error("❌ [XYPlot] No images were successfully decoded for grid assembly. Cannot assemble plot.")
                 raise RuntimeError("Failed to decode any images for the plot.")

            # Determine actual grid size based on decoded images
            actual_cols = num_cols
            actual_rows = (len(decoded_image_tensors_for_grid) + actual_cols - 1) // actual_cols
            logger.info(f"🔲 [XYPlot] Assembling grid from {len(decoded_image_tensors_for_grid)} decoded images into {actual_rows}x{actual_cols} grid.")

            # Assemble the grid (images are already on CPU)
            assembled_grid_tensor = assemble_image_grid(
                decoded_image_tensors_for_grid, actual_rows, actual_cols, row_gap, col_gap
            )
            del decoded_image_tensors_for_grid # Free memory

            # --- Label Drawing ---
            final_labeled_tensor = assembled_grid_tensor # Already on CPU, float32
            if draw_labels:
                logger.info("✍️ [XYPlot] Drawing labels on grid...")
                # Adjust labels if interrupted or if some images failed
                actual_plot_loras = plot_loras[:actual_cols]
                actual_plot_strengths = plot_strengths[:actual_rows]
                x_axis_labels = [os.path.splitext(name)[0] if name != "No LoRA" else "No LoRA" for name in actual_plot_loras]
                y_axis_labels = [f"{s:.3f}" for s in actual_plot_strengths]
                try:
                    final_labeled_tensor = draw_labels_on_grid(
                        assembled_grid_tensor, x_labels=x_axis_labels, y_labels=y_axis_labels,
                        x_axis_label=x_axis_label, y_axis_label=y_axis_label,
                        row_gap=row_gap, col_gap=col_gap
                    )
                    logger.debug(f"🐛 [XYPlot] Labels drawn. Final tensor shape: {final_labeled_tensor.shape}, Device: {final_labeled_tensor.device}")
                except Exception as e_label:
                     logger.error(f"❌ [XYPlot] Failed to draw labels on grid. The plot image will be generated without labels. Details: {e_label}", exc_info=True)
                     # Use unlabeled grid if labeling fails
            else:
                 logger.info("ℹ️ [XYPlot] Label drawing skipped.")

            # --- Final Output Preparation ---
            # Ensure final tensor is float32 and has batch dimension
            final_output_tensor = final_labeled_tensor.float().unsqueeze(0)

            # Prepare preview tensor
            if display_last_image and last_decoded_image_tensor_cpu is not None:
                preview_tensor = last_decoded_image_tensor_cpu.float().unsqueeze(0)
            elif display_last_image:
                 logger.warning("⚠️ [XYPlot] Display last image requested, but no image was successfully generated or kept. Outputting a blank image.")
                 # Keep default empty preview

            end_time = datetime.now()
            duration = end_time - start_time
            logger.log(SUCCESS_HIGHLIGHT, f"--- XY Plot: Generation Finished (Duration: {duration}) ---")
            if not generation_successful:
                 logger.warning("⚠️ [XYPlot] Plot generation finished, but one or more images may have failed or the process was interrupted. Please check the console for details.")

            return (final_output_tensor, preview_tensor)

        except Exception as e:
            logger.critical(f"🚨 [XYPlot] --- XY Plot: Generation FAILED: {e} --- Please review the error and your workflow settings.", exc_info=True)
            # Return default empty tensors on critical failure
            return (final_output_tensor, preview_tensor)

        finally:
            # --- Cleanup ---
            del loaded_loras_cache
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"🗑️ [XYPlot] Removed temporary directory for latents: {temp_dir}")
                except Exception as e_cleanup:
                    logger.error(f"❌ [XYPlot] Failed to remove temporary directory {temp_dir}. Please manually delete it if it persists. Details: {e_cleanup}", exc_info=True)

            comfy.model_management.soft_empty_cache()

# Note: Mappings are handled in xy_plotting/__init__.py
