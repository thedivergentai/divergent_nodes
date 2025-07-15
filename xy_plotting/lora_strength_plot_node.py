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
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logging.error("Failed to import utility functions from grid_assembly or plot_utils. Node will likely fail.", exc_info=True)
    # Define dummy functions to prevent NameErrors, though the node will be broken
    def assemble_image_grid(*args: Any, **kwargs: Any) -> torch.Tensor: raise RuntimeError("grid_assembly not found")
    def draw_labels_on_grid(*args: Any, **kwargs: Any) -> torch.Tensor: raise RuntimeError("grid_assembly not found")
    def validate_lora_path(*args: Any, **kwargs: Any) -> str: raise RuntimeError("plot_utils not found")
    def get_lora_files(*args: Any, **kwargs: Any) -> List[str]: raise RuntimeError("plot_utils not found")
    def determine_plot_axes(*args: Any, **kwargs: Any) -> Tuple[List[str], List[float]]: raise RuntimeError("plot_utils not found")
    def preload_loras(*args: Any, **kwargs: Any) -> Dict[str, Any]: raise RuntimeError("plot_utils not found")
    def setup_output_directory(*args: Any, **kwargs: Any) -> Optional[str]: raise RuntimeError("plot_utils not found")


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

class LoraStrengthXYPlot:
    """
    Generates an XY plot grid comparing LoRAs (X-axis) vs Strength (Y-axis).

    Uses provided Model, CLIP, VAE. Optimized for memory by generating images
    individually, saving them to a temporary directory, and then assembling
    the grid from these files. Supports cancellation and optional preview.
    """
    CATEGORY = "Divergent Nodes 👽/XY Plots"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """Defines the input types for the ComfyUI node interface."""
        try:
            sampler_list = comfy.samplers.KSampler.SAMPLERS or ["ERROR: No Samplers Found"]
        except Exception as e:
            logger.error(f"Failed to get sampler list: {e}", exc_info=True)
            sampler_list = ["ERROR: Failed to Load"]
        try:
            scheduler_list = comfy.samplers.KSampler.SCHEDULERS or ["ERROR: No Schedulers Found"]
        except Exception as e:
            logger.error(f"Failed to get scheduler list: {e}", exc_info=True)
            scheduler_list = ["ERROR: Failed to Load"]

        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Input model."}),
                "clip": ("CLIP", {"tooltip": "Input CLIP model."}),
                "vae": ("VAE", {"tooltip": "Input VAE model."}),
                "lora_folder_path": ("STRING", {"default": "loras/", "multiline": False, "tooltip": "Path to LoRA folder (relative to ComfyUI/models/loras or absolute)."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning."}),
                "latent_image": ("LATENT", {"tooltip": "Initial latent image."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (sampler_list,),
                "scheduler": (scheduler_list,),
                "x_lora_steps": ("INT", {"default": 3, "min": 0, "max": 100, "tooltip": "Number of LoRAs for X-axis (0=all, 1=last)."}),
                "y_strength_steps": ("INT", {"default": 3, "min": 1, "max": 100, "tooltip": "Number of strength steps for Y-axis."}),
                "max_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
            },
            "optional": {
                "save_individual_images": ("BOOLEAN", {"default": False, "tooltip": "Save individual grid images to the output folder."}),
                "display_last_image": ("BOOLEAN", {"default": False, "tooltip": "Output the last generated image as a preview."}), # New Input
                "output_folder_name": ("STRING", {"default": "XYPlot_LoRA-Strength"}),
                "row_gap": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1}),
                "col_gap": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1}),
                "draw_labels": ("BOOLEAN", {"default": True}),
                "x_axis_label": ("STRING", {"default": "LoRA"}),
                "y_axis_label": ("STRING", {"default": "Strength"}),
            },
        }

    # Updated return types and names
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("xy_plot_image", "last_generated_image")
    FUNCTION = "generate_plot"

    # --------------------------------------------------------------------------
    # Core Image Generation Logic (Internal Helpers)
    # --------------------------------------------------------------------------
    def _prepare_latent_for_sampling(self,
                                     base_latent: ComfyLatentT,
                                     positive_cond: ComfyConditioningT
                                     ) -> ComfyLatentT:
        """Prepares the latent dictionary for sampling, handling batch size."""
        current_latent = base_latent.copy()
        if not isinstance(current_latent, dict) or "samples" not in current_latent:
            raise TypeError(f"Invalid latent_image format: {type(base_latent)}. Expected dict with 'samples'.")
        latent_samples = current_latent['samples']
        latent_batch_size = latent_samples.shape[0]
        cond_batch_size = len(positive_cond) if isinstance(positive_cond, list) else 1
        if latent_batch_size != cond_batch_size:
            if latent_batch_size == 1 and cond_batch_size > 1:
                logger.warning(f"  Latent batch (1) != Cond batch ({cond_batch_size}). Repeating latent sample.")
                current_latent['samples'] = latent_samples.repeat(cond_batch_size, 1, 1, 1)
            else:
                # Force batch size 1 for latent if mismatch occurs and latent isn't already 1
                # This might be a less common case but handles potential user errors
                if latent_batch_size > 1:
                    logger.warning(f"  Latent batch ({latent_batch_size}) != Cond batch ({cond_batch_size}). Using only the first latent sample.")
                    current_latent['samples'] = latent_samples[0:1]
                else:
                    logger.warning(f"  Latent batch ({latent_batch_size}) != Cond batch ({cond_batch_size}). Mismatch might cause errors.")
        # Ensure batch size is 1 for the generation loop
        if current_latent['samples'].shape[0] > 1:
             logger.debug("Ensuring latent batch size is 1 for individual image generation.")
             current_latent['samples'] = current_latent['samples'][0:1]

        return current_latent

    def _run_sampling_and_decode(self,
                                 model: ComfyModelObjectT,
                                 clip: ComfyCLIPObjectT,
                                 vae: ComfyVAEObjectT,
                                 positive: ComfyConditioningT,
                                 negative: ComfyConditioningT,
                                 latent: ComfyLatentT, # Keep input name as 'latent' for consistency
                                 seed: int,
                                 steps: int,
                                 cfg: float,
                                 sampler_name: str,
                                 scheduler: str
                                 ) -> TensorHWC:
        """Performs sampling and VAE decoding. Returns tensor [H, W, C]."""
        logger.debug(f"  Starting sampling: {sampler_name}/{scheduler}, Steps: {steps}, CFG: {cfg}, Seed: {seed}")
        try:
            # Ensure latent batch size is 1 before preparing noise
            if latent['samples'].shape[0] != 1:
                 logger.warning(f"Sampler received latent batch size {latent['samples'].shape[0]}, expected 1. Using first sample.")
                 latent['samples'] = latent['samples'][0:1]

            noise = comfy.sample.prepare_noise(latent['samples'], seed)

            samples_latent = comfy.sample.sample(
                model=model,
                noise=noise,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=latent['samples'], # Pass the tensor [1, C, H_lat, W_lat]
                denoise=1.0
            )

            if isinstance(samples_latent, dict) and "samples" in samples_latent:
                latent_to_decode = samples_latent["samples"]
            elif isinstance(samples_latent, torch.Tensor):
                latent_to_decode = samples_latent
            else:
                 raise ValueError(f"Sampler output unexpected format: {type(samples_latent)}. Expected dict with 'samples' or torch.Tensor.")

            logger.debug("  Sampling complete. Decoding...")
            # Decode expects [B, C, H, W], our B is 1
            img_tensor_chw = vae.decode(latent_to_decode)
            logger.debug(f"  Decoding complete. Shape: {img_tensor_chw.shape}")
            # Remove batch dim and permute: [C, H, W] -> [H, W, C]
            img_tensor_hwc = img_tensor_chw.squeeze(0).permute(1, 2, 0)
            return img_tensor_hwc # Return [H, W, C]
        except Exception as e:
            logger.error(f"  Error during sampling or decoding: {e}", exc_info=True)
            raise RuntimeError("Sampling or VAE Decoding failed.") from e

    def _save_tensor_to_file(self,
                             image_tensor_hwc: TensorHWC,
                             filepath: str):
        """Saves a [H, W, C] tensor to a file using PIL."""
        try:
            # Convert tensor to PIL Image
            img_tensor_float32 = image_tensor_hwc.float() # Ensure float32
            img_np = img_tensor_float32.cpu().numpy()
            img_pil = Image.fromarray(np.clip(img_np * 255.0, 0, 255).astype(np.uint8))
            # Save the image
            img_pil.save(filepath)
            logger.debug(f"  Saved image tensor to: {filepath}")
        except Exception as e_save:
            logger.warning(f"  Failed to save image tensor to {filepath}. Error: {e_save}", exc_info=True)
            # Raise the error so the main loop knows saving failed if needed
            raise IOError(f"Failed to save image to {filepath}") from e_save

    def _create_placeholder_image(self, H: int, W: int, C: int, device: torch.device) -> TensorHWC:
        """Creates a black placeholder image tensor [H, W, C] on the specified device."""
        try:
            # Ensure placeholder is float32 for consistency
            placeholder = torch.zeros((H, W, C), dtype=torch.float32, device=device)
            logger.info("  Created black placeholder image due to generation error.")
            return placeholder
        except Exception as e_placeholder:
            logger.error(f"  Failed to create placeholder image: {e_placeholder}", exc_info=True)
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
        loaded_lora: LoadedLoraT, # Pass pre-loaded LoRA tensor
        strength: float,
        img_index: int,
        lora_filename_part: str
    ) -> TensorHWC:
        """
        Generates a single image tile using pre-loaded LoRA. Returns tensor [H, W, C].
        Handles errors by returning a placeholder tensor.
        """
        logger.info(f"\nGenerating image {img_index} (LoRA: '{lora_filename_part}', Strength: {strength:.3f})")
        img_tensor_hwc: Optional[TensorHWC] = None
        current_model = None
        current_clip = None
        generation_failed = False

        try:
            # 1. Clone base models
            current_model = base_model.clone()
            current_clip = base_clip.clone()

            # 2. Apply LoRA (if pre-loaded)
            if loaded_lora is not None:
                logger.debug(f"  Applying pre-loaded LoRA: {lora_filename_part} with strength {strength:.3f}")
                try:
                    current_model, current_clip = comfy.sd.load_lora_for_models(
                        current_model, current_clip, loaded_lora, strength, strength
                    )
                except Exception as e_apply:
                    logger.warning(f"  Failed to apply pre-loaded LoRA '{lora_filename_part}'. Skipping. Error: {e_apply}", exc_info=True)
            else:
                logger.debug(f"  Skipping LoRA application ('{lora_filename_part}' not loaded or is baseline).")

            # 3. Prepare Latent (ensure batch size 1)
            current_latent = self._prepare_latent_for_sampling(base_latent, positive)

            # 4. Run Sampling and Decode
            img_tensor_hwc = self._run_sampling_and_decode(
                current_model, current_clip, base_vae, positive, negative, current_latent,
                seed + img_index - 1, # Increment seed per image
                steps, cfg, sampler_name, scheduler
            )

        except Exception as e_generate:
            logger.error(f"  ERROR generating image {img_index} (LoRA: '{lora_filename_part}', Str: {strength:.3f}). Error: {e_generate}", exc_info=True)
            generation_failed = True
            # Create placeholder on the model's device
            try:
                latent_shape = base_latent['samples'].shape
                H_img, W_img = latent_shape[2] * 8, latent_shape[3] * 8 # Calculate image dimensions from latent
                C_img = 3 # Assume 3 channels (RGB)

                # Determine device from model, fallback to CPU
                if hasattr(base_model, 'model') and hasattr(base_model.model, 'device'):
                    device = base_model.model.device
                elif hasattr(base_model, 'load_device'):
                    device = base_model.load_device
                else:
                    logger.warning("Could not reliably determine model device for placeholder. Falling back to CPU.")
                    device = torch.device('cpu')

                img_tensor_hwc = self._create_placeholder_image(H_img, W_img, C_img, device)

            except Exception as e_placeholder_fallback:
                logger.critical(f"  Failed to determine placeholder dimensions or create placeholder: {e_placeholder_fallback}", exc_info=True)
                # Re-raise the original generation error if placeholder fails
                raise RuntimeError("Failed to generate image and could not create placeholder.") from e_generate

        finally:
            # Clean up clones
            del current_model
            del current_clip

        if img_tensor_hwc is None: # Should not happen due to error handling, but safeguard
             raise RuntimeError(f"Image tensor generation failed unexpectedly for index {img_index}.")

        return img_tensor_hwc # Return [H, W, C] tensor (either generated or placeholder)

    def _load_images_from_paths(self, image_paths: List[str], device: torch.device = torch.device('cpu')) -> List[TensorHWC]:
        """Loads images from file paths into a list of tensors [H, W, C] on the specified device."""
        loaded_tensors = []
        logger.info(f"Loading {len(image_paths)} images from temporary files...")
        for i, path in enumerate(image_paths):
            try:
                if not os.path.exists(path):
                    logger.warning(f"Temporary image file not found: {path}. Skipping.")
                    # Optionally, create a placeholder here if strict grid matching is needed
                    continue

                img_pil = Image.open(path).convert('RGB')
                img_np = np.array(img_pil).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np).to(device) # Move to target device
                loaded_tensors.append(img_tensor)
                logger.debug(f"  Loaded image {i+1}/{len(image_paths)}: {path}")
            except Exception as e:
                logger.error(f"Failed to load image from path {path}: {e}", exc_info=True)
                # Optionally, append a placeholder if an image fails to load
        logger.info(f"Successfully loaded {len(loaded_tensors)} images.")
        return loaded_tensors

    # --------------------------------------------------------------------------
    # Main Orchestration Method
    # --------------------------------------------------------------------------
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
                      display_last_image: bool = False, # New parameter
                      output_folder_name: str = "XYPlot_LoRA-Strength",
                      row_gap: int = 0,
                      col_gap: int = 0,
                      draw_labels: bool = True,
                      x_axis_label: str = "",
                      y_axis_label: str = ""
                      ) -> Tuple[torch.Tensor, torch.Tensor]: # Updated return tuple
        """Orchestrates the XY plot generation using temporary file storage."""
        logger.info("--- Starting LoRA vs Strength XY Plot Generation (Temp File Storage) ---")
        start_time = datetime.now()
        generation_successful = True
        loaded_loras_cache = {}
        temp_dir = None
        generated_image_paths: List[str] = []
        last_image_tensor_cpu: Optional[TensorHWC] = None
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

            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix="lora_strength_plot_")
            logger.info(f"Created temporary directory: {temp_dir}")

            logger.info(f"Preparing to generate {total_images} images ({num_rows} rows x {num_cols} cols).")
            run_folder = setup_output_directory(output_folder_name) if save_individual_images else None

            # --- Generation Loop (Saving to Temp Files) ---
            logger.info("Starting main generation loop...")
            img_idx = 0
            interrupted = False
            for y_idx, strength in enumerate(plot_strengths):
                if interrupted: break
                for x_idx, lora_name in enumerate(plot_loras):
                    img_idx += 1

                    loaded_lora = loaded_loras_cache.get(lora_name)
                    lora_filename_part = os.path.splitext(lora_name)[0] if lora_name != "No LoRA" else "NoLoRA"
                    safe_lora_name = re.sub(r'[\\/*?:"<>|]', '_', lora_filename_part)
                    temp_filename = f"img_{img_idx:04d}_row-{y_idx}_col-{x_idx}_lora-{safe_lora_name}_str-{strength:.3f}.png"
                    temp_filepath = os.path.join(temp_dir, temp_filename)

                    img_tensor_hwc: Optional[TensorHWC] = None
                    try:
                        # Generate image (returns [H, W, C] tensor on model device, or placeholder)
                        img_tensor_hwc = self._generate_single_image(
                            base_model=model, base_clip=clip, base_vae=vae,
                            positive=positive, negative=negative, base_latent=latent_image,
                            seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
                            loaded_lora=loaded_lora, strength=strength,
                            img_index=img_idx, lora_filename_part=lora_filename_part
                        )

                        # Save to temporary file
                        self._save_tensor_to_file(img_tensor_hwc, temp_filepath)
                        generated_image_paths.append(temp_filepath)

                        # Save to permanent output folder if requested
                        if run_folder:
                            perm_filename = f"row-{y_idx}_col-{x_idx}_lora-{safe_lora_name}_str-{strength:.3f}.png"
                            perm_filepath = os.path.join(run_folder, perm_filename)
                            try:
                                self._save_tensor_to_file(img_tensor_hwc, perm_filepath)
                            except Exception as e_perm_save:
                                logger.warning(f"Failed to save image to permanent location {perm_filepath}: {e_perm_save}")
                                # Continue even if permanent save fails

                        # Update last image for preview (move to CPU)
                        last_image_tensor_cpu = img_tensor_hwc.cpu().clone()

                    except Exception as e_inner:
                         logger.error(f"Error processing image {img_idx} for cell ({y_idx},{x_idx}): {e_inner}", exc_info=True)
                         generation_successful = False
                         # Placeholder should have been created by _generate_single_image
                         # If saving failed, path won't be added, grid assembly will handle missing files
                    finally:
                         # Clean up GPU tensor immediately
                         del img_tensor_hwc
                         comfy.model_management.soft_empty_cache()

            # --- Post-Loop Assembly ---
            if not generated_image_paths:
                logger.error("No images were generated or saved successfully.")
                raise RuntimeError("Failed to generate any images for the plot.")

            # Load images from temporary files (to CPU)
            # Determine device for assembly (prefer CPU for PIL drawing)
            assembly_device = torch.device('cpu')
            loaded_image_tensors = self._load_images_from_paths(generated_image_paths, device=assembly_device)

            if not loaded_image_tensors:
                 raise RuntimeError("Failed to load any generated images from temporary files.")

            # Determine actual grid size based on loaded images
            actual_cols = num_cols
            actual_rows = (len(loaded_image_tensors) + actual_cols - 1) // actual_cols # Calculate rows based on loaded count
            logger.info(f"Assembling grid from {len(loaded_image_tensors)} loaded images into {actual_rows}x{actual_cols} grid.")

            # Assemble the grid
            assembled_grid_tensor = assemble_image_grid(
                loaded_image_tensors, actual_rows, actual_cols, row_gap, col_gap
            )
            del loaded_image_tensors # Free memory

            # --- Label Drawing ---
            final_labeled_tensor = assembled_grid_tensor # Already on CPU, float32
            if draw_labels:
                logger.info("Drawing labels on grid...")
                # Adjust labels if interrupted
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
                    logger.debug(f"Labels drawn. Final tensor shape: {final_labeled_tensor.shape}, Device: {final_labeled_tensor.device}")
                except Exception as e_label:
                     logger.error(f"Failed to draw labels on grid: {e_label}", exc_info=True)
                     # Use unlabeled grid if labeling fails
            else:
                 logger.info("Label drawing skipped.")

            # --- Final Output Preparation ---
            # Ensure final tensor is float32 and has batch dimension
            final_output_tensor = final_labeled_tensor.float().unsqueeze(0)

            # Prepare preview tensor
            if display_last_image and last_image_tensor_cpu is not None:
                preview_tensor = last_image_tensor_cpu.float().unsqueeze(0)
            elif display_last_image:
                 logger.warning("Display last image requested, but no image was successfully generated/kept.")
                 # Keep default empty preview

            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"--- XY Plot: Generation Finished (Duration: {duration}) ---")
            if not generation_successful:
                 logger.warning("Plot generation finished, but one or more images may have failed or process was interrupted.")

            return (final_output_tensor, preview_tensor)

        except Exception as e:
            logger.critical(f"--- XY Plot: Generation FAILED: {e} ---", exc_info=True)
            # Return default empty tensors on critical failure
            return (final_output_tensor, preview_tensor)

        finally:
            # --- Cleanup ---
            del loaded_loras_cache
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Removed temporary directory: {temp_dir}")
                except Exception as e_cleanup:
                    logger.error(f"Failed to remove temporary directory {temp_dir}: {e_cleanup}", exc_info=True)
            comfy.model_management.soft_empty_cache()

# Note: Mappings are handled in xy_plotting/__init__.py
