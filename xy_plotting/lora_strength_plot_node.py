"""
Node implementation for generating an XY plot comparing LoRA models vs. strength.
Refactored for memory efficiency, direct model/clip/vae inputs, and modularity.
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
from typing import Dict, Any, Tuple, Optional, List, Union, Sequence, TypeAlias

# Import utility functions
try:
    from .grid_assembly import draw_labels_on_grid
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

class LoraStrengthXYPlot:
    """
    Generates an XY plot grid comparing LoRAs (X-axis) vs Strength (Y-axis).

    Uses provided Model, CLIP, VAE. Optimized for memory by pre-loading
    LoRAs and assembling the grid directly on the GPU. Setup logic is
    delegated to plot_utils.
    """
    CATEGORY = "👽 Divergent Nodes/XY Plots"
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
                "save_individual_images": ("BOOLEAN", {"default": False}),
                "output_folder_name": ("STRING", {"default": "XYPlot_LoRA-Strength"}),
                "row_gap": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1}),
                "col_gap": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1}),
                "draw_labels": ("BOOLEAN", {"default": True}),
                "x_axis_label": ("STRING", {"default": "LoRA"}),
                "y_axis_label": ("STRING", {"default": "Strength"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("xy_plot_image",)
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
                logger.warning(f"  Latent batch ({latent_batch_size}) != Cond batch ({cond_batch_size}). Mismatch might cause errors.")
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
                                 ) -> torch.Tensor:
        """Performs sampling and VAE decoding."""
        logger.debug(f"  Starting sampling: {sampler_name}/{scheduler}, Steps: {steps}, CFG: {cfg}, Seed: {seed}")
        try:
            # --- FIX 1: Correct keyword argument for latent tensor ---
            # Use keyword arguments for clarity and correctness
            samples_latent = comfy.sample.sample(
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=latent['samples'], # Pass the tensor with correct keyword
                denoise=1.0
                # Note: clip and vae are typically handled implicitly by comfy.sample.sample
            )
            # --- End FIX 1 ---

            # Handle potential variations in the return type of sample
            if isinstance(samples_latent, dict) and "samples" in samples_latent:
                latent_to_decode = samples_latent["samples"]
            elif isinstance(samples_latent, torch.Tensor): # Direct tensor return
                latent_to_decode = samples_latent
            else:
                 raise ValueError(f"Sampler output unexpected format: {type(samples_latent)}. Expected dict with 'samples' or torch.Tensor.")

            logger.debug("  Sampling complete. Decoding...")
            img_tensor_chw = vae.decode(latent_to_decode) # [B, C, H, W]
            logger.debug(f"  Decoding complete. Shape: {img_tensor_chw.shape}")
            img_tensor_bhwc = img_tensor_chw.permute(0, 2, 3, 1) # [B, H, W, C]
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
        """Saves the generated image to disk if enabled."""
        if not (save_individual_images and run_folder): return
        try:
            img_tensor_hwc = image_tensor_bhwc[0] # Assume batch 1 for saving
            # --- FIX 2a: Convert to float32 before numpy ---
            img_np = img_tensor_hwc.float().cpu().numpy()
            # --- End FIX 2a ---
            img_pil = Image.fromarray(np.clip(img_np * 255.0, 0, 255).astype(np.uint8))
            safe_lora_name = re.sub(r'[\\/*?:"<>|]', '_', lora_filename_part)
            filename = f"row-{row_idx}_col-{col_idx}_lora-{safe_lora_name}_str-{strength:.3f}.png"
            filepath = os.path.join(run_folder, filename)
            logger.debug(f"  Saving individual image to: {filepath}")
            img_pil.save(filepath)
        except Exception as e_save:
            logger.warning(f"  Failed to save individual image {img_index}. Error: {e_save}", exc_info=True)

    def _create_placeholder_image(self, H: int, W: int, C: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Creates a black placeholder image."""
        try:
            # Ensure placeholder is float32 for consistency, even if model dtype was different
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
        run_folder: Optional[str],
        save_individual_images: bool,
        img_index: int,
        row_idx: int,
        col_idx: int,
        lora_filename_part: str
    ) -> torch.Tensor:
        """
        Generates a single image tile using pre-loaded LoRA. Returns tensor [H, W, C].
        """
        logger.info(f"\nGenerating image {img_index} (LoRA: '{lora_filename_part}', Strength: {strength:.3f})")
        img_tensor_hwc: Optional[torch.Tensor] = None
        current_model = None
        current_clip = None

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

            # 3. Prepare Latent
            current_latent = self._prepare_latent_for_sampling(base_latent, positive)

            # 4. Run Sampling and Decode
            image_tensor_bhwc = self._run_sampling_and_decode(
                current_model, current_clip, base_vae, positive, negative, current_latent,
                seed + img_index - 1, # Increment seed per image
                steps, cfg, sampler_name, scheduler
            )

            # Extract single image [H, W, C]
            if image_tensor_bhwc.shape[0] > 1:
                 logger.warning(f"  Sampler returned batch size {image_tensor_bhwc.shape[0]}, using only the first image.")
            img_tensor_hwc = image_tensor_bhwc[0]

            # 5. Save Individual Image (Optional)
            self._save_image_if_enabled(
                image_tensor_bhwc, run_folder, save_individual_images,
                img_index, row_idx, col_idx, lora_filename_part, strength
            )

        except Exception as e_generate:
            logger.error(f"  ERROR generating image {img_index} (LoRA: '{lora_filename_part}', Str: {strength:.3f}). Error: {e_generate}", exc_info=True)
            # Create placeholder
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

                # Placeholder dtype is handled in _create_placeholder_image (always float32)
                img_tensor_hwc = self._create_placeholder_image(H_img, W_img, C_img, device, torch.float32) # Pass float32 explicitly

            except Exception as e_placeholder_fallback:
                logger.critical(f"  Failed to determine placeholder dimensions or create placeholder: {e_placeholder_fallback}", exc_info=True)
                raise RuntimeError("Failed to generate image and could not create placeholder.") from e_generate

        finally:
            # Clean up clones
            del current_model
            del current_clip

        if img_tensor_hwc is None: # Should not happen due to error handling, but safeguard
             raise RuntimeError(f"Image tensor generation failed unexpectedly for index {img_index}.")

        return img_tensor_hwc

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
                      output_folder_name: str = "XYPlot_LoRA-Strength",
                      row_gap: int = 0,
                      col_gap: int = 0,
                      draw_labels: bool = True,
                      x_axis_label: str = "",
                      y_axis_label: str = ""
                      ) -> Tuple[torch.Tensor]:
        """Orchestrates the XY plot generation using utilities."""
        logger.info("--- Starting LoRA vs Strength XY Plot Generation (Optimized/Modular) ---")
        start_time = datetime.now()
        generation_successful = True
        loaded_loras_cache = {} # Keep cache local to this execution

        try:
            # --- Setup using imported utils ---
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

            logger.info(f"Preparing to generate {total_images} images ({num_rows} rows x {num_cols} cols).")
            run_folder = setup_output_directory(output_folder_name) if save_individual_images else None

            # --- Determine Grid Geometry & Allocate Grid Tensor ---
            first_img_index = 1
            first_lora_name = plot_loras[0]
            first_strength = plot_strengths[0]
            first_loaded_lora = loaded_loras_cache.get(first_lora_name)
            first_lora_filename_part = os.path.splitext(first_lora_name)[0] if first_lora_name != "No LoRA" else "NoLoRA"

            logger.info("Generating first image to determine grid geometry...")
            first_image_hwc = self._generate_single_image(
                base_model=model, base_clip=clip, base_vae=vae,
                positive=positive, negative=negative, base_latent=latent_image,
                seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
                loaded_lora=first_loaded_lora, strength=first_strength,
                run_folder=run_folder, save_individual_images=save_individual_images,
                img_index=first_img_index, row_idx=0, col_idx=0,
                lora_filename_part=first_lora_filename_part
            )
            comfy.model_management.soft_empty_cache()

            H, W, C = first_image_hwc.shape
            dtype = torch.float32 # Standardize grid to float32
            device = first_image_hwc.device
            logger.info(f"Image dimensions: {H}H x {W}W x {C}C, Type: {first_image_hwc.dtype} -> Grid Type: {dtype}, Device: {device}")

            grid_height = H * num_rows + max(0, row_gap * (num_rows - 1))
            grid_width = W * num_cols + max(0, col_gap * (num_cols - 1))
            logger.info(f"Allocating final grid tensor: {grid_height}H x {grid_width}W x {C}C on device {device}")
            final_grid = torch.zeros((grid_height, grid_width, C), dtype=dtype, device=device)

            # Paste the first image (convert to float32 if needed)
            y_start, x_start = 0, 0
            final_grid[y_start:y_start + H, x_start:x_start + W, :] = first_image_hwc.to(dtype=dtype, device=device)
            del first_image_hwc
            logger.debug("Pasted first image into grid.")

            # --- Generation Loop (Optimized) ---
            logger.info("Starting main generation loop...")
            img_idx = 1
            for y_idx, strength in enumerate(plot_strengths):
                current_row_y = y_idx * (H + row_gap)
                for x_idx, lora_name in enumerate(plot_loras):
                    img_idx += 1
                    if y_idx == 0 and x_idx == 0: continue # Skip first

                    current_col_x = x_idx * (W + col_gap)
                    loaded_lora = loaded_loras_cache.get(lora_name)
                    lora_filename_part = os.path.splitext(lora_name)[0] if lora_name != "No LoRA" else "NoLoRA"

                    try:
                        img_tensor_hwc = self._generate_single_image(
                            base_model=model, base_clip=clip, base_vae=vae,
                            positive=positive, negative=negative, base_latent=latent_image,
                            seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
                            loaded_lora=loaded_lora, strength=strength,
                            run_folder=run_folder, save_individual_images=save_individual_images,
                            img_index=img_idx, row_idx=y_idx, col_idx=x_idx,
                            lora_filename_part=lora_filename_part
                        )
                        # Paste directly into the grid (convert to float32)
                        final_grid[current_row_y:current_row_y + H, current_col_x:current_col_x + W, :] = img_tensor_hwc.to(dtype=dtype, device=device)
                        del img_tensor_hwc # Free memory immediately
                    except Exception as e_inner:
                         logger.error(f"Error generating/pasting image {img_idx} for cell ({y_idx},{x_idx}): {e_inner}", exc_info=True)
                         placeholder = self._create_placeholder_image(H, W, C, device, dtype) # Placeholder is already float32
                         final_grid[current_row_y:current_row_y + H, current_col_x:current_col_x + W, :] = placeholder
                         del placeholder
                         generation_successful = False
                    finally:
                         comfy.model_management.soft_empty_cache() # Clean up periodically

            # --- Label Drawing ---
            final_labeled_tensor = final_grid
            if draw_labels:
                logger.info("Drawing labels on grid...")
                x_axis_labels = [os.path.splitext(name)[0] if name != "No LoRA" else "No LoRA" for name in plot_loras]
                y_axis_labels = [f"{s:.3f}" for s in plot_strengths]
                try:
                    logger.debug("Transferring final grid to CPU for label drawing...")
                    # Grid is already float32, pass directly
                    grid_cpu = final_labeled_tensor.cpu()
                    labeled_grid_cpu = draw_labels_on_grid(
                        grid_cpu, x_labels=x_axis_labels, y_labels=y_axis_labels,
                        x_axis_label=x_axis_label, y_axis_label=y_axis_label
                    )
                    final_labeled_tensor = labeled_grid_cpu # Keep on CPU for return
                    logger.debug(f"Labels drawn. Final tensor shape: {final_labeled_tensor.shape}, Device: {final_labeled_tensor.device}")
                except Exception as e_label:
                     logger.error(f"Failed to draw labels on grid: {e_label}", exc_info=True)
                     final_labeled_tensor = final_grid.cpu() # Return unlabeled grid on CPU
            else:
                 logger.info("Label drawing skipped.")
                 final_labeled_tensor = final_grid.cpu() # Ensure CPU tensor for return

            # --- Final Output ---
            if final_labeled_tensor.device != torch.device('cpu'):
                 final_labeled_tensor = final_labeled_tensor.cpu() # Final safety check

            # --- FIX 2c: Ensure final output tensor is float32 ---
            final_output_tensor = final_labeled_tensor.float().unsqueeze(0) # Add batch dim and ensure float32
            # --- End FIX 2c ---

            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"--- XY Plot: Generation Complete (Duration: {duration}) ---")
            if not generation_successful:
                 logger.warning("Plot generation finished, but one or more images failed.")

            return (final_output_tensor,)

        except Exception as e:
            logger.critical(f"--- XY Plot: Generation FAILED: {e} ---", exc_info=True)
            raise RuntimeError(f"XY Plot generation failed: {e}") from e
        finally:
            # Clean up pre-loaded LoRAs from memory after execution finishes or fails
            del loaded_loras_cache
            comfy.model_management.soft_empty_cache()

# Note: Mappings are handled in xy_plotting/__init__.py
