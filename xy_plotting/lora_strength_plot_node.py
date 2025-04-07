"""
Node implementation for generating an XY plot comparing LoRA models vs. strength.
Refactored for memory efficiency and direct model/clip/vae inputs.
"""
import torch
import numpy as np
import os
import comfy.sd
import comfy.utils
import comfy.samplers
import comfy.model_management # Added for memory management
import folder_paths
from PIL import Image # Needed for saving individual images
from datetime import datetime
import re # For cleaning paths
import logging
from typing import Dict, Any, Tuple, Optional, List, Union, Sequence, TypeAlias

# Import grid assembly functions
try:
    # Only need draw_labels_on_grid now
    from .grid_assembly import draw_labels_on_grid
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logging.error("Failed to import draw_labels_on_grid from grid_assembly. Label drawing will fail.")
    def draw_labels_on_grid(*args: Any, **kwargs: Any) -> torch.Tensor: raise RuntimeError("grid_assembly not found")

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
LoadedLoraT: TypeAlias = Optional[Dict[str, torch.Tensor]] # Type for pre-loaded LoRA tensors

class LoraStrengthXYPlot:
    """
    Generates an XY plot grid comparing different LoRAs (X-axis)
    against varying model strengths (Y-axis), using provided models.

    Optimized to pre-load LoRAs and assemble the grid directly on the GPU
    to reduce peak memory usage.
    """
    CATEGORY = "👽 Divergent Nodes/XY Plots"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """Defines the input types for the ComfyUI node interface."""
        # --- Dynamic Lists (Samplers/Schedulers only) ---
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
                # Direct model inputs
                "model": ("MODEL", {"tooltip": "Input model."}),
                "clip": ("CLIP", {"tooltip": "Input CLIP model."}),
                "vae": ("VAE", {"tooltip": "Input VAE model."}),
                # LoRA and Plotting Params
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
                # Output/Display Options
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
        """Validates the LoRA folder path input."""
        logger.debug(f"Validating LoRA folder path input: '{lora_folder_path}'")
        clean_path = lora_folder_path.strip('\'" ')
        if os.path.isdir(clean_path):
            return os.path.abspath(clean_path)
        try:
            abs_lora_path = folder_paths.get_full_path("loras", clean_path)
            if abs_lora_path and os.path.isdir(abs_lora_path):
                logger.info(f"Resolved relative LoRA path '{clean_path}' to: {abs_lora_path}")
                return abs_lora_path
        except Exception as e:
            logger.warning(f"Error resolving path relative to loras folder: {e}", exc_info=True)
        if os.path.isabs(clean_path) and os.path.isdir(clean_path):
             return clean_path
        error_msg = f"LoRA folder path is not a valid directory: '{lora_folder_path}' (Checked: '{clean_path}')"
        logger.error(error_msg)
        raise ValueError(error_msg)

    def _get_lora_files(self, lora_folder_path: str) -> List[str]:
        """Scans the directory for valid LoRA filenames."""
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
        return lora_files

    def _determine_plot_axes(self,
                             lora_files: List[str],
                             x_lora_steps: int,
                             y_strength_steps: int,
                             max_strength: float
                             ) -> Tuple[List[str], List[float]]:
        """Determines the items (LoRAs and strengths) for the plot axes."""
        # --- X-Axis (LoRAs) ---
        plot_loras: List[str] = ["No LoRA"]
        num_available_loras: int = len(lora_files)
        if num_available_loras > 0:
            if x_lora_steps == 0:
                plot_loras.extend(lora_files)
            elif x_lora_steps == 1:
                 plot_loras.append(lora_files[-1])
            elif x_lora_steps > 1:
                if x_lora_steps >= num_available_loras:
                     plot_loras.extend(lora_files)
                else:
                    indices = np.linspace(0, num_available_loras - 1, num=x_lora_steps, dtype=int)
                    unique_indices = sorted(list(set(indices)))
                    for i in unique_indices:
                        if 0 <= i < num_available_loras:
                            plot_loras.append(lora_files[i])
        # --- Y-Axis (Strengths) ---
        num_strength_points = max(1, y_strength_steps)
        if num_strength_points == 1:
            plot_strengths: List[float] = [max_strength]
        else:
            plot_strengths = [ (i / num_strength_points) * max_strength for i in range(1, num_strength_points + 1) ]
        logger.info(f"Determined Grid Dimensions: {len(plot_strengths)} rows (Strengths), {len(plot_loras)} columns (LoRAs)")
        logger.debug(f"LoRAs to plot (X-axis): {plot_loras}")
        logger.debug(f"Strengths to plot (Y-axis): {[f'{s:.4f}' for s in plot_strengths]}")
        return plot_loras, plot_strengths

    def _preload_loras(self, lora_names: List[str], lora_folder_path: str) -> Dict[str, LoadedLoraT]:
        """Loads LoRA tensors from disk into memory."""
        loaded_loras: Dict[str, LoadedLoraT] = {}
        logger.info("Pre-loading LoRA files...")
        for name in lora_names:
            if name == "No LoRA":
                loaded_loras[name] = None # Placeholder for no LoRA
                continue
            lora_path = os.path.join(lora_folder_path, name)
            if not os.path.exists(lora_path):
                logger.warning(f"  LoRA file not found during pre-load: {lora_path}. Will skip.")
                loaded_loras[name] = None
                continue
            try:
                logger.debug(f"  Loading LoRA: {name}")
                # safe_load=True is important
                lora_tensor = comfy.utils.load_torch_file(lora_path, safe_load=True)
                loaded_loras[name] = lora_tensor
                logger.debug(f"  Successfully loaded LoRA: {name}")
            except Exception as e:
                logger.warning(f"  Failed to pre-load LoRA '{name}'. Will skip. Error: {e}", exc_info=True)
                loaded_loras[name] = None # Mark as failed to load
        logger.info("LoRA pre-loading complete.")
        return loaded_loras

    def _setup_output_directory(self, output_folder_name: str) -> Optional[str]:
        """Creates the output directory structure."""
        try:
            output_path = folder_paths.get_output_directory()
            safe_folder_name = re.sub(r'[\\/*?:"<>|]', '_', output_folder_name).strip()
            if not safe_folder_name: safe_folder_name = "XYPlot_Output"
            base_output_folder = os.path.join(output_path, safe_folder_name)
            run_folder = os.path.join(base_output_folder, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(run_folder, exist_ok=True)
            logger.info(f"Output directory for individual images prepared: {run_folder}")
            return run_folder
        except Exception as e:
             logger.error(f"Error setting up output directory '{output_folder_name}': {e}", exc_info=True)
             return None

    # --------------------------------------------------------------------------
    # Core Image Generation Logic
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
                                 latent: ComfyLatentT,
                                 seed: int,
                                 steps: int,
                                 cfg: float,
                                 sampler_name: str,
                                 scheduler: str
                                 ) -> torch.Tensor:
        """Performs sampling and VAE decoding."""
        logger.debug(f"  Starting sampling: {sampler_name}/{scheduler}, Steps: {steps}, CFG: {cfg}, Seed: {seed}")
        try:
            samples_latent = comfy.sample.sample(
                model, clip, vae, positive, negative, latent,
                seed=seed, steps=steps, cfg=cfg,
                sampler_name=sampler_name, scheduler=scheduler,
                denoise=1.0
            )
            if not isinstance(samples_latent, dict) or "samples" not in samples_latent:
                raise ValueError("Sampler output missing 'samples' key.")
            logger.debug("  Sampling complete. Decoding...")
            img_tensor_chw = vae.decode(samples_latent["samples"]) # [B, C, H, W]
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
            img_np = img_tensor_hwc.cpu().numpy()
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
            placeholder = torch.zeros((H, W, C), dtype=dtype, device=device)
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
        row_idx: int, # Pass grid indices for saving
        col_idx: int,
        lora_filename_part: str # Pass filename part for saving
    ) -> torch.Tensor:
        """
        Generates a single image tile using pre-loaded LoRA.
        Returns tensor [H, W, C].
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
                    logger.debug(f"  Successfully applied pre-loaded LoRA: {lora_filename_part}")
                except Exception as e_apply:
                    logger.warning(f"  Failed to apply pre-loaded LoRA '{lora_filename_part}'. Skipping. Error: {e_apply}", exc_info=True)
                    # Continue with the cloned base models if application fails
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
            # Need H, W, C etc. to create placeholder - try getting from latent
            try:
                 latent_shape = base_latent['samples'].shape # [B, C, H_lat, W_lat]
                 H_img, W_img = latent_shape[2] * 8, latent_shape[3] * 8
                 C_img = 3 # Assume RGB output
                 device = base_model.device # Get device from model
                 dtype = base_model.model.dtype # Get dtype from model
                 img_tensor_hwc = self._create_placeholder_image(H_img, W_img, C_img, device, dtype)
            except Exception as e_placeholder_fallback:
                 logger.critical(f"  Failed to determine placeholder dimensions or create placeholder: {e_placeholder_fallback}", exc_info=True)
                 raise RuntimeError("Failed to generate image and could not create placeholder.") from e_generate

        finally:
            # --- Clean up GPU Memory ---
            del current_model
            del current_clip
            # No need to delete loaded_lora here, it's managed outside the loop
            # Request garbage collection periodically (moved outside this func)
            # comfy.model_management.soft_empty_cache() # Moved to main loop
            # logger.debug("  GPU memory cleanup requested.")

        # Ensure we always return a valid tensor
        if img_tensor_hwc is None:
             logger.error("Image tensor was unexpectedly None after generation attempt. Creating final placeholder.")
             # Repeat placeholder creation logic as a last resort
             try:
                 latent_shape = base_latent['samples'].shape
                 H_img, W_img = latent_shape[2] * 8, latent_shape[3] * 8
                 C_img = 3
                 device = base_model.device
                 dtype = base_model.model.dtype
                 img_tensor_hwc = self._create_placeholder_image(H_img, W_img, C_img, device, dtype)
             except Exception:
                 raise RuntimeError("Failed to generate image and placeholder creation failed.")

        return img_tensor_hwc

    # --------------------------------------------------------------------------
    # Main Orchestration Method
    # --------------------------------------------------------------------------
    def generate_plot(self,
                      # Required inputs
                      model: ComfyModelObjectT, # Direct model input
                      clip: ComfyCLIPObjectT,   # Direct clip input
                      vae: ComfyVAEObjectT,     # Direct vae input
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
                      save_individual_images: bool = False,
                      output_folder_name: str = "XYPlot_LoRA-Strength",
                      row_gap: int = 0,
                      col_gap: int = 0,
                      draw_labels: bool = True,
                      x_axis_label: str = "",
                      y_axis_label: str = ""
                      ) -> Tuple[torch.Tensor]:
        """Orchestrates the XY plot generation with optimizations."""
        logger.info("--- Starting LoRA vs Strength XY Plot Generation (Optimized) ---")
        start_time = datetime.now()
        generation_successful = True # Track if all images generated ok

        try:
            # --- Setup ---
            validated_lora_path = self._validate_inputs(lora_folder_path)
            # Models (model, clip, vae) are now passed directly
            lora_files = self._get_lora_files(validated_lora_path)
            plot_loras, plot_strengths = self._determine_plot_axes(
                lora_files, x_lora_steps, y_strength_steps, max_strength
            )
            loaded_loras = self._preload_loras(plot_loras, validated_lora_path)

            num_rows = len(plot_strengths)
            num_cols = len(plot_loras)
            total_images = num_rows * num_cols
            if total_images == 0:
                 raise ValueError("Plot generation resulted in zero images based on inputs.")

            logger.info(f"Preparing to generate {total_images} images ({num_rows} rows x {num_cols} cols).")
            run_folder = self._setup_output_directory(output_folder_name) if save_individual_images else None

            # --- Determine Grid Geometry & Allocate Grid Tensor ---
            # Generate the first image to get dimensions
            first_img_index = 1
            first_lora_name = plot_loras[0]
            first_strength = plot_strengths[0]
            first_loaded_lora = loaded_loras.get(first_lora_name)
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
            comfy.model_management.soft_empty_cache() # Clean up after first image

            H, W, C = first_image_hwc.shape
            dtype = first_image_hwc.dtype
            device = first_image_hwc.device # Should be GPU if generated correctly
            logger.info(f"Image dimensions: {H}H x {W}W x {C}C, Type: {dtype}, Device: {device}")

            grid_height = H * num_rows + max(0, row_gap * (num_rows - 1))
            grid_width = W * num_cols + max(0, col_gap * (num_cols - 1))
            logger.info(f"Allocating final grid tensor: {grid_height}H x {grid_width}W x {C}C on device {device}")

            # Allocate the full grid tensor on the target device (GPU)
            # Initialize with a value that indicates empty/background if needed, e.g., 0 for black
            # Using zeros like the placeholder for consistency if errors occur
            final_grid = torch.zeros((grid_height, grid_width, C), dtype=dtype, device=device)

            # Paste the first image into the grid
            y_start, x_start = 0, 0 # Position for the first image (row 0, col 0)
            final_grid[y_start:y_start + H, x_start:x_start + W, :] = first_image_hwc
            del first_image_hwc # Free memory of the first image tensor
            logger.debug("Pasted first image into grid.")

            # --- Generation Loop (Optimized) ---
            logger.info("Starting main generation loop...")
            img_idx = 1 # Start from 1 (first image already done)
            for y_idx, strength in enumerate(plot_strengths):
                current_row_y = y_idx * (H + row_gap)
                for x_idx, lora_name in enumerate(plot_loras):
                    img_idx += 1
                    if y_idx == 0 and x_idx == 0:
                        continue # Skip the first image, already generated and pasted

                    current_col_x = x_idx * (W + col_gap)
                    loaded_lora = loaded_loras.get(lora_name)
                    lora_filename_part = os.path.splitext(lora_name)[0] if lora_name != "No LoRA" else "NoLoRA"

                    try:
                        # Generate image
                        img_tensor_hwc = self._generate_single_image(
                            base_model=model, base_clip=clip, base_vae=vae,
                            positive=positive, negative=negative, base_latent=latent_image,
                            seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
                            loaded_lora=loaded_lora, strength=strength,
                            run_folder=run_folder, save_individual_images=save_individual_images,
                            img_index=img_idx, row_idx=y_idx, col_idx=x_idx,
                            lora_filename_part=lora_filename_part
                        )

                        # Paste directly into the pre-allocated grid tensor
                        final_grid[current_row_y:current_row_y + H, current_col_x:current_col_x + W, :] = img_tensor_hwc
                        logger.debug(f"Pasted image {img_idx} into grid at ({current_row_y}, {current_col_x})")

                        # Immediately delete the tensor to free memory
                        del img_tensor_hwc

                    except Exception as e_inner:
                         logger.error(f"Error generating or pasting image {img_idx} for cell ({y_idx},{x_idx}): {e_inner}", exc_info=True)
                         # Create placeholder and paste it
                         placeholder = self._create_placeholder_image(H, W, C, device, dtype)
                         final_grid[current_row_y:current_row_y + H, current_col_x:current_col_x + W, :] = placeholder
                         del placeholder
                         generation_successful = False # Mark that at least one image failed

                    finally:
                         # Clean up GPU memory periodically (e.g., after each image)
                         comfy.model_management.soft_empty_cache()

            # --- Label Drawing ---
            final_labeled_tensor = final_grid # Start with the assembled grid
            if draw_labels:
                logger.info("Drawing labels on grid...")
                x_axis_labels = [os.path.splitext(name)[0] if name != "No LoRA" else "No LoRA" for name in plot_loras]
                y_axis_labels = [f"{s:.3f}" for s in plot_strengths]
                try:
                    # Transfer grid to CPU *once* for PIL operations
                    logger.debug("Transferring final grid to CPU for label drawing...")
                    grid_cpu = final_labeled_tensor.cpu()
                    # Ensure draw_labels_on_grid handles CPU tensor input correctly
                    # (The existing implementation converts to numpy/PIL, which requires CPU)
                    labeled_grid_cpu = draw_labels_on_grid(
                        grid_cpu, x_labels=x_axis_labels, y_labels=y_axis_labels,
                        x_axis_label=x_axis_label, y_axis_label=y_axis_label
                    )
                    # Transfer back to original device if needed, though output is usually CPU->ComfyUI
                    # Assuming ComfyUI expects CPU tensor for IMAGE output? Check this.
                    # If ComfyUI expects GPU, transfer back: final_labeled_tensor = labeled_grid_cpu.to(device)
                    final_labeled_tensor = labeled_grid_cpu # Keep on CPU for return
                    logger.debug(f"Labels drawn. Final tensor shape: {final_labeled_tensor.shape}, Device: {final_labeled_tensor.device}")
                except Exception as e_label:
                     logger.error(f"Failed to draw labels on grid: {e_label}", exc_info=True)
                     # Return the unlabeled grid (still on CPU after transfer attempt)
                     final_labeled_tensor = final_grid.cpu() # Ensure it's on CPU
            else:
                 logger.info("Label drawing skipped.")
                 # Ensure final tensor is on CPU for return if labels skipped
                 final_labeled_tensor = final_grid.cpu()


            # --- Final Output ---
            # Add batch dimension [1, H, W, C] - ComfyUI expects this format for IMAGE output
            # Ensure it's on CPU before unsqueezing if returning CPU tensor
            if final_labeled_tensor.device != torch.device('cpu'):
                 final_labeled_tensor = final_labeled_tensor.cpu() # Final safety check

            final_output_tensor = final_labeled_tensor.unsqueeze(0)

            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"--- XY Plot: Generation Complete (Duration: {duration}) ---")
            if not generation_successful:
                 logger.warning("Plot generation finished, but one or more images failed and were replaced by placeholders.")

            # Clean up pre-loaded LoRAs from memory
            del loaded_loras
            comfy.model_management.soft_empty_cache()

            return (final_output_tensor,)

        except Exception as e:
            logger.critical(f"--- XY Plot: Generation FAILED due to critical error: {e} ---", exc_info=True)
            # Clean up any potentially loaded LoRAs on critical failure
            if 'loaded_loras' in locals(): del loaded_loras
            comfy.model_management.soft_empty_cache()
            raise RuntimeError(f"XY Plot generation failed: {e}") from e

# Note: Mappings are handled in xy_plotting/__init__.py
