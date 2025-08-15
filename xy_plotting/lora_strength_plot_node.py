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
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logging.error("❌ [XYPlot] Failed to import utility functions from grid_assembly or plot_utils. This node will likely fail. Please ensure all Divergent Nodes files are correctly installed.", exc_info=True)
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
                logger.warning(f"⚠️ [XYPlot] Latent batch (1) != Cond batch ({cond_batch_size}). Repeating latent sample to match conditioning batch size.")
                current_latent['samples'] = latent_samples.repeat(cond_batch_size, 1, 1, 1)
            else:
                # Force batch size 1 for latent if mismatch occurs and latent isn't already 1
                # This might be a less common case but handles potential user errors
                if latent_batch_size > 1:
                    logger.warning(f"⚠️ [XYPlot] Latent batch ({latent_batch_size}) != Cond batch ({cond_batch_size}). Using only the first latent sample for individual generation.")
                    current_latent['samples'] = latent_samples[0:1]
                else:
                    logger.warning(f"⚠️ [XYPlot] Latent batch ({latent_batch_size}) != Cond batch ({cond_batch_size}). This mismatch might lead to unexpected behavior or errors.")
        # Ensure batch size is 1 for the generation loop
        if current_latent['samples'].shape[0] > 1:
             logger.debug("🐛 [XYPlot] Ensuring latent batch size is 1 for individual image generation.")
             current_latent['samples'] = current_latent['samples'][0:1]

        return current_latent

    def _run_sampling_and_save_latent(self,
                                      model: ComfyModelObjectT,
                                      clip: ComfyCLIPObjectT,
                                      positive: ComfyConditioningT,
                                      negative: ComfyConditioningT,
                                      latent: ComfyLatentT,
                                      seed: int,
                                      steps: int,
                                      cfg: float,
                                      sampler_name: str,
                                      scheduler: str,
                                      temp_filepath: str
                                      ) -> ComfyLatentT:
        """
        Performs sampling and saves the resulting latent to a temporary file.
        
        Args:
            model (ComfyModelObjectT): The ComfyUI model object.
            clip (ComfyCLIPObjectT): The ComfyUI CLIP object.
            positive (ComfyConditioningT): Positive conditioning.
            negative (ComfyConditioningT): Negative conditioning.
            latent (ComfyLatentT): The input latent image dictionary.
            seed (int): The seed for random number generation.
            steps (int): Number of sampling steps.
            cfg (float): Classifier-free guidance scale.
            sampler_name (str): Name of the sampler to use.
            scheduler (str): Name of the scheduler to use.
            temp_filepath (str): Full path to save the latent tensor.

        Returns:
            ComfyLatentT: The generated latent dictionary.

        Raises:
            RuntimeError: If sampling or latent saving fails.
            ValueError: If sampler output is in an unexpected format.
        """
        logger.debug(f"🐛 [XYPlot] Starting sampling: {sampler_name}/{scheduler}, Steps: {steps}, CFG: {cfg}, Seed: {seed}")
        try:
            # Ensure latent batch size is 1 before preparing noise
            if latent['samples'].shape[0] != 1:
                 logger.warning(f"⚠️ [XYPlot] Sampler received latent batch size {latent['samples'].shape[0]}, expected 1. Using only the first sample for this step.")
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
                result_latent = samples_latent["samples"]
            elif isinstance(samples_latent, torch.Tensor):
                result_latent = samples_latent
            else:
                 raise ValueError(f"Sampler output unexpected format: {type(samples_latent)}. Expected dict with 'samples' or torch.Tensor.")

            logger.debug("🐛 [XYPlot] Sampling complete. Saving latent to temporary file...")
            # Save the latent tensor directly to disk (move to CPU to free GPU memory)
            torch.save(result_latent.cpu(), temp_filepath)
            logger.debug(f"🐛 [XYPlot] Saved latent to: {temp_filepath}")
            return {'samples': result_latent} # Return as ComfyLatentT dict for consistency
        except Exception as e:
            logger.error(f"❌ [XYPlot] Error during sampling or saving latent. This image will be a placeholder. Details: {e}", exc_info=True)
            raise RuntimeError("Sampling or latent saving failed.") from e

    def _load_latent_and_decode(self,
                                vae: ComfyVAEObjectT,
                                latent_filepath: str,
                                device: torch.device = torch.device('cpu')
                                ) -> TensorHWC:
        """
        Loads a latent tensor from file, decodes it using the VAE, and returns the image tensor.

        Args:
            vae (ComfyVAEObjectT): The ComfyUI VAE model object.
            latent_filepath (str): Full path to the temporary latent file.
            device (torch.device): The device to load the latent onto initially (e.g., 'cpu').

        Returns:
            TensorHWC: The decoded image tensor in [H, W, C] format.

        Raises:
            RuntimeError: If loading or decoding fails.
        """
        logger.debug(f"🐛 [XYPlot] Loading latent from {latent_filepath} and decoding...")
        try:
            latent_to_decode = torch.load(latent_filepath, map_location=device)
            if latent_to_decode.dim() == 3: # If saved without batch dim, add it
                latent_to_decode = latent_to_decode.unsqueeze(0)
            
            # Ensure latent is on the correct device for VAE decoding
            latent_to_decode = latent_to_decode.to(vae.device)

            img_tensor_chw = vae.decode(latent_to_decode)
            logger.debug(f"🐛 [XYPlot] Decoding complete. Shape: {img_tensor_chw.shape}")
            # Remove batch dim and permute: [C, H, W] -> [H, W, C]
            img_tensor_hwc = img_tensor_chw.squeeze(0).permute(1, 2, 0)
            return img_tensor_hwc # Return [H, W, C]
        except Exception as e:
            logger.error(f"❌ [XYPlot] Error decoding latent from {latent_filepath}. A placeholder image will be used for the grid. Details: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load or decode latent from {latent_filepath}.") from e

    def _create_placeholder_latent(self, base_latent: ComfyLatentT, device: torch.device) -> ComfyLatentT:
        """
        Creates a black placeholder latent tensor with the same shape as the base latent.

        Args:
            base_latent (ComfyLatentT): The original input latent dictionary, used for shape.
            device (torch.device): The device on which to create the placeholder tensor.

        Returns:
            ComfyLatentT: A dictionary containing the black placeholder latent tensor.

        Raises:
            RuntimeError: If placeholder latent creation fails.
        """
        try:
            latent_samples = base_latent['samples']
            # Create a zero tensor with the same shape and dtype as the original latent samples
            placeholder_latent_samples = torch.zeros_like(latent_samples, device=device)
            logger.info("ℹ️ [XYPlot] Created black placeholder latent due to generation error.")
            return {'samples': placeholder_latent_samples}
        except Exception as e_placeholder:
            logger.error(f"❌ [XYPlot] Failed to create placeholder latent. This is a critical error. Details: {e_placeholder}", exc_info=True)
            raise RuntimeError("Latent generation failed and placeholder latent creation also failed.") from e_placeholder

    def _create_placeholder_image(self, H: int, W: int, C: int, device: torch.device) -> TensorHWC:
        """
        Creates a black placeholder image tensor with specified dimensions.
        """
        return torch.zeros((H, W, C), dtype=torch.float32, device=device)

    def _generate_single_latent(
        self,
        base_model: ComfyModelObjectT,
        base_clip: ComfyCLIPObjectT,
        positive: ComfyConditioningT,
        negative: ComfyConditioningT,
        base_latent: ComfyLatentT,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        loaded_lora: LoadedLoraT,
        strength: float,
        img_index: int,
        lora_filename_part: str,
        temp_filepath: str
    ) -> str:
        """
        Generates a single latent tile using pre-loaded LoRA and saves it to a temporary file.
        Handles errors by creating and saving a placeholder latent.

        Args:
            base_model (ComfyModelObjectT): The base ComfyUI model object.
            base_clip (ComfyCLIPObjectT): The base ComfyUI CLIP object.
            positive (ComfyConditioningT): Positive conditioning.
            negative (ComfyConditioningT): Negative conditioning.
            base_latent (ComfyLatentT): The initial latent image dictionary.
            seed (int): The seed for random number generation.
            steps (int): Number of sampling steps.
            cfg (float): Classifier-free guidance scale.
            sampler_name (str): Name of the sampler to use.
            scheduler (str): Name of the scheduler to use.
            loaded_lora (LoadedLoraT): Pre-loaded LoRA tensor or None.
            strength (float): LoRA strength to apply.
            img_index (int): Index of the current image in the plot (for logging/seed).
            lora_filename_part (str): Cleaned LoRA filename for logging/temp file naming.
            temp_filepath (str): Full path to save the generated (or placeholder) latent.

        Returns:
            str: The filepath of the saved latent (either generated or placeholder).

        Raises:
            RuntimeError: If latent generation fails and a placeholder cannot be created/saved.
        """
        logger.info(f"✨ [XYPlot] Generating latent {img_index} (LoRA: '{lora_filename_part}', Strength: {strength:.3f})")
        current_model = None
        current_clip = None
        
        try:
            # 1. Clone base models
            current_model = base_model.clone()
            current_clip = base_clip.clone()

            # 2. Apply LoRA (if pre-loaded)
            if loaded_lora is not None:
                logger.debug(f"🐛 [XYPlot] Applying pre-loaded LoRA: {lora_filename_part} with strength {strength:.3f}")
                try:
                    current_model, current_clip = comfy.sd.load_lora_for_models(
                        current_model, current_clip, loaded_lora, strength, strength
                    )
                except Exception as e_apply:
                    logger.warning(f"⚠️ [XYPlot] Failed to apply pre-loaded LoRA '{lora_filename_part}'. Skipping this LoRA for the current generation. Error: {e_apply}", exc_info=True)
            else:
                logger.debug(f"🐛 [XYPlot] Skipping LoRA application ('{lora_filename_part}' not loaded or is baseline).")

            # 3. Prepare Latent (ensure batch size 1)
            current_latent = self._prepare_latent_for_sampling(base_latent, positive)

            # 4. Run Sampling and Save Latent
            self._run_sampling_and_save_latent(
                current_model, current_clip, positive, negative, current_latent,
                seed + img_index - 1, # Increment seed per image
                steps, cfg, sampler_name, scheduler,
                temp_filepath # Pass filepath for saving
            )
            return temp_filepath # Return path on success

        except Exception as e_generate:
            logger.error(f"❌ [XYPlot] ERROR generating latent {img_index} (LoRA: '{lora_filename_part}', Strength: {strength:.3f}). A placeholder will be used. Error: {e_generate}", exc_info=True)
            # Create and save placeholder latent on error
            try:
                # Ensure placeholder is created on CPU to avoid immediate GPU memory pressure
                device = torch.device('cpu') 

                placeholder_latent = self._create_placeholder_latent(base_latent, device)
                torch.save(placeholder_latent['samples'].cpu(), temp_filepath) # Save placeholder to disk
                logger.info(f"ℹ️ [XYPlot] Saved placeholder latent to: {temp_filepath}")
                return temp_filepath # Return path to placeholder
            except Exception as e_placeholder_fallback:
                logger.critical(f"🚨 [XYPlot] CRITICAL: Failed to determine placeholder dimensions or create/save placeholder latent. Plot generation may be severely affected. Details: {e_placeholder_fallback}", exc_info=True)
                # Re-raise the original generation error if placeholder fails
                raise RuntimeError("Failed to generate latent and could not create/save placeholder.") from e_generate

        finally: # This finally block belongs to the outer try in _generate_single_latent
            # Clean up clones
            del current_model
            del current_clip
            comfy.model_management.soft_empty_cache() # Ensure GPU memory is freed

    def _save_tensor_to_file(self,
                             image_tensor_hwc: TensorHWC,
                             filepath: str):
        """
        Saves a [H, W, C] tensor (decoded image) to a file using PIL.

        Args:
            image_tensor_hwc (TensorHWC): The image tensor to save, expected shape [H, W, C].
            filepath (str): The full path including filename and extension to save the image.

        Raises:
            IOError: If saving the image fails.
        """
        try:
            # Convert tensor to PIL Image
            img_tensor_float32 = image_tensor_hwc.float() # Ensure float32
            img_np = img_tensor_float32.cpu().numpy()
            img_pil = Image.fromarray(np.clip(img_np * 255.0, 0, 255).astype(np.uint8))
            # Save the image
            img_pil.save(filepath)
            logger.debug(f"🐛 [XYPlot] Saved decoded image tensor to: {filepath}")
        except Exception as e_save:
            logger.warning(f"⚠️ [XYPlot] Failed to save decoded image tensor to {filepath}. This individual image may be missing. Error: {e_save}", exc_info=True)
            # Raise the error so the main loop knows saving failed if needed
            raise IOError(f"Failed to save image to {filepath}") from e_save

    def _load_images_from_paths(self, image_paths: List[str], device: torch.device = torch.device('cpu')) -> List[TensorHWC]:
        """
        This function is no longer used for loading images for grid assembly.
        Images are now decoded one by one from latents.
        Keeping it as a placeholder or for other potential uses.
        """
        logger.warning("⚠️ [XYPlot] _load_images_from_paths is deprecated in this workflow and should not be called.")
        return [] # Return empty list as it's not used for grid assembly anymore

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
                # Ensure the base_latent passed to _prepare_latent_for_sampling is also just the first sample
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
                        saved_latent_path = self._generate_single_latent(
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
                         # If _generate_single_latent failed to save even a placeholder, this path won't be added.
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
                    decoded_img_tensor_hwc = self._load_latent_and_decode(vae, latent_path, device=vae_device)
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
                                self._save_tensor_to_file(decoded_img_tensor_hwc, perm_filepath)
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
                        placeholder_img = self._create_placeholder_image(H_img, W_img, C_img, torch.device('cpu'))
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
