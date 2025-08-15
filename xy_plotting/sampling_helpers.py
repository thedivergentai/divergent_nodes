import torch
import comfy.sample
import comfy.sd
import comfy.model_management
import logging
import os
from typing import Dict, Any, Tuple, Optional, List, Union, Sequence, TypeAlias
from PIL import Image

# Define type hints
ComfyConditioningT: TypeAlias = List[Tuple[torch.Tensor, Dict[str, Any]]]
ComfyCLIPObjectT: TypeAlias = Any
ComfyVAEObjectT: TypeAlias = Any
ComfyModelObjectT: TypeAlias = Any
ComfyLatentT: TypeAlias = Dict[str, torch.Tensor]
LoadedLoraT: TypeAlias = Optional[Dict[str, torch.Tensor]]
TensorHWC: TypeAlias = torch.Tensor # Expected shape [H, W, C]

# Setup logger
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def prepare_latent_for_sampling(base_latent: ComfyLatentT, positive_cond: ComfyConditioningT) -> ComfyLatentT:
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
            if latent_batch_size > 1:
                logger.warning(f"⚠️ [XYPlot] Latent batch ({latent_batch_size}) != Cond batch ({cond_batch_size}). Using only the first latent sample for individual generation.")
                current_latent['samples'] = latent_samples[0:1]
            else:
                logger.warning(f"⚠️ [XYPlot] Latent batch ({latent_batch_size}) != Cond batch ({cond_batch_size}). This mismatch might lead to unexpected behavior or errors.")
    if current_latent['samples'].shape[0] > 1:
         logger.debug("🐛 [XYPlot] Ensuring latent batch size is 1 for individual image generation.")
         current_latent['samples'] = current_latent['samples'][0:1]
    return current_latent

def run_sampling_and_save_latent(model: ComfyModelObjectT, clip: ComfyCLIPObjectT,
                                 positive: ComfyConditioningT, negative: ComfyConditioningT,
                                 latent: ComfyLatentT, seed: int, steps: int, cfg: float,
                                 sampler_name: str, scheduler: str, temp_filepath: str) -> ComfyLatentT:
    """
    Performs sampling and saves the resulting latent to a temporary file.
    """
    logger.debug(f"🐛 [XYPlot] Starting sampling: {sampler_name}/{scheduler}, Steps: {steps}, CFG: {cfg}, Seed: {seed}")
    try:
        if latent['samples'].shape[0] != 1:
             logger.warning(f"⚠️ [XYPlot] Sampler received latent batch size {latent['samples'].shape[0]}, expected 1. Using only the first sample for this step.")
             latent['samples'] = latent['samples'][0:1]

        noise = comfy.sample.prepare_noise(latent['samples'], seed)

        samples_latent = comfy.sample.sample(
            model=model, noise=noise, seed=seed, steps=steps, cfg=cfg,
            sampler_name=sampler_name, scheduler=scheduler,
            positive=positive, negative=negative, latent_image=latent['samples'], denoise=1.0
        )

        if isinstance(samples_latent, dict) and "samples" in samples_latent:
            result_latent = samples_latent["samples"]
        elif isinstance(samples_latent, torch.Tensor):
            result_latent = samples_latent
        else:
             raise ValueError(f"Sampler output unexpected format: {type(samples_latent)}. Expected dict with 'samples' or torch.Tensor.")

        logger.debug("🐛 [XYPlot] Sampling complete. Saving latent to temporary file...")
        torch.save(result_latent.cpu(), temp_filepath)
        logger.debug(f"🐛 [XYPlot] Saved latent to: {temp_filepath}")
        return {'samples': result_latent}
    except Exception as e:
        logger.error(f"❌ [XYPlot] Error during sampling or saving latent. This image will be a placeholder. Details: {e}", exc_info=True)
        raise RuntimeError("Sampling or latent saving failed.") from e

def load_latent_and_decode(vae: ComfyVAEObjectT, latent_filepath: str, device: torch.device = torch.device('cpu')) -> TensorHWC:
    """
    Loads a latent tensor from file, decodes it using the VAE, and returns the image tensor.
    """
    logger.debug(f"🐛 [XYPlot] Loading latent from {latent_filepath} and decoding...")
    try:
        latent_to_decode = torch.load(latent_filepath, map_location=device)
        if latent_to_decode.dim() == 3:
            latent_to_decode = latent_to_decode.unsqueeze(0)
        
        latent_to_decode = latent_to_decode.to(vae.device)

        img_tensor_chw = vae.decode(latent_to_decode)
        logger.debug(f"🐛 [XYPlot] Decoding complete. Shape: {img_tensor_chw.shape}")
        img_tensor_hwc = img_tensor_chw.squeeze(0).permute(1, 2, 0)
        return img_tensor_hwc
    except Exception as e:
        logger.error(f"❌ [XYPlot] Error decoding latent from {latent_filepath}. A placeholder image will be used for the grid. Details: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load or decode latent from {latent_filepath}.") from e

def create_placeholder_latent(base_latent: ComfyLatentT, device: torch.device) -> ComfyLatentT:
    """
    Creates a black placeholder latent tensor with the same shape as the base latent.
    """
    try:
        latent_samples = base_latent['samples']
        placeholder_latent_samples = torch.zeros_like(latent_samples, device=device)
        logger.info("ℹ️ [XYPlot] Created black placeholder latent due to generation error.")
        return {'samples': placeholder_latent_samples}
    except Exception as e_placeholder:
        logger.error(f"❌ [XYPlot] Failed to create placeholder latent. This is a critical error. Details: {e_placeholder}", exc_info=True)
        raise RuntimeError("Latent generation failed and placeholder latent creation also failed.") from e_placeholder

def create_placeholder_image(H: int, W: int, C: int, device: torch.device) -> TensorHWC:
    """
    Creates a black placeholder image tensor with specified dimensions.
    """
    return torch.zeros((H, W, C), dtype=torch.float32, device=device)

def generate_single_latent(
    base_model: ComfyModelObjectT, base_clip: ComfyCLIPObjectT,
    positive: ComfyConditioningT, negative: ComfyConditioningT,
    base_latent: ComfyLatentT, seed: int, steps: int, cfg: float,
    sampler_name: str, scheduler: str, loaded_lora: LoadedLoraT,
    strength: float, img_index: int, lora_filename_part: str, temp_filepath: str
) -> str:
    """
    Generates a single latent tile using pre-loaded LoRA and saves it to a temporary file.
    Handles errors by creating and saving a placeholder latent.
    """
    logger.info(f"✨ [XYPlot] Generating latent {img_index} (LoRA: '{lora_filename_part}', Strength: {strength:.3f})")
    current_model = None
    current_clip = None
    
    try:
        current_model = base_model.clone()
        current_clip = base_clip.clone()

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

        current_latent = prepare_latent_for_sampling(base_latent, positive)

        run_sampling_and_save_latent(
            current_model, current_clip, positive, negative, current_latent,
            seed + img_index - 1, steps, cfg, sampler_name, scheduler,
            temp_filepath
        )
        return temp_filepath

    except Exception as e_generate:
        logger.error(f"❌ [XYPlot] ERROR generating latent {img_index} (LoRA: '{lora_filename_part}', Strength: {strength:.3f}). A placeholder will be used. Error: {e_generate}", exc_info=True)
        try:
            device = torch.device('cpu') 
            placeholder_latent = create_placeholder_latent(base_latent, device)
            torch.save(placeholder_latent['samples'].cpu(), temp_filepath)
            logger.info(f"ℹ️ [XYPlot] Saved placeholder latent to: {temp_filepath}")
            return temp_filepath
        except Exception as e_placeholder_fallback:
            logger.critical(f"🚨 [XYPlot] CRITICAL: Failed to determine placeholder dimensions or create/save placeholder latent. Plot generation may be severely affected. Details: {e_placeholder_fallback}", exc_info=True)
            raise RuntimeError("Failed to generate latent and could not create/save placeholder.") from e_generate

    finally:
        del current_model
        del current_clip
        comfy.model_management.soft_empty_cache()
