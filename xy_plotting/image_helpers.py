import torch
import numpy as np
import logging
from PIL import Image
from typing import TypeAlias

# Define type hints
TensorHWC: TypeAlias = torch.Tensor # Expected shape [H, W, C]

# Setup logger
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def save_tensor_to_file(image_tensor_hwc: TensorHWC, filepath: str):
    """
    Saves a [H, W, C] tensor (decoded image) to a file using PIL.

    Args:
        image_tensor_hwc (TensorHWC): The image tensor to save, expected shape [H, W, C].
        filepath (str): The full path including filename and extension to save the image.

    Raises:
        IOError: If saving the image fails.
    """
    try:
        img_tensor_float32 = image_tensor_hwc.float()
        img_np = img_tensor_float32.cpu().numpy()
        img_pil = Image.fromarray(np.clip(img_np * 255.0, 0, 255).astype(np.uint8))
        img_pil.save(filepath)
        logger.debug(f"🐛 [XYPlot] Saved decoded image tensor to: {filepath}")
    except Exception as e_save:
        logger.warning(f"⚠️ [XYPlot] Failed to save decoded image tensor to {filepath}. This individual image may be missing. Error: {e_save}", exc_info=True)
        raise IOError(f"Failed to save image to {filepath}") from e_save
