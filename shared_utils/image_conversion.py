import torch
import numpy as np
from PIL import Image
import io
import base64
import sys # For fallback print
from typing import Optional, List, Union, Tuple, TypeAlias
import logging

# --- Type Aliases ---
PilImageT: TypeAlias = Image.Image

logger = logging.getLogger(__name__)
# Ensure handler is configured if root logger isn't set up
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Note: safe_print import is removed as it's not used in this file.
# If needed later, re-add the try-except block.


def tensor_to_pil(tensor: Optional[torch.Tensor]) -> Optional[PilImageT]:
    """
    Converts a ComfyUI IMAGE tensor (Batch, Height, Width, Channel) to a single PIL Image.

    Takes the first image from the batch (index 0). Assumes input tensor values
    are floats in the range [0.0, 1.0] and converts them to uint8 [0, 255].
    Handles basic dimension and dtype validation.

    Args:
        tensor (Optional[torch.Tensor]): The input torch tensor, expected shape
                                         [B, H, W, C] and dtype float32/64, or None.

    Returns:
        Optional[PilImageT]: A PIL.Image object on success, None if conversion fails
                             or input is None/invalid.
    """
    if tensor is None:
        logger.debug("tensor_to_pil received None input.")
        return None
    if not isinstance(tensor, torch.Tensor):
        logger.error(f"Input is not a torch.Tensor, but {type(tensor)}. Cannot convert to PIL.")
        return None
    if tensor.ndim != 4:
        logger.error(f"Input tensor has incorrect dimensions ({tensor.ndim}). Expected 4 (B, H, W, C).")
        return None
    if tensor.shape[0] == 0: # Check if batch dimension is empty
        logger.error("Input tensor batch dimension is empty (size 0).")
        return None

    # Process the first image in the batch
    try:
        # Get first image, ensure CPU, detach from graph
        img_tensor_slice = tensor[0].detach().cpu()
        logger.debug(f"Processing tensor slice with shape: {img_tensor_slice.shape} and dtype: {img_tensor_slice.dtype}")

        # Convert to numpy array
        img_np: np.ndarray = img_tensor_slice.numpy()

        # Handle dtype conversion and range scaling (common case: float [0,1] -> uint8 [0,255])
        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
            if np.min(img_np) < 0.0 or np.max(img_np) > 1.0:
                 logger.warning(f"Input float tensor values outside expected [0, 1] range (min: {np.min(img_np):.2f}, max: {np.max(img_np):.2f}). Clipping.")
            img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
            logger.debug("Converted float tensor to uint8 [0, 255].")
        elif img_np.dtype == np.uint8:
            logger.debug("Input tensor is already uint8.")
            # Pass through if already uint8
        else:
            logger.warning(f"Unexpected tensor dtype {img_np.dtype}. Attempting direct cast to uint8. Data range might be incorrect.")
            try:
                # Attempt direct cast, but be aware this might not be visually correct
                # if the original range wasn't [0, 255]
                img_np = img_np.astype(np.uint8)
            except ValueError as e:
                 logger.error(f"Error converting tensor dtype {img_np.dtype} to uint8: {e}", exc_info=True)
                 return None # Cannot proceed if cast fails

        # Handle channel dimension for PIL (expecting H, W, C or H, W)
        if img_np.ndim == 3 and img_np.shape[2] == 1: # Grayscale image with channel dim
            img_np = np.squeeze(img_np, axis=2) # Convert [H, W, 1] to [H, W]
            logger.debug("Squeezed grayscale tensor [H, W, 1] to [H, W] for PIL.")
        elif img_np.ndim == 2: # Grayscale [H, W]
            logger.debug("Input is 2D grayscale numpy array.")
        elif img_np.ndim == 3 and img_np.shape[2] in [3, 4]: # RGB or RGBA [H, W, C]
            logger.debug(f"Input is {img_np.shape[2]}-channel numpy array.")
        else:
             logger.error(f"Unsupported numpy array shape for PIL conversion: {img_np.shape}. Expected HWC with C=1,3,4 or HW.")
             return None

        # Convert numpy array to PIL Image
        pil_image: PilImageT = Image.fromarray(img_np)
        logger.debug(f"Successfully converted tensor slice to PIL Image (mode: {pil_image.mode}).")
        return pil_image

    except IndexError:
        # This case should be caught by the initial batch size check, but included for safety
        logger.error("IndexError: Input tensor batch dimension might be empty (should have been caught earlier).")
        return None
    except Exception as e:
        logger.error(f"Unexpected error converting tensor slice to PIL Image: {e}", exc_info=True)
        return None


def pil_to_base64(pil_image: Optional[PilImageT], format: str = "JPEG") -> Optional[str]:
    """
    Converts a PIL Image object to a Base64 encoded string.

    Handles potential transparency issues when saving to formats like JPEG.

    Args:
        pil_image (Optional[PilImageT]): The PIL.Image object to convert.
        format (str): The target image format (e.g., "JPEG", "PNG", "WEBP").
                      Defaults to "JPEG". Case-insensitive.

    Returns:
        Optional[str]: A Base64 encoded string representing the image (without data URI prefix),
                       or None if conversion fails or input is None.
    """
    if not pil_image:
        logger.debug("pil_to_base64 received None input.")
        return None
    if not isinstance(pil_image, Image.Image):
        logger.error(f"Input is not a PIL.Image, but {type(pil_image)}. Cannot convert to Base64.")
        return None

    logger.debug(f"Attempting to convert PIL Image (mode: {pil_image.mode}) to Base64 with format: {format}")
    try:
        buffer = io.BytesIO()
        img_format: str = format.upper()
        save_image = pil_image # Start with the original image

        # Handle transparency: Convert RGBA/P modes to RGB if saving as JPEG
        # JPEG does not support transparency.
        if save_image.mode in ['RGBA', 'P'] and img_format in ['JPEG', 'JPG']:
            logger.warning(f"Image mode is {save_image.mode} but saving as {img_format}. Transparency will be lost. Converting to RGB with white background.")
            # Create a white background image matching the original size
            bg = Image.new("RGB", save_image.size, (255, 255, 255))
            try:
                # Paste the image onto the background.
                # If the image has an alpha channel, use it as the mask.
                # If it's mode 'P' with transparency info, convert first.
                img_to_paste = save_image
                if save_image.mode == 'P':
                     # Ensure palette transparency is handled if present
                     img_to_paste = save_image.convert('RGBA')

                if img_to_paste.mode == 'RGBA':
                     bg.paste(img_to_paste, mask=img_to_paste.split()[-1]) # Use alpha channel
                else: # Should be RGB after conversion, paste directly
                     bg.paste(img_to_paste)
                save_image = bg # Use the image pasted onto the background
                logger.debug("Converted RGBA/P image to RGB for JPEG saving.")
            except Exception as paste_e:
                 logger.error(f"Error handling transparency during conversion to RGB for JPEG: {paste_e}", exc_info=True)
                 # Fallback: try saving the original image directly, might raise error in save()
                 save_image = pil_image

        # Save image to buffer
        save_kwargs = {}
        if img_format == 'JPEG':
             save_kwargs['quality'] = 95 # Set default JPEG quality
             save_kwargs['optimize'] = True
        elif img_format == 'WEBP':
             save_kwargs['quality'] = 90
             save_kwargs['lossless'] = False

        logger.debug(f"Saving PIL image to buffer with format {img_format} and options {save_kwargs}")
        save_image.save(buffer, format=img_format, **save_kwargs)
        img_bytes: bytes = buffer.getvalue()
        buffer.close() # Close the buffer

        # Encode bytes to Base64 string
        base64_string: str = base64.b64encode(img_bytes).decode('utf-8')
        logger.info(f"Successfully converted PIL image to Base64 ({img_format}, {len(base64_string)} chars).")
        return base64_string

    except Exception as e:
        logger.error(f"Error converting PIL image to Base64 (format: {format}): {e}", exc_info=True)
        return None
