import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set default logging level for this module

# Global cache for MusiQ models to avoid re-downloading
_musiq_model_cache = {}

class MusiQScorer:
    """
    Handles loading and interacting with the MusiQ TensorFlow Hub models (aesthetic and technical).
    Models are cached globally to prevent repeated downloads.
    """
    def __init__(self):
        pass # No need for a queue here, as it's not a GUI application

    def _load_model(self, model_url: str, model_type: str):
        """
        Loads a MusiQ model from TensorFlow Hub, using a global cache.
        """
        if model_url in _musiq_model_cache:
            logger.info(f"{model_type} MusiQ model already loaded from cache: {model_url}")
            return _musiq_model_cache[model_url]

        logger.info(f"Downloading and loading {model_type} MusiQ model from {model_url}...")
        try:
            model = hub.load(model_url)
            _musiq_model_cache[model_url] = model
            logger.info(f"{model_type} MusiQ model loaded successfully from {model_url}.")
            return model
        except Exception as e:
            logger.error(f"Failed to load {model_type} MusiQ model from {model_url}: {e}", exc_info=True)
            return None

    def get_scores(self, pil_image: Image.Image, aesthetic_enabled: bool, aesthetic_model_url: str, technical_enabled: bool, technical_model_url: str):
        """
        Gets aesthetic and/or technical scores for a PIL Image using the MusiQ models.
        Expects a PIL Image as input. Returns (aesthetic_score, technical_score).
        Returns 0.0 for disabled models or on error.
        """
        aesthetic_score = 0.0
        technical_score = 0.0

        # Convert PIL Image to bytes for TensorFlow model input
        # The MusiQ model expects raw image bytes, not a processed tensor
        try:
            # Save PIL image to a BytesIO object as JPEG (or PNG if transparency is needed, but JPEG is common)
            # This simulates tf.io.read_file(image_path)
            from io import BytesIO
            img_byte_arr = BytesIO()
            pil_image.save(img_byte_arr, format='JPEG') # Use JPEG for efficiency and common compatibility
            image_bytes = tf.constant(img_byte_arr.getvalue(), dtype=tf.string)
        except Exception as e:
            logger.error(f"Error converting PIL Image to bytes: {e}", exc_info=True)
            return 0.0, 0.0 # Indicate error

        if aesthetic_enabled and aesthetic_model_url:
            aesthetic_model = self._load_model(aesthetic_model_url, "Aesthetic")
            if aesthetic_model:
                try:
                    aesthetic_score_tensor = aesthetic_model.signatures['serving_default'](image_bytes_tensor=image_bytes)['output_0']
                    aesthetic_score = float(aesthetic_score_tensor.numpy())
                    logger.info(f"Successfully scored aesthetic: {aesthetic_score:.4f}")
                except Exception as e:
                    logger.error(f"Error scoring aesthetic: {e}", exc_info=True)
                    aesthetic_score = 0.0 # Indicate error
            else:
                logger.warning("Aesthetic model not loaded, skipping aesthetic scoring.")
                aesthetic_score = 0.0
        
        if technical_enabled and technical_model_url:
            technical_model = self._load_model(technical_model_url, "Technical")
            if technical_model:
                try:
                    technical_score_tensor = technical_model.signatures['serving_default'](image_bytes_tensor=image_bytes)['output_0']
                    technical_score = float(technical_score_tensor.numpy())
                    logger.info(f"Successfully scored technical: {technical_score:.4f}")
                except Exception as e:
                    logger.error(f"Error scoring technical: {e}", exc_info=True)
                    technical_score = 0.0 # Indicate error
            else:
                logger.warning("Technical model not loaded, skipping technical scoring.")
                technical_score = 0.0

        return aesthetic_score, technical_score
