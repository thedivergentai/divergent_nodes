import torch
from PIL import Image
import numpy as np
import logging
from typing import Tuple, Dict, Any, List

# Import ComfyUI types for better type hinting and autocomplete
from comfy.comfy_types import ComfyNodeABC, IO, InputTypeDict

from ..shared_utils.image_conversion import tensor_to_pil
from ..shared_utils.logging_utils import SUCCESS_HIGHLIGHT # Import custom log level
from .musiq_scorer import MusiQScorer

logger = logging.getLogger(__name__)

class MusiQNode(ComfyNodeABC): # Inherit from ComfyNodeABC
    """
    A ComfyUI node for scoring images using Google's MusiQ models (Aesthetic and Technical).
    """
    def __init__(self):
        self.musiq_scorer = MusiQScorer()
        self.model_urls: Dict[str, str] = { # Add type hint for model_urls
            "AVA": "https://www.kaggle.com/models/google/musiq/frameworks/TensorFlow2/variations/ava/versions/1",
            "KonIQ-10k": "https://www.kaggle.com/models/google/musiq/frameworks/TensorFlow2/variations/koniq-10k/versions/1",
            "SPAQ": "https://www.kaggle.com/models/google/musiq/frameworks/TensorFlow2/variations/spaq/versions/1",
            "PaQ-2-PiQ": "https://www.kaggle.com/models/google/musiq/frameworks/TensorFlow2/variations/paq2piq/versions/1"
        }

    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict: # Use InputTypeDict for type hinting
        return {
            "required": {
                "image": (IO.IMAGE,), # Use IO.IMAGE
                "aesthetic_model": (["AVA"], {"default": "AVA", "tooltip": "The aesthetic model to use (currently only AVA)."}),
                "technical_model": (["KonIQ-10k", "SPAQ", "PaQ-2-PiQ"], {"default": "KonIQ-10k", "tooltip": "The technical model to use."}),
                "score_aesthetic": (IO.BOOLEAN, {"default": True, "tooltip": "Enable/disable aesthetic scoring."}), # Use IO.BOOLEAN
                "score_technical": (IO.BOOLEAN, {"default": False, "tooltip": "Enable/disable technical scoring."}), # Use IO.BOOLEAN
            }
        }

    RETURN_TYPES: Tuple[str] = (IO.FLOAT, IO.FLOAT, IO.INT, IO.INT, IO.STRING,) # Use IO types
    RETURN_NAMES: Tuple[str] = ("AESTHETIC_SCORE", "TECHNICAL_SCORE", "FINAL_AVERAGE_SCORE_OUT_OF_10", "FINAL_AVERAGE_SCORE_OUT_OF_100", "ERROR_MESSAGE",)
    FUNCTION: str = "score_image" # Use type hint for FUNCTION
    CATEGORY: str = "Divergent Nodes 👽/MusiQ" # Keep category consistent
    OUTPUT_NODE: bool = True # Use type hint for OUTPUT_NODE

    def score_image(self, image: torch.Tensor, aesthetic_model: str, technical_model: str, score_aesthetic: bool, score_technical: bool) -> Tuple[float, float, int, int, str]:
        aesthetic_score = 0.0
        technical_score = 0.0
        final_average_score_10 = 0
        final_average_score_100 = 0
        error_message = ""

        logger.info("MusiQ Node: Starting image scoring.")

        if not score_aesthetic and not score_technical:
            error_message = "ERROR: At least one scoring option (Aesthetic or Technical) must be enabled."
            logger.error(error_message)
            return (aesthetic_score, technical_score, final_average_score_10, final_average_score_100, error_message)

        try:
            # Convert ComfyUI tensor image to PIL Image
            pil_image: Image.Image = tensor_to_pil(image)
            
            aesthetic_model_url = self.model_urls.get(aesthetic_model) if score_aesthetic else None
            technical_model_url = self.model_urls.get(technical_model) if score_technical else None

            logger.debug(f"Aesthetic scoring enabled: {score_aesthetic}, Technical scoring enabled: {score_technical}")
            logger.debug(f"Aesthetic model URL: {aesthetic_model_url}, Technical model URL: {technical_model_url}")

            aesthetic_score, technical_score = self.musiq_scorer.get_scores(
                pil_image,
                score_aesthetic, aesthetic_model_url,
                score_technical, technical_model_url
            )

            if score_aesthetic and aesthetic_score == 0.0:
                error_message += "Aesthetic scoring failed or model not loaded. "
                logger.warning("Aesthetic score is 0.0 despite being enabled. Check MusiQScorer logs.")
            if score_technical and technical_score == 0.0:
                error_message += "Technical scoring failed or model not loaded. "
                logger.warning("Technical score is 0.0 despite being enabled. Check MusiQScorer logs.")
            
            # Calculate final average score based on enabled options and correct scaling
            average_score_out_of_100 = 0.0
            
            if score_aesthetic and score_technical:
                # Both enabled: normalize aesthetic to 100, then average 50/50
                normalized_aesthetic_score = aesthetic_score * 10 # Scale 0-10 to 0-100
                if normalized_aesthetic_score != 0.0 and technical_score != 0.0:
                    average_score_out_of_100 = (normalized_aesthetic_score + technical_score) / 2
                elif normalized_aesthetic_score != 0.0: # If only aesthetic succeeded
                    average_score_out_of_100 = normalized_aesthetic_score
                elif technical_score != 0.0: # If only technical succeeded
                    average_score_out_of_100 = technical_score
            elif score_aesthetic:
                # Only aesthetic enabled: scale aesthetic to 100
                average_score_out_of_100 = aesthetic_score * 10
            elif score_technical:
                # Only technical enabled: use technical score directly (already out of 100)
                average_score_out_of_100 = technical_score
            
            final_average_score_10 = int(round(average_score_out_of_100 / 10))
            final_average_score_100 = int(round(average_score_out_of_100))
            
            if error_message:
                logger.error(f"MusiQNode encountered issues: {error_message.strip()}")
            else:
                logger.log(SUCCESS_HIGHLIGHT, f"Image scored successfully. Aesthetic: {aesthetic_score:.2f}, Technical: {technical_score:.2f}, Final: {final_average_score_100} (out of 100).")

        except Exception as e:
            error_message = f"ERROR: MusiQNode encountered an unexpected error: {e}"
            logger.error(error_message, exc_info=True)
            aesthetic_score = 0.0
            technical_score = 0.0
            final_average_score_10 = 0
            final_average_score_100 = 0

        logger.info("MusiQ Node: Execution finished.")
        return (aesthetic_score, technical_score, final_average_score_10, final_average_score_100, error_message.strip())
