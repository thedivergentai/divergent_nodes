import torch
from PIL import Image
import numpy as np
import logging

from ..shared_utils.image_conversion import tensor_to_pil
from .musiq_scorer import MusiQScorer

logger = logging.getLogger(__name__)

class MusiQNode:
    """
    A ComfyUI node for scoring images using Google's MusiQ models (Aesthetic and Technical).
    """
    def __init__(self):
        self.musiq_scorer = MusiQScorer()
        self.model_urls = {
            "AVA": "https://tfhub.dev/google/musiq/ava/1",
            "KonIQ-10k": "https://tfhub.dev/google/musiq/koniq-10k/1",
            "SPAQ": "https://tfhub.dev/google/musiq/spaq/1",
            "PaQ-2-PiQ": "https://tfhub.dev/google/musiq/paq2piq/1"
        }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "aesthetic_model": (["AVA"], {"default": "AVA"}),
                "technical_model": (["KonIQ-10k", "SPAQ", "PaQ-2-PiQ"], {"default": "SPAQ"}),
                "score_aesthetic": ("BOOLEAN", {"default": True}),
                "score_technical": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING",)
    RETURN_NAMES = ("AESTHETIC_SCORE", "TECHNICAL_SCORE", "ERROR_MESSAGE",)
    FUNCTION = "score_image"
    CATEGORY = "Divergent AI 👽/Image"
    OUTPUT_NODE = True # This node primarily outputs scores, not images for further processing

    def score_image(self, image: torch.Tensor, aesthetic_model: str, technical_model: str, score_aesthetic: bool, score_technical: bool):
        aesthetic_score = 0.0
        technical_score = 0.0
        error_message = ""

        if not score_aesthetic and not score_technical:
            error_message = "ERROR: At least one scoring option (Aesthetic or Technical) must be enabled."
            logger.error(error_message)
            return (aesthetic_score, technical_score, error_message)

        try:
            # Convert ComfyUI tensor image to PIL Image
            pil_image = tensor_to_pil(image)
            
            aesthetic_model_url = self.model_urls.get(aesthetic_model) if score_aesthetic else None
            technical_model_url = self.model_urls.get(technical_model) if score_technical else None

            aesthetic_score, technical_score = self.musiq_scorer.get_scores(
                pil_image,
                score_aesthetic, aesthetic_model_url,
                score_technical, technical_model_url
            )

            if aesthetic_score == 0.0 and score_aesthetic:
                error_message += "Aesthetic scoring failed or model not loaded. "
            if technical_score == 0.0 and score_technical:
                error_message += "Technical scoring failed or model not loaded. "
            
            if error_message:
                logger.error(f"MusiQNode encountered issues: {error_message.strip()}")

        except Exception as e:
            error_message = f"ERROR: MusiQNode encountered an unexpected error: {e}"
            logger.error(error_message, exc_info=True)
            aesthetic_score = 0.0
            technical_score = 0.0

        return (aesthetic_score, technical_score, error_message.strip())
