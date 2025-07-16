"""
ComfyUI node for interacting with the Google Gemini API.
Supports text generation and multimodal input (text + image).
Requires a GEMINI_API_KEY environment variable.
"""
import logging
import torch
import io
import time # Import time for caching
from typing import Optional, Dict, Tuple, Any, List
from PIL import Image

# Import necessary functions and constants from the new utils module
from .gemini_utils import (
    configure_api_key,
    prepare_safety_settings,
    prepare_generation_config,
    prepare_content_parts,
    generate_content,
    prepare_thinking_config, # New import for thinking config
    SAFETY_SETTINGS_MAP,
    SAFETY_THRESHOLD_TO_NAME,
    ERROR_PREFIX,
    google_exceptions
)

# Import shared utilities
from ..shared_utils.text_encoding_utils import ensure_utf8_friendly
from ..shared_utils.image_conversion import tensor_to_pil

# Import genai and types for direct API interaction
from google import genai
from google.genai import types
from google.generativeai import GenerativeModel # Corrected import for GenerativeModel
from google.generativeai.client import Client # For dynamic model listing

# Setup logger for this module
logger = logging.getLogger(__name__)
# Ensure handler is configured if root logger isn't set up
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- ComfyUI Node Definition ---

class GeminiNode:
    """
    A ComfyUI node to interact with the Google Gemini API for text generation,
    optionally using an image input for multimodal models.

    Requires a `GEMINI_API_KEY` environment variable (e.g., in a `.env` file
    in the ComfyUI root or a parent directory). It dynamically fetches the list
    of available models supporting content generation when ComfyUI loads the node.
    """
    # Define a static list of available models. This list is hardcoded to ensure
    # the node loads without any API calls or key checks at startup.
    AVAILABLE_MODELS = [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite-preview-06-17",
    ]
    SAFETY_OPTIONS = list(SAFETY_SETTINGS_MAP.keys())

    # Define ComfyUI node attributes
    RETURN_TYPES: Tuple[str] = ("STRING",)
    RETURN_NAMES: Tuple[str] = ("text",)
    FUNCTION: str = "generate"
    CATEGORY: str = "Divergent Nodes 👽/Gemini"

    _model_cache: List[str] = []
    _last_cache_update: float = 0
    _CACHE_LIFETIME_SECONDS: int = 3600 # Cache models for 1 hour

    def __init__(self):
        """Initializes the Gemini node instance. No API calls are made here."""
        logger.debug("GeminiNode instance created.")

    @classmethod
    def _get_cached_models(cls, api_key: str) -> List[str]:
        """
        Fetches and caches available Gemini models.
        Models are cached for _CACHE_LIFETIME_SECONDS to avoid excessive API calls.
        """
        current_time = time.time()
        if not cls._model_cache or (current_time - cls._last_cache_update > cls._CACHE_LIFETIME_SECONDS):
            logger.info("Refreshing Gemini model list cache...")
            try:
                client = Client(api_key=api_key)
                # Filter for models that support generateContent (text and multimodal)
                # and exclude tuned models for simplicity in this list
                models = [
                    m.name for m in client.models.list()
                    if "generateContent" in m.supported_actions and not m.name.startswith("tunedModels/")
                ]
                # Sort models alphabetically for consistent display
                models.sort()
                cls._model_cache = models
                cls._last_cache_update = current_time
                logger.info(f"Refreshed model list. Found {len(models)} models.")
            except Exception as e:
                logger.error(f"Failed to fetch dynamic model list: {e}", exc_info=True)
                # Fallback to hardcoded models if API call fails
                if not cls._model_cache: # Only use hardcoded if cache is empty
                    cls._model_cache = cls.AVAILABLE_MODELS
                    logger.warning("Using hardcoded model list due to API error.")
        return cls._model_cache

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """
        Defines the input types and options for the ComfyUI node interface.
        This method is called by ComfyUI to build the node's UI.
        It is static and does not perform any API calls.
        """
        # Determine default safety setting names using constants from utils
        default_safety = SAFETY_THRESHOLD_TO_NAME.get("BLOCK_MEDIUM_AND_ABOVE", cls.SAFETY_OPTIONS[0])
        # Set a default model from the static list
        default_model = "gemini-1.5-flash" # A good general-purpose default

        # Attempt to get cached models. This will trigger an API call if cache is stale/empty.
        # We pass a dummy API key here as INPUT_TYPES is static and cannot access instance variables.
        # The actual API key will be used during the 'generate' method.
        # This is a limitation of ComfyUI's static INPUT_TYPES.
        # The user's actual API key will be used during the 'generate' method.
        # For now, we'll use a placeholder or rely on the configure_api_key() in _get_cached_models
        # to attempt to load from .env/config.json.
        # If that fails, it will fall back to AVAILABLE_MODELS.
        current_models = cls._get_cached_models(configure_api_key()) # Attempt to get models using configured API key

        # Combine hardcoded and dynamically fetched models, ensuring uniqueness and order
        all_models = sorted(list(set(cls.AVAILABLE_MODELS + current_models)))

        logger.debug(f"Setting up INPUT_TYPES. Default model: '{default_model}'. Default safety: '{default_safety}'.")

        return {
            "required": {
                "model": (all_models, {"default": default_model, "tooltip": "Select the Gemini model to use."}),
                "prompt": ("STRING", {"multiline": True, "default": "Describe the image.", "tooltip": "The text prompt for generation."}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Controls randomness. Higher values (e.g., 1.0) are more creative, lower values (e.g., 0.2) are more deterministic."}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Nucleus sampling probability threshold (e.g., 0.95). 1.0 disables."}),
                "top_k": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "tooltip": "Top-K sampling threshold (consider probability of token)."}),
                "max_output_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192, "step": 1, "tooltip": "Maximum number of tokens to generate in the response."}),
                "safety_harassment": (cls.SAFETY_OPTIONS, {"default": default_safety, "tooltip": "Safety threshold for harassment content."}),
                "safety_hate_speech": (cls.SAFETY_OPTIONS, {"default": default_safety, "tooltip": "Safety threshold for hate speech content."}),
                "safety_sexually_explicit": (cls.SAFETY_OPTIONS, {"default": default_safety, "tooltip": "Safety threshold for sexually explicit content."}),
                "safety_dangerous_content": (cls.SAFETY_OPTIONS, {"default": default_safety, "tooltip": "Safety threshold for dangerous content."}),
            },
            "optional": {
                 "image_optional": ("IMAGE", {"tooltip": "Optional image input for multimodal models (e.g., gemini-pro-vision, gemini-1.5-*)."}),
                 "api_key_override": ("STRING", {"multiline": False, "default": "", "tooltip": "Optional: Override API key for this node run. Takes precedence over .env/config.json."}),
                 "max_retries": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1, "tooltip": "Maximum number of retries for transient API errors."}),
                 "retry_delay_seconds": ("INT", {"default": 5, "min": 1, "max": 60, "step": 1, "tooltip": "Delay in seconds between retries."}),
                 "include_model_thoughts": ("BOOLEAN", {"default": False, "tooltip": "If true, the model's internal thoughts may be included in the response if supported."}),
                 "thinking_token_budget": ("INT", {"default": -1, "min": -1, "max": 8192, "step": 1, "tooltip": "Token budget for model's thinking process. -1 for automatic, 0 to disable."}),
            }
        }

    # Main execution method, using imported helper functions
    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_output_tokens: int,
        safety_harassment: str,
        safety_hate_speech: str,
        safety_sexually_explicit: str,
        safety_dangerous_content: str,
        image_optional: Optional[torch.Tensor] = None,
        api_key_override: str = "",
        max_retries: int = 3,
        retry_delay_seconds: int = 5,
        include_model_thoughts: bool = False, # New parameter
        thinking_token_budget: int = -1, # New parameter
    ) -> Tuple[str]:
        """
        Executes the Gemini API call for text generation by orchestrating helper methods.
        """
        logger.info("Gemini Node: Starting execution.")
        final_output = ""

        # Determine API key to use: override > configure_api_key()
        api_key = api_key_override.strip() if api_key_override.strip() else configure_api_key()
        if not api_key:
            final_output = f"{ERROR_PREFIX} GEMINI_API_KEY not found. Please provide in 'API Key Override' or set in .env/config.json."
            logger.info("Gemini Node: Execution finished due to API key error.")
            return (final_output,)

        try:
            # 2. Prepare Configurations using utility functions
            safety_settings = prepare_safety_settings(
                safety_harassment, safety_hate_speech, safety_sexually_explicit, safety_dangerous_content
            )
            generation_config = prepare_generation_config(
                temperature, top_p, top_k, max_output_tokens
            )
            # Prepare thinking config
            thinking_config = prepare_thinking_config(
                include_model_thoughts, thinking_token_budget
            )
            # Add thinking_config to generation_config if it's not None
            if thinking_config:
                generation_config["thinkingConfig"] = thinking_config


            # Ensure prompt is UTF-8 friendly
            safe_prompt = ensure_utf8_friendly(prompt)

            # 3. Prepare initial Content Parts (text only) using utility function
            content_parts, error_msg_from_prepare = prepare_content_parts(safe_prompt)
            if error_msg_from_prepare:
                # Raise error to be caught by generic handler below
                raise RuntimeError(error_msg_from_prepare)

            # 4. Handle optional image input and add to content_parts if provided and supported
            model_name_lower = model.lower()
            supports_vision = "vision" in model_name_lower or "image" in model_name_lower or "pro" in model_name_lower or "flash" in model_name_lower # Broad check for common multimodal models

            if image_optional is not None and supports_vision:
                logger.debug("Processing image input for multimodal model.")
                try:
                    pil_image: Optional[Image.Image] = tensor_to_pil(image_optional)
                    if pil_image:
                        logger.debug("Successfully converted image tensor to PIL Image.")
                        img_byte_arr = io.BytesIO()
                        # Save as JPEG for common compatibility
                        pil_image.save(img_byte_arr, format='JPEG', quality=90)
                        img_bytes = img_byte_arr.getvalue()
                        # Create types.Part.from_bytes directly here
                        image_part = types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg')
                        logger.debug("Created image types.Part.")
                        # Add image part to the content_parts list
                        content_parts.append(image_part)
                        logger.debug("Image part added to content_parts.")
                    else:
                        error_msg = f"{ERROR_PREFIX} Image conversion failed: tensor_to_pil returned None. Check input tensor format."
                        logger.error(error_msg)
                        # Decide whether to raise error or proceed with just text
                        # For now, raise the error
                        raise RuntimeError(error_msg)
                except Exception as e:
                    error_msg = f"{ERROR_PREFIX} Error during image processing for API: {type(e).__name__}: {e}"
                    logger.error(error_msg, exc_info=True)
                    raise RuntimeError(error_msg) # Re-raise the error

            elif image_optional is not None and not supports_vision:
                 logger.warning(f"Image input provided but model '{model}' may not support vision. Proceeding with text only.")


            # 5. Call API using the updated generate_content function
            # generate_content now accepts the contents list directly
            generated_text, response_error_msg = generate_content(
                api_key=api_key,
                model_name=model,
                contents=content_parts,
                generation_config=generation_config,
                safety_settings=safety_settings,
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds
            )
            # Prioritize showing the API block error message if it exists
            final_output = response_error_msg if response_error_msg else generated_text

        # Handle potential Google API errors
        except google_exceptions.GoogleAPIError as e:
            error_msg = f"{ERROR_PREFIX} Google API Error - Status: {getattr(e, 'code', 'N/A')}, Message: {e}"
            logger.error(error_msg, exc_info=True)
            final_output = f"{ERROR_PREFIX} A Google API error occurred ({getattr(e, 'code', 'N/A')}). Check console logs."

        # Catch broader errors during the overall process
        except Exception as e:
            error_details = str(e)
            if hasattr(e, 'message') and e.message: error_details = e.message
            elif hasattr(e, 'details') and callable(e.details) and e.details(): error_details = e.details()
            error_msg = f"{ERROR_PREFIX} Gemini Node Error ({type(e).__name__}): {error_details}"
            logger.error(error_msg, exc_info=True)
            # Use the already formatted error if it came from a raised exception, otherwise use the new one
            final_output = error_msg if not final_output.startswith(ERROR_PREFIX) else final_output

        # Ensure final_output is always a string before returning
        if not isinstance(final_output, str):
            logger.error(f"Final output was not a string ({type(final_output)}), converting. Value: {final_output}")
            final_output = str(final_output)

        logger.info("Gemini Node: Execution finished.")
        # Ensure the final output is UTF-8 friendly before returning
        return (ensure_utf8_friendly(final_output),)

# Note: Mappings are handled in google_ai/__init__.py
