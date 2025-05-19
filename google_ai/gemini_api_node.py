"""
ComfyUI node for interacting with the Google Gemini API.
Includes advanced generation options.
Configuration is handled via config.json or node input override.
"""
import logging
import torch
from typing import Optional, Dict, Any, Tuple

# Import necessary functions and constants from the utils module
from .gemini_utils import (
    get_available_models,
    configure_api_key,
    prepare_image_part,
    generate_content,
    prepare_safety_settings, # Re-add import
    prepare_generation_config, # Re-add import
    SAFETY_SETTINGS_MAP, # Re-add import
    SAFETY_THRESHOLD_TO_NAME, # Re-add import
    ERROR_PREFIX,
)

# Import shared utility for text encoding (if still needed, check usage)
# from ..shared_utils.text_encoding_utils import ensure_utf8_friendly # Not used in this simplified version

# Setup logger for this module
logger = logging.getLogger(__name__)
# Ensure handler is configured if root logger isn't set up
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- ComfyUI Node Definition ---

class GeminiNode:
    """
    A ComfyUI node to interact with the Google Gemini API for text generation,
    optionally using an image input for multimodal models, with advanced options.

    Configuration is handled via config.json or the node's api_key_override input.
    It dynamically fetches the list of available models supporting content generation
    when ComfyUI loads the node, using the configured API key.
    """
    # Fetch models using the utility function during class initialization
    # This requires an API key to be available at this stage, which might be
    # problematic if it only exists in config.json or node input.
    # A more robust approach might be to fetch models dynamically in the execute
    # function or provide a static list and allow user to override.
    # For now, we'll attempt to load the key and fetch models here,
    # but be aware this might fail if the key isn't in env vars at load time.
    _api_key_for_init = configure_api_key() # Attempt to load API key for model fetching
    AVAILABLE_MODELS = get_available_models(_api_key_for_init) if _api_key_for_init else [] # Fetch models if key is available
    if not AVAILABLE_MODELS:
        logger.warning("Could not fetch dynamic model list during node init. Using default list from gemini_utils.")
        # Fallback to the default list defined in gemini_utils if dynamic fetch fails
        from .gemini_utils import DEFAULT_MODELS as FALLBACK_MODELS
        AVAILABLE_MODELS = FALLBACK_MODELS

    SAFETY_OPTIONS = list(SAFETY_SETTINGS_MAP.keys()) # Re-add safety options

    # Define ComfyUI node attributes
    RETURN_TYPES: Tuple[str] = ("STRING",)
    RETURN_NAMES: Tuple[str] = ("text",)
    FUNCTION: str = "generate"
    CATEGORY: str = "👽 Divergent Nodes/Gemini"

    def __init__(self):
        """Initializes the Gemini node instance."""
        logger.debug("GeminiNode instance created.")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """
        Defines the input types and options for the ComfyUI node interface.
        Re-added advanced inputs.
        """
        # Determine default safety setting names using constants from utils
        default_safety = SAFETY_THRESHOLD_TO_NAME.get("BLOCK_MEDIUM_AND_ABOVE", cls.SAFETY_OPTIONS[0])
        # Select a reasonable default model
        default_model = ""
        if cls.AVAILABLE_MODELS:
             # Prefer models with 'flash' or 'pro' and 'latest' if available
             preferred_models = [m for m in cls.AVAILABLE_MODELS if "flash" in m or "pro" in m]
             latest_preferred = next((m for m in preferred_models if "latest" in m), None)
             any_preferred = next((m for m in preferred_models), None)
             default_model = latest_preferred or any_preferred or cls.AVAILABLE_MODELS[0]
        else:
             # If dynamic list is empty, use a common default from the fallback list
             from .gemini_utils import DEFAULT_MODELS as FALLBACK_MODELS
             default_model = next((m for m in FALLBACK_MODELS if "flash" in m or "pro" in m), FALLBACK_MODELS[0] if FALLBACK_MODELS else "")

        logger.debug(f"Setting up INPUT_TYPES. Default model: '{default_model}'. Default safety: '{default_safety}'.")

        return {
            "required": {
                "model": (cls.AVAILABLE_MODELS, {"default": default_model, "tooltip": "Select the Gemini model to use."}),
                "prompt": ("STRING", {"multiline": True, "default": "Describe the image.", "tooltip": "The text prompt for generation."}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Controls randomness. Higher values (e.g., 1.0) are more creative, lower values (e.2) are more deterministic."}), # Re-add input
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Nucleus sampling probability threshold (e.g., 0.95). 1.0 disables."}), # Re-add input
                "top_k": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "tooltip": "Top-K sampling threshold (consider probability of token)."}), # Re-add input
                "max_output_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192, "step": 1, "tooltip": "Maximum number of tokens to generate in the response."}), # Re-add input
                "safety_harassment": (cls.SAFETY_OPTIONS, {"default": default_safety, "tooltip": "Safety threshold for harassment content."}), # Re-add input
                "safety_hate_speech": (cls.SAFETY_OPTIONS, {"default": default_safety, "tooltip": "Safety threshold for hate speech content."}), # Re-add input
                "safety_sexually_explicit": (cls.SAFETY_OPTIONS, {"default": default_safety, "tooltip": "Safety threshold for sexually explicit content."}), # Re-add input
                "safety_dangerous_content": (cls.SAFETY_OPTIONS, {"default": default_safety, "tooltip": "Safety threshold for dangerous content."}), # Re-add input
            },
            "optional": {
                 "image_optional": ("IMAGE", {"tooltip": "Optional image input for multimodal models (e.g., gemini-pro-vision, gemini-1.5-*)."}),
                 "api_key_override": ("STRING", {"multiline": False, "default": "", "tooltip": "Optional API key override. If provided, this key will be used instead of the one in config.json or environment variables."}),
            }
        }

    # Main execution method, using imported helper functions
    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float, # Add param
        top_p: float, # Add param
        top_k: int, # Add param
        max_output_tokens: int, # Add param
        safety_harassment: str, # Add param
        safety_hate_speech: str, # Add param
        safety_sexually_explicit: str, # Add param
        safety_dangerous_content: str, # Add param
        image_optional: Optional[torch.Tensor] = None,
        api_key_override: str = ""
    ) -> Tuple[str]:
        """
        Executes the Gemini API call for text generation by orchestrating helper methods.
        Includes advanced options.
        """
        logger.info("Gemini Node: Starting execution.")
        final_output = ""

        # 1. Load API Key using utility function, passing the override
        api_key = configure_api_key(api_key_override=api_key_override)
        if not api_key:
            final_output = f"{ERROR_PREFIX} GOOGLE_API_KEY not found. Check config.json, node input, or environment variables."
            logger.error(final_output)
            logger.info("Gemini Node: Execution finished due to API key error.")
            return (final_output,)

        try:
            # 2. Prepare Configurations using utility functions (Re-added)
            safety_settings = prepare_safety_settings(
                safety_harassment, safety_hate_speech, safety_sexually_explicit, safety_dangerous_content
            )
            generation_config = prepare_generation_config(
                temperature, top_p, top_k, max_output_tokens
            )

            # Ensure prompt is UTF-8 friendly (Re-add if needed, check gemini_utils)
            # safe_prompt = ensure_utf8_friendly(prompt) # Assuming gemini_utils handles this now
            safe_prompt = prompt # Use raw prompt for now, let gemini_utils handle encoding if needed

            # 3. Prepare Image Part if provided
            image_part = None
            if image_optional is not None:
                image_part, img_error = prepare_image_part(image_optional)
                if img_error:
                    logger.error(f"Image preparation failed: {img_error}")
                    return (img_error,)
                if image_part is None:
                     error_msg = f"{ERROR_PREFIX} Image preparation failed unexpectedly."
                     logger.error(error_msg)
                     return (error_msg,)

            # 4. Call API using the generate_content utility function
            # Pass generation_config and safety_settings here
            final_output, response_error_msg = generate_content(
                api_key=api_key,
                model_name=model,
                prompt=safe_prompt, # Use safe_prompt if ensure_utf8_friendly is re-added
                image_part=image_part,
                generation_config=generation_config, # Pass config
                safety_settings=safety_settings, # Pass safety
            )

            # The generate_content function already processes the response
            # and returns the final output or an error message.

        except Exception as e:
            error_details = str(e)
            if hasattr(e, 'message') and e.message: error_details = e.message
            elif hasattr(e, 'details') and callable(e.details) and e.details(): error_details = e.details()
            error_msg = f"{ERROR_PREFIX} Gemini Node Error ({type(e).__name__}): {error_details}"
            logger.error(error_msg, exc_info=True)
            final_output = error_msg if not final_output.startswith(ERROR_PREFIX) else final_output

        # Ensure final_output is always a string before returning
        if not isinstance(final_output, str):
            logger.error(f"Final output was not a string ({type(final_output)}), converting. Value: {final_output}")
            final_output = str(final_output)

        logger.info("Gemini Node: Execution finished.")
        # Ensure the final output is UTF-8 friendly before returning (optional, depends on needs)
        # return (ensure_utf8_friendly(final_output),) # Re-add if needed
        return (final_output,) # Return directly for simplicity

# Note: Mappings are handled in google_ai/__init__.py
