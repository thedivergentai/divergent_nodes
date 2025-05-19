"""
ComfyUI node for interacting with the Google Gemini API.
Simplified implementation based on Google AI documentation examples.
Supports text generation and multimodal input (text + image).
Configuration is handled via config.json or node input override.
"""
import logging
import torch
from typing import Optional, Dict, Any, Tuple

# Import necessary functions and constants from the simplified utils module
from .gemini_utils import (
    get_available_models,
    configure_api_key,
    prepare_image_part, # Use the simplified image preparation
    generate_content, # Use the simplified content generation
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
    optionally using an image input for multimodal models.

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
        Simplified inputs based on the rewrite plan.
        """
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

        logger.debug(f"Setting up INPUT_TYPES. Default model: '{default_model}'.")

        return {
            "required": {
                "model": (cls.AVAILABLE_MODELS, {"default": default_model, "tooltip": "Select the Gemini model to use."}),
                "prompt": ("STRING", {"multiline": True, "default": "Describe the image.", "tooltip": "The text prompt for generation."}),
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
        image_optional: Optional[torch.Tensor] = None,
        api_key_override: str = "" # Add api_key_override as an input parameter
    ) -> Tuple[str]:
        """
        Executes the Gemini API call for text generation by orchestrating helper methods.
        Simplified execution based on the rewrite plan.
        """
        logger.info("Gemini Node: Starting execution.")
        final_output = ""

        # 1. Load API Key using utility function, passing the override
        api_key = configure_api_key(api_key_override=api_key_override) # Pass the override here
        if not api_key:
            final_output = f"{ERROR_PREFIX} GOOGLE_API_KEY not found. Check config.json, node input, or environment variables."
            logger.error(final_output) # Use error level for missing key
            logger.info("Gemini Node: Execution finished due to API key error.")
            return (final_output,)

        try:
            # 2. Prepare Image Part if provided
            image_part = None
            if image_optional is not None:
                image_part, img_error = prepare_image_part(image_optional)
                if img_error:
                    # If image preparation fails, return the error
                    logger.error(f"Image preparation failed: {img_error}")
                    return (img_error,)
                if image_part is None:
                     # Should not happen if img_error is None, but as a safeguard
                     error_msg = f"{ERROR_PREFIX} Image preparation failed unexpectedly."
                     logger.error(error_msg)
                     return (error_msg,)


            # 3. Call API using the simplified generate_content utility function
            # This function handles client initialization and content generation
            final_output, response_error_msg = generate_content(
                api_key=api_key,
                model_name=model,
                prompt=prompt,
                image_part=image_part,
            )

            # The generate_content function already processes the response
            # and returns the final output or an error message.

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
        # Ensure the final output is UTF-8 friendly before returning (optional, depends on needs)
        # return (ensure_utf8_friendly(final_output),) # Re-add if needed
        return (final_output,) # Return directly for simplicity

# Note: Mappings are handled in google_ai/__init__.py
