"""
ComfyUI node for interacting with the Google Gemini API.
Supports text generation and multimodal input (text + image).
Requires a GEMINI_API_KEY environment variable.
"""
import logging
import torch
from typing import Optional, Dict, Any, Tuple

"""
ComfyUI node for interacting with the Google Gemini API.
Supports text generation and multimodal input (text + image).
Configuration is handled via config.json or node input override.
"""
import logging
import torch
from typing import Optional, Dict, Any, Tuple

# Import necessary functions and constants from the new utils module
from .gemini_utils import (
    get_available_models,
    configure_api_key,
    prepare_safety_settings,
    prepare_generation_config,
    generate_content_via_client, # Use the client-based generation function
    prepare_content_parts,
    process_api_response, # Keep if needed for processing responses outside generate_content_via_client
    SAFETY_SETTINGS_MAP,
    SAFETY_THRESHOLD_TO_NAME,
    ERROR_PREFIX,
    google_exceptions # Import for exception handling
)

# Import shared utility for text encoding
from ..shared_utils.text_encoding_utils import ensure_utf8_friendly

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

    SAFETY_OPTIONS = list(SAFETY_SETTINGS_MAP.keys())

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
                 "api_key_override": ("STRING", {"multiline": False, "default": "", "tooltip": "Optional API key override. If provided, this key will be used instead of the one in config.json or environment variables."}),
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
        api_key_override: str = "" # Add api_key_override as an input parameter
    ) -> Tuple[str]:
        """
        Executes the Gemini API call for text generation by orchestrating helper methods.
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
            # 2. Prepare Configurations using utility functions
            safety_settings = prepare_safety_settings(
                safety_harassment, safety_hate_speech, safety_sexually_explicit, safety_dangerous_content
            )
            generation_config = prepare_generation_config(
                temperature, top_p, top_k, max_output_tokens
            )

            # Ensure prompt is UTF-8 friendly
            safe_prompt = ensure_utf8_friendly(prompt)

            # 3. Prepare Content Parts using utility function
            content_parts, img_error = prepare_content_parts(safe_prompt, image_optional, model)
            if img_error:
                # Raise error to be caught by generic handler below
                raise RuntimeError(img_error)

            # 4. Call API using the client-based utility function
            # This function now handles client initialization and content generation
            final_output, response_error_msg = generate_content_via_client(
                api_key=api_key,
                model_name=model,
                contents=content_parts,
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            # The generate_content_via_client function already processes the response
            # and returns the final output or an error message.
            # No need to call process_api_response here again.

        # Handle potential Google API errors if sdk types were imported
        except google_exceptions.GoogleAPIError as e:
             # Check if google_exceptions was successfully imported before using it
             if google_exceptions:
                 error_msg = f"{ERROR_PREFIX} Google API Error - Status: {getattr(e, 'code', 'N/A')}, Message: {e}"
                 logger.error(error_msg, exc_info=True)
                 final_output = f"{ERROR_PREFIX} A Google API error occurred ({getattr(e, 'code', 'N/A')}). Check console logs."
             else:
                 # Fallback if specific exception types aren't available
                 error_msg = f"{ERROR_PREFIX} An API error occurred: {e}"
                 logger.error(error_msg, exc_info=True)
                 final_output = f"{ERROR_PREFIX} An API error occurred. Check console logs."

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
