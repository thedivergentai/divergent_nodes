"""
ComfyUI node for interacting with the Google Gemini API.
Supports text generation and multimodal input (text + image).
Requires a GEMINI_API_KEY environment variable.
"""
import os
import torch
import numpy as np
from PIL import Image
import google.generativeai as genai
# Attempt to import generation_types safely for potential future use or type checking
try:
    from google.generativeai.types import generation_types, SafetySettingDict, GenerationConfigDict
    from google.generativeai.generative_models import GenerativeModel, GenerateContentResponse
    from google.api_core import exceptions as google_exceptions # For specific error handling
except ImportError:
    # Define placeholders if imports fail, allowing basic functionality
    # but potentially losing some type safety or specific error handling.
    import logging # Import logging here for the warning
    logging.warning("Could not import specific google.generativeai types. Type safety might be reduced.")
    generation_types = None
    SafetySettingDict = Dict[str, Any]
    GenerationConfigDict = Dict[str, Any]
    GenerativeModel = Any
    GenerateContentResponse = Any
    google_exceptions = None # Cannot catch specific exceptions if module not found

from dotenv import load_dotenv, find_dotenv
from typing import Optional, Dict, Any, Tuple, List, Union, TypeAlias
import logging

# Use relative import for shared utils
try:
    from ..shared_utils import tensor_to_pil
except ImportError:
    # Fallback for direct execution or different structure
    logging.warning("Could not perform relative import for shared_utils, attempting direct import.")
    # Ensure logger is configured before first use if fallback occurs
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    from shared_utils import tensor_to_pil

# Setup logger for this module
logger = logging.getLogger(__name__)
# Ensure handler is configured if root logger isn't set up
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Type Aliases ---
PilImageT: TypeAlias = Image.Image

# --- Constants ---
# Default list of models known to support generateContent (as of early 2024)
# Sorted newest -> oldest, with 'latest' tags first. Dynamic fetch overrides this.
# Keep this updated based on Google's model releases.
DEFAULT_MODELS: List[str] = [
    "models/gemini-1.5-pro-latest",
    "models/gemini-1.5-flash-latest",
    # Add other relevant models here based on current availability
    "models/gemini-1.0-pro",
    "models/gemini-pro-vision", # Example vision model
]

# Mapping from user-friendly safety setting names to API constants
# Using strings directly as API might change constants; ensures basic functionality.
SAFETY_SETTINGS_MAP: Dict[str, str] = {
    "Default (Unspecified)": "HARM_BLOCK_THRESHOLD_UNSPECIFIED",
    "Block Low & Above": "BLOCK_LOW_AND_ABOVE",
    "Block Medium & Above": "BLOCK_MEDIUM_AND_ABOVE",
    "Block High Only": "BLOCK_ONLY_HIGH",
    "Block None": "BLOCK_NONE",
}
# Reverse map for finding default friendly name from API constant
SAFETY_THRESHOLD_TO_NAME: Dict[str, str] = {v: k for k, v in SAFETY_SETTINGS_MAP.items()}

# --- Helper Functions ---

def get_available_models() -> List[str]:
    """
    Fetches available Gemini models supporting 'generateContent' via the API.

    Requires the GEMINI_API_KEY environment variable to be set (e.g., in a .env file).
    Falls back to a default list (`DEFAULT_MODELS`) if the API key is missing or
    if the API call fails.

    Returns:
        List[str]: A list of available model names (e.g., "models/gemini-1.5-pro-latest"),
                   sorted with 'latest' tags first, then alphabetically.
    """
    logger.debug("Attempting to fetch available Gemini models...")
    api_key_found = False
    try:
        # Load API key specifically for listing models
        # Use find_dotenv to locate .env in parent directories if necessary
        load_dotenv(find_dotenv(), override=True) # Override ensures fresh read if called again
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not found in environment or .env file. Cannot fetch dynamic model list.")
            return DEFAULT_MODELS # Return default list immediately

        api_key_found = True
        genai.configure(api_key=api_key)
        model_list: List[str] = []
        logger.debug("Fetching available models from Gemini API...")
        # Iterate through models and check supported methods
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                 model_list.append(m.name) # Use the full model name

        if not model_list:
             logger.warning("No models supporting 'generateContent' found via API. Using default list.")
             return DEFAULT_MODELS

        # Sort for consistency: 'latest' first, then alphabetically
        model_list.sort(key=lambda x: ('latest' not in x, x))
        logger.info(f"Fetched {len(model_list)} models supporting 'generateContent' from Gemini API.")
        return model_list

    except ImportError as e:
         logger.error(f"Google Generative AI SDK not installed or import failed: {e}. Cannot fetch models. Using default list.", exc_info=True)
         return DEFAULT_MODELS
    except Exception as e:
        # Log detailed error but return default list for resilience
        log_func = logger.error if api_key_found else logger.warning
        log_func(f"Failed to fetch models from Gemini API: {e}. Using default list.", exc_info=True)
        return DEFAULT_MODELS

# --- ComfyUI Node Definition ---

class GeminiNode:
    """
    A ComfyUI node to interact with the Google Gemini API for text generation,
    optionally using an image input for multimodal models.

    Requires a `GEMINI_API_KEY` environment variable (e.g., in a `.env` file
    in the ComfyUI root or a parent directory). It dynamically fetches the list
    of available models supporting content generation when ComfyUI loads the node.
    """
    # Fetch models when the class is loaded by ComfyUI
    # This ensures the dropdown is populated on startup.
    AVAILABLE_MODELS: List[str] = get_available_models()
    SAFETY_OPTIONS: List[str] = list(SAFETY_SETTINGS_MAP.keys())

    # Define ComfyUI node attributes
    RETURN_TYPES: Tuple[str] = ("STRING",)
    RETURN_NAMES: Tuple[str] = ("text",)
    FUNCTION: str = "generate"
    CATEGORY: str = "Divergent Nodes 👽/Gemini" # Keep consistent category

    def __init__(self):
        """Initializes the Gemini node instance."""
        logger.debug("GeminiNode instance created.")
        # API key loading/checking is handled within the execute method per call

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """
        Defines the input types and options for the ComfyUI node interface.

        Dynamically populates the model list and sets defaults. Includes tooltips
        for better user understanding of each parameter.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary defining "required" and "optional" inputs.
        """
        # Determine default safety setting names (user-friendly)
        # Use BLOCK_MEDIUM_AND_ABOVE as a sensible default if available
        default_safety = SAFETY_THRESHOLD_TO_NAME.get("BLOCK_MEDIUM_AND_ABOVE", cls.SAFETY_OPTIONS[0])
        # Select a reasonable default model, prioritizing 'latest' flash or pro
        default_model = ""
        if cls.AVAILABLE_MODELS:
             # Prefer latest flash, then latest pro, then first available
             flash_latest = next((m for m in cls.AVAILABLE_MODELS if "flash-latest" in m), None)
             pro_latest = next((m for m in cls.AVAILABLE_MODELS if "pro-latest" in m), None)
             default_model = flash_latest or pro_latest or cls.AVAILABLE_MODELS[0]

        logger.debug(f"Setting up INPUT_TYPES. Default model: '{default_model}'. Default safety: '{default_safety}'.")

        return {
            "required": {
                "model": (cls.AVAILABLE_MODELS, {"default": default_model, "tooltip": "Select the Gemini model to use."}),
                "prompt": ("STRING", {"multiline": True, "default": "Describe the image.", "tooltip": "The text prompt for generation."}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Controls randomness. Higher values (e.g., 1.0) are more creative, lower values (e.g., 0.2) are more deterministic."}), # Max 2.0 as per Gemini API docs Mar 2024
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Nucleus sampling probability threshold (e.g., 0.95). 1.0 disables."}), # Default 1.0 as per Gemini API docs Mar 2024
                "top_k": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "tooltip": "Top-K sampling threshold (consider probability of token)."}), # Default 1 as per Gemini API docs Mar 2024
                "max_output_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192, "step": 1, "tooltip": "Maximum number of tokens to generate in the response."}), # Max 8192 for 1.5 models as per docs Mar 2024
                # Safety settings using user-friendly names
                "safety_harassment": (cls.SAFETY_OPTIONS, {"default": default_safety, "tooltip": "Safety threshold for harassment content."}),
                "safety_hate_speech": (cls.SAFETY_OPTIONS, {"default": default_safety, "tooltip": "Safety threshold for hate speech content."}),
                "safety_sexually_explicit": (cls.SAFETY_OPTIONS, {"default": default_safety, "tooltip": "Safety threshold for sexually explicit content."}),
                "safety_dangerous_content": (cls.SAFETY_OPTIONS, {"default": default_safety, "tooltip": "Safety threshold for dangerous content."}),
            },
            "optional": {
                 "image_optional": ("IMAGE", {"tooltip": "Optional image input for multimodal models (e.g., gemini-pro-vision, gemini-1.5-*)."}), # ComfyUI IMAGE type
            }
        }

    # --------------------------------------------------------------------------
    # Private Helper Methods for Generation Logic
    # --------------------------------------------------------------------------

    def _configure_api_key(self) -> Optional[str]:
        """Checks for and configures the Gemini API key."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            load_dotenv(find_dotenv(), override=True)
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.error("GEMINI_API_KEY not found in environment or .env file.")
                return None
        try:
            genai.configure(api_key=api_key)
            logger.info("Gemini API configured successfully.")
            return api_key
        except Exception as e:
            logger.error(f"Failed to configure Gemini API with key: {e}", exc_info=True)
            return None

    def _prepare_safety_settings(self, safety_harassment: str, safety_hate_speech: str,
                                 safety_sexually_explicit: str, safety_dangerous_content: str) -> List[SafetySettingDict]:
        """Builds the safety settings list from node inputs."""
        logger.debug("Preparing safety settings.")
        return [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": SAFETY_SETTINGS_MAP.get(safety_harassment, "HARM_BLOCK_THRESHOLD_UNSPECIFIED")},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": SAFETY_SETTINGS_MAP.get(safety_hate_speech, "HARM_BLOCK_THRESHOLD_UNSPECIFIED")},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": SAFETY_SETTINGS_MAP.get(safety_sexually_explicit, "HARM_BLOCK_THRESHOLD_UNSPECIFIED")},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": SAFETY_SETTINGS_MAP.get(safety_dangerous_content, "HARM_BLOCK_THRESHOLD_UNSPECIFIED")},
        ]

    def _prepare_generation_config(self, temperature: float, top_p: float, top_k: int,
                                   max_output_tokens: int) -> Union[GenerationConfigDict, Dict[str, Any]]:
        """Builds the generation configuration object or dictionary."""
        logger.debug("Preparing generation config.")
        if generation_types:
            return generation_types.GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_output_tokens,
            )
        else:
            logger.warning("Using dictionary for generation_config due to failed type import.")
            return {
                "temperature": temperature, "top_p": top_p,
                "top_k": top_k, "max_output_tokens": max_output_tokens,
            }

    def _initialize_model(self, model_name: str, safety_settings: List[SafetySettingDict],
                          generation_config: Union[GenerationConfigDict, Dict[str, Any]]) -> GenerativeModel:
        """Initializes the GenerativeModel instance."""
        logger.info(f"Initializing Gemini model: {model_name}")
        try:
            return genai.GenerativeModel(
                model_name=model_name,
                safety_settings=safety_settings,
                generation_config=generation_config
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model '{model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize Gemini model: {e}") from e

    def _prepare_content_parts(self, prompt: str, image_optional: Optional[torch.Tensor], model_name: str) -> Tuple[List[Any], Optional[str]]:
        """Prepares the list of content parts (text and optional image)."""
        content_parts: List[Any] = [prompt]
        pil_image: Optional[PilImageT] = None
        error_msg: Optional[str] = None

        if image_optional is not None:
            logger.debug("Processing optional image input.")
            try:
                pil_image = tensor_to_pil(image_optional)
                if pil_image:
                    logger.debug("Successfully converted image tensor to PIL Image.")
                    content_parts.append(pil_image)
                else:
                    logger.warning("tensor_to_pil returned None. Check input tensor format.")
                    # Decide whether to error or proceed without image
                    # error_msg = "ERROR: Failed to convert input tensor to PIL image."
            except Exception as img_e:
                logger.error(f"Failed to convert input tensor to PIL Image: {img_e}", exc_info=True)
                error_msg = f"ERROR: Failed to process input image: {img_e}"
        elif "vision" in model_name: # Simple check if vision model selected without image
            logger.warning(f"Vision model '{model_name}' selected, but no image provided.")

        return content_parts, error_msg

    def _process_api_response(self, response: GenerateContentResponse) -> Tuple[str, Optional[str]]:
        """Processes the Gemini API response, extracting text or handling errors/blocks."""
        generated_text: str = ""
        error_msg: Optional[str] = None
        error_prefix = "ERROR:"

        try:
            if not getattr(response, 'candidates', None):
                block_reason_code = "UNSPECIFIED"
                prompt_feedback = getattr(response, 'prompt_feedback', None)
                if prompt_feedback and hasattr(prompt_feedback, 'block_reason'):
                    block_reason_code = prompt_feedback.block_reason
                block_reason_str = getattr(block_reason_code, 'name', str(block_reason_code))
                error_msg = f"{error_prefix} Blocked/Failed: Generation failed. Block Reason: {block_reason_str}."
                logger.error(error_msg + f" Raw Response: {response}")
                return "", error_msg

            candidate = response.candidates[0]
            finish_reason = getattr(candidate, 'finish_reason', None)
            finish_reason_name = getattr(finish_reason, 'name', str(finish_reason))

            if finish_reason_name == 'SAFETY':
                ratings = getattr(candidate, 'safety_ratings', [])
                ratings_str = ', '.join([f"{getattr(r.category, 'name', 'UNK')}: {getattr(r.probability, 'name', 'UNK')}" for r in ratings])
                error_msg = f"{error_prefix} Blocked: Response stopped by safety settings. Ratings: [{ratings_str}]"
                logger.error(error_msg)
                return "", error_msg
            elif finish_reason_name == 'RECITATION':
                error_msg = f"{error_prefix} Blocked: Response stopped for potential recitation."
                logger.error(error_msg)
                return "", error_msg
            elif finish_reason_name == 'MAX_TOKENS':
                logger.warning("Generation stopped: Reached max_output_tokens limit.")

            content = getattr(candidate, 'content', None)
            if content and getattr(content, 'parts', None):
                generated_text = "".join(part.text for part in content.parts if hasattr(part, 'text'))
                logger.info(f"Successfully generated text (length: {len(generated_text)}). Finish Reason: {finish_reason_name}")
            else:
                status_msg = f"Response received but no text content. Finish Reason: {finish_reason_name}"
                logger.warning(status_msg)
                # Return status message as text if no actual text generated but not explicitly blocked
                generated_text = status_msg if not error_msg else "" # Avoid overwriting block errors

        except (ValueError, IndexError, AttributeError) as e:
             logger.error(f"Error accessing response structure: {type(e).__name__}. Raw response: {response}", exc_info=True)
             error_msg = f"{error_prefix} Error parsing API response: {type(e).__name__}."

        return generated_text, error_msg


    # Main execution method, refactored to use helpers
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
    ) -> Tuple[str]:
        """
        Executes the Gemini API call for text generation by orchestrating helper methods.

        Handles API key configuration, image conversion, request building,
        API interaction, and response/error handling.

        Args:
            (See INPUT_TYPES for parameter details)

        Returns:
            Tuple[str]: A tuple containing a single string: the generated text or an error message.
        """
        logger.info("Gemini Node: Starting execution.")
        error_prefix = "ERROR:"

        # 1. Configure API Key
        if not self._configure_api_key():
            return (f"{error_prefix} GEMINI_API_KEY configuration failed. Check environment/.env.",)

        try:
            # 2. Prepare Configurations
            safety_settings = self._prepare_safety_settings(
                safety_harassment, safety_hate_speech, safety_sexually_explicit, safety_dangerous_content
            )
            generation_config = self._prepare_generation_config(
                temperature, top_p, top_k, max_output_tokens
            )

            # 3. Initialize Model
            gemini_model = self._initialize_model(model, safety_settings, generation_config)

            # 4. Prepare Content Parts
            content_parts, img_error = self._prepare_content_parts(prompt, image_optional, model)
            if img_error:
                return (img_error,) # Return image processing error

            # 5. Call API
            logger.info(f"Sending request to Gemini API model '{model}'...")
            response = gemini_model.generate_content(content_parts)

            # 6. Process Response
            generated_text, response_error = self._process_api_response(response)
            if response_error:
                return (response_error,) # Return error from response processing
            else:
                return (generated_text,)

        # Handle potential Google API errors if sdk types were imported
        except google_exceptions.GoogleAPIError as e:
             error_msg = f"{error_prefix} Google API Error - {e}"
             logger.error(error_msg, exc_info=True)
             return (f"{error_prefix} A Google API error occurred. Check console logs.",)
        # Catch broader errors during the overall process
        except Exception as e:
            error_details = str(e)
            if hasattr(e, 'message'): error_details = e.message
            elif hasattr(e, 'details'): error_details = e.details()
            error_msg = f"{error_prefix} Gemini Node Error - {error_details}"
            logger.error(error_msg, exc_info=True)
            return (f"{error_prefix} An unexpected error occurred. Check console logs.",)
        finally:
             logger.info("Gemini Node: Execution finished.")


# Note: Mappings are handled in google_ai/__init__.py
