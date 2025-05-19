"""
Utility functions and constants for the Gemini API node.
"""
import os
import logging
from typing import Optional, Dict, Any, Tuple, List, Union, TypeAlias
from PIL import Image
import torch
from google import genai
from google.genai import types

# Import the new config manager
from ..shared_utils.config_manager import load_config

# Attempt to import specific types for better safety
try:
    from google.generativeai.types import (
        generation_types,
        SafetySettingDict,
        GenerationConfigDict,
        GenerateContentResponse
    )
    from google.generativeai.generative_models import GenerativeModel
    from google.api_core import exceptions as google_exceptions
except ImportError:
    logging.warning("gemini_utils: Could not import specific google.generativeai types. Type safety might be reduced.")
    generation_types = None
    SafetySettingDict = dict # type: ignore
    GenerationConfigDict = dict # type: ignore
    GenerativeModel = Any # type: ignore
    GenerateContentResponse = Any # type: ignore
    google_exceptions = None

# Relative import for shared utils within the same package structure
try:
    from ..shared_utils.image_conversion import tensor_to_pil
except ImportError:
    logging.warning("gemini_utils: Could not perform relative import for shared_utils, attempting direct import.")
    try:
        from shared_utils.image_conversion import tensor_to_pil # type: ignore
    except ImportError:
        logging.critical("gemini_utils: Failed both relative and direct import of shared_utils.image_conversion.")
        def tensor_to_pil(*args: Any, **kwargs: Any) -> None:
            logger.error("tensor_to_pil function is unavailable due to import errors.")
            return None

# Setup logger for this module
logger = logging.getLogger(__name__)
# Ensure handler is configured if root logger isn't set up
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Type Aliases ---
PilImageT: TypeAlias = Image.Image

# --- Constants ---
DEFAULT_MODELS: List[str] = [
    "models/gemini-2.5-flash-preview-04-17",
    "models/gemini-2.5-pro-preview-05-06",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-preview-image-generation",
    "models/gemini-2.0-flash-lite",
    "models/gemini-1.5-flash",
    "models/gemini-1.5-flash-8b",
    "models/gemini-1.5-pro",
    "models/gemini-embedding-exp",
    "models/imagen-3.0-generate-002",
    "models/veo-2.0-generate-001",
    "models/gemini-2.0-flash-live-001",
]

SAFETY_SETTINGS_MAP: Dict[str, str] = {
    "Default (Unspecified)": "HARM_BLOCK_THRESHOLD_UNSPECIFIED",
    "Block Low & Above": "BLOCK_LOW_AND_ABOVE",
    "Block Medium & Above": "BLOCK_MEDIUM_AND_ABOVE",
    "Block High Only": "BLOCK_ONLY_HIGH",
    "Block None": "BLOCK_NONE",
}
SAFETY_THRESHOLD_TO_NAME: Dict[str, str] = {v: k for k, v in SAFETY_SETTINGS_MAP.items()}
ERROR_PREFIX = "ERROR:"

# --- Helper Functions ---

# Create a client instance (as shown in some examples)
# Note: The API key is configured globally or per-client.
# We will continue to use the .env loading and global configure for simplicity with the existing node structure.
# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY")) # Not using client instance for now

def get_available_models(api_key: Optional[str]) -> List[str]:
    """Fetches available Gemini models supporting 'generateContent' using genai.Client."""
    logger.debug("Attempting to fetch available Gemini models using genai.Client...")
    if not api_key:
         logger.warning("API key not provided. Cannot fetch dynamic model list.")
         return DEFAULT_MODELS

    try:
        # Instantiate client with the provided API key
        client = genai.Client(api_key=api_key)
        logger.debug("genai.Client instantiated for model listing.")

        # Use client.list_models()
        model_list: List[str] = [
            m.name for m in client.list_models()
            if 'generateContent' in m.supported_generation_methods
        ]
        if not model_list:
             logger.warning("No models supporting 'generateContent' found via API. Using default list.")
             return DEFAULT_MODELS
        model_list.sort(key=lambda x: ('latest' not in x, x))
        logger.info(f"Fetched {len(model_list)} models supporting 'generateContent' from Gemini API.")
        return model_list
    except ImportError as e:
         logger.error(f"Google SDK import failed: {e}. Using default list.", exc_info=True)
         return DEFAULT_MODELS
    except Exception as e:
        logger.error(f"Failed to fetch models from Gemini API: {e}. Using default list.", exc_info=True)
        return DEFAULT_MODELS

def configure_api_key(api_key_override: Optional[str] = None) -> Optional[str]:
    """
    Determines the Gemini API key to use (override > config.json > environment).
    Returns the key string.
    """
    logger.debug("configure_api_key called.")

    # 1. Check override first
    if api_key_override and api_key_override.strip():
        logger.debug("Using API key from override.")
        return api_key_override.strip()

    # 2. Load config from config.json
    config = load_config()
    api_key = config.get("GOOGLE_API_KEY")
    if api_key:
        logger.debug("GOOGLE_API_KEY found in config.json.")
        return api_key

    # 3. Fallback to environment variable (less preferred)
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        logger.warning("Using GOOGLE_API_KEY from environment variable. Consider moving to config.json.")
        return api_key

    # 4. Fallback to old GEMINI_API_KEY environment variable (deprecated)
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        logger.warning("Using deprecated GEMINI_API_KEY from environment variable. Consider renaming to GOOGLE_API_KEY and moving to config.json.")
        return api_key

    logger.error("GOOGLE_API_KEY or GEMINI_API_KEY not found in config.json or environment.")
    return None

def prepare_safety_settings(safety_harassment: str, safety_hate_speech: str,
                             safety_sexually_explicit: str, safety_dangerous_content: str) -> List[Union[SafetySettingDict, dict]]:
    """Builds the safety settings list from node inputs."""
    logger.debug("Preparing safety settings.")
    return [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": SAFETY_SETTINGS_MAP.get(safety_harassment, "HARM_BLOCK_THRESHOLD_UNSPECIFIED")},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": SAFETY_SETTINGS_MAP.get(safety_hate_speech, "HARM_BLOCK_THRESHOLD_UNSPECIFIED")},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": SAFETY_SETTINGS_MAP.get(safety_sexually_explicit, "HARM_BLOCK_THRESHOLD_UNSPECIFIED")},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": SAFETY_SETTINGS_MAP.get(safety_dangerous_content, "HARM_BLOCK_THRESHOLD_UNSPECIFIED")},
    ]

def prepare_generation_config(temperature: float, top_p: float, top_k: int,
                               max_output_tokens: int) -> Union[GenerationConfigDict, dict]:
    """Builds the generation configuration object or dictionary."""
    logger.debug("Preparing generation config.")
    config_data = {
        "temperature": temperature, "top_p": top_p,
        "top_k": top_k, "max_output_tokens": max_output_tokens,
    }
    if generation_types and GenerationConfigDict is not dict:
        try:
            # Use the specific type if available and not the fallback dict
            return generation_types.GenerationConfig(**config_data)
        except Exception as e:
             logger.warning(f"Error creating GenerationConfig object: {e}. Falling back to dict.")
    # Fallback if types weren't imported or instantiation failed
    logger.warning("Using dictionary for generation_config due to failed type import or instantiation error.")
    return config_data


def prepare_content_parts(prompt: str, image_optional: Optional[torch.Tensor], model_name: str) -> Tuple[List[Any], Optional[str]]:
    """
    Prepares the list of content parts (text and optional image),
    using types.Part if available, aligning with documentation examples.
    Handles image conversion and potential errors.
    """
    content_parts: List[Any] = [prompt]
    error_msg: Optional[str] = None

    if image_optional is not None:
        logger.debug("Processing optional image input.")
        # Check if tensor_to_pil is callable AND not the dummy function defined in the except block
        # Comparing __code__ is a way to check if it's the specific dummy function
        if callable(tensor_to_pil) and getattr(tensor_to_pil, '__code__', None) != (lambda *args, **kwargs: None).__code__:
            try:
                pil_image: Optional[PilImageT] = tensor_to_pil(image_optional)
                if pil_image:
                    logger.debug("Successfully converted image tensor to PIL Image.")
                    # Convert PIL Image to bytes and use types.Part.from_bytes if types is available
                    if types and hasattr(types, 'Part') and hasattr(types.Part, 'from_bytes'):
                        import io
                        img_byte_arr = io.BytesIO()
                        # Save as JPEG for common compatibility, quality 90
                        pil_image.save(img_byte_arr, format='JPEG', quality=90)
                        img_bytes = img_byte_arr.getvalue()
                        content_parts.append(types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'))
                        logger.debug("Added image as types.Part.from_bytes.")
                    else:
                        # Fallback to appending PIL Image directly if types.Part is not available
                        content_parts.append(pil_image)
                        logger.debug("Added image as PIL Image (types.Part not available).")
                else:
                    logger.error("tensor_to_pil returned None. Check input tensor format.")
                    error_msg = f"{ERROR_PREFIX} Image conversion failed: tensor_to_pil returned None. Check input tensor format."
            except Exception as img_e:
                logger.error(f"An unexpected error occurred during image processing: {img_e}", exc_info=True)
                error_msg = f"{ERROR_PREFIX} An unexpected error occurred during image processing: {type(img_e).__name__}: {img_e}"
        else:
            # This block is hit if tensor_to_pil is not callable or is the dummy function
            logger.error("tensor_to_pil function is not available or is a dummy due to import errors.")
            error_msg = f"{ERROR_PREFIX} Image processing utility is unavailable. Ensure 'shared_utils' is correctly installed/accessible."
    elif "vision" in model_name or "1.5" in model_name: # Broader check for multimodal models
        logger.warning(f"Multimodal model '{model_name}' selected, but no image provided.")

    return content_parts, error_msg

def process_api_response(response: Union[GenerateContentResponse, Any]) -> Tuple[str, Optional[str]]:
    """
    Processes the Gemini API response, extracting text or handling errors/blocks.
    Returns a tuple: (final_text_or_error, api_block_error_msg).
    """
    generated_text: str = ""
    error_msg: Optional[str] = None # For processing errors or API block messages

    try:
        # Use getattr for safe access, especially if response type is Any
        candidates = getattr(response, 'candidates', None)
        if not candidates:
            block_reason_code = "UNSPECIFIED"
            prompt_feedback_msg = "No prompt feedback available."
            prompt_feedback = getattr(response, 'prompt_feedback', None)
            if prompt_feedback:
                if hasattr(prompt_feedback, 'block_reason'):
                    block_reason_obj = getattr(prompt_feedback, 'block_reason', None)
                    block_reason_code = getattr(block_reason_obj, 'name', str(block_reason_obj))
                if hasattr(prompt_feedback, 'safety_ratings'):
                    ratings = getattr(prompt_feedback, 'safety_ratings', [])
                    ratings_str = ', '.join([f"{getattr(r.category, 'name', 'UNK')}: {getattr(r.probability, 'name', 'UNK')}" for r in ratings])
                    prompt_feedback_msg = f"Prompt Feedback Safety Ratings: [{ratings_str}]"
                elif hasattr(prompt_feedback, 'block_reason'):
                     prompt_feedback_msg = f"Prompt Block Reason: {block_reason_code}"
                else:
                     prompt_feedback_msg = f"Prompt Feedback: {prompt_feedback}"


            error_msg = f"{ERROR_PREFIX} Blocked/Failed: Generation failed. Reason: {block_reason_code}. {prompt_feedback_msg}"
            logger.error(error_msg)
            return error_msg, error_msg # Return error as both text and block message

        # Assume at least one candidate if the list exists
        candidate = candidates[0]
        finish_reason_obj = getattr(candidate, 'finish_reason', None)
        finish_reason_name = getattr(finish_reason_obj, 'name', str(finish_reason_obj))

        if finish_reason_name == 'SAFETY':
            ratings = getattr(candidate, 'safety_ratings', [])
            ratings_str = ', '.join([f"{getattr(r.category, 'name', 'UNK')}: {getattr(r.probability, 'name', 'UNK')}" for r in ratings])
            error_msg = f"{ERROR_PREFIX} Blocked: Response stopped by safety settings. Ratings: [{ratings_str}]"
            logger.error(error_msg)
            return error_msg, error_msg # Return error as both text and block message
        elif finish_reason_name == 'RECITATION':
            error_msg = f"{ERROR_PREFIX} Blocked: Response stopped for potential recitation."
            logger.error(error_msg)
            return error_msg, error_msg # Return error as both text and block message
        elif finish_reason_name == 'MAX_TOKENS':
            logger.warning("Generation stopped: Reached max_output_tokens limit.")
        elif finish_reason_name not in ['STOP', 'UNSPECIFIED', 'FINISH_REASON_UNSPECIFIED', None]:
             logger.warning(f"Generation finished with reason: {finish_reason_name}")

        content = getattr(candidate, 'content', None)
        if content and getattr(content, 'parts', None):
            parts_list = getattr(content, 'parts', [])
            generated_text = "".join(getattr(part, 'text', '') for part in parts_list)
            if generated_text:
                logger.info(f"Successfully generated text (length: {len(generated_text)}). Finish Reason: {finish_reason_name}")
            else:
                logger.warning(f"Response received with parts, but no text extracted. Finish Reason: {finish_reason_name}")
                generated_text = f"Response received, but no text content found (Finish Reason: {finish_reason_name})."
        else:
            status_msg = f"Response received but no valid content parts found. Finish Reason: {finish_reason_name}"
            logger.warning(status_msg)
            generated_text = status_msg

    except (IndexError, AttributeError, TypeError) as e:
         logger.error(f"Error accessing response structure: {type(e).__name__}: {e}. Check API response format.", exc_info=True)
         error_msg = f"{ERROR_PREFIX} Error parsing API response: {type(e).__name__}."
         return error_msg, None # Return processing error, no specific API block

    # If we reached here without an error_msg, it means generation was successful (or stopped for non-blocking reasons)
    final_text = generated_text if not error_msg else error_msg
    return final_text, error_msg # Return text (or processing error) and potential block message

def generate_content_via_client(
    api_key: str,
    model_name: str,
    contents: List[Any],
    generation_config: Union[GenerationConfigDict, dict],
    safety_settings: List[Union[SafetySettingDict, dict]]
) -> Tuple[str, Optional[str]]:
    """
    Generates content using genai.Client and processes the response.
    Returns a tuple: (final_text_or_error, api_block_error_msg).
    """
    logger.info(f"Generating content for model: {model_name} using genai.Client")
    final_output = ""
    response_error_msg: Optional[str] = None

    if not api_key:
        error_msg = f"{ERROR_PREFIX} API key not provided for generation."
        logger.error(error_msg)
        return error_msg, error_msg

    try:
        # Instantiate client with the provided API key
        client = genai.Client(api_key=api_key)
        logger.debug("genai.Client instantiated for content generation.")

        # Call client.models.generate_content
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        # Process Response using utility function
        processed_text, response_error_msg = process_api_response(response)
        # Prioritize showing the API block error message if it exists
        final_output = response_error_msg if response_error_msg else processed_text

    except google_exceptions.GoogleAPIError as e:
         if google_exceptions:
             error_msg = f"{ERROR_PREFIX} Google API Error - Status: {getattr(e, 'code', 'N/A')}, Message: {e}"
             logger.error(error_msg, exc_info=True)
             final_output = f"{ERROR_PREFIX} A Google API error occurred ({getattr(e, 'code', 'N/A')}). Check console logs."
         else:
             error_msg = f"{ERROR_PREFIX} An API error occurred: {e}"
             logger.error(error_msg, exc_info=True)
             final_output = f"{ERROR_PREFIX} An API error occurred. Check console logs."

    except Exception as e:
        error_details = str(e)
        if hasattr(e, 'message') and e.message: error_details = e.message
        elif hasattr(e, 'details') and callable(e.details) and e.details(): error_details = e.details()
        error_msg = f"{ERROR_PREFIX} Gemini Generation Error ({type(e).__name__}): {error_details}"
        logger.error(error_msg, exc_info=True)
        final_output = error_msg # Always use the formatted error for consistency

    # Ensure final_output is always a string before returning
    if not isinstance(final_output, str):
        logger.error(f"Final output was not a string ({type(final_output)}), converting. Value: {final_output}")
        final_output = str(final_output)

    logger.info("Content generation finished.")
    return final_output, response_error_msg # Return text (or processing error) and potential block message
