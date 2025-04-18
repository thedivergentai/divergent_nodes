"""
Utility functions and constants for the Gemini API node.
"""
import os
import logging
from typing import Optional, Dict, Any, Tuple, List, Union, TypeAlias
from PIL import Image
import torch
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv

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
    "models/gemini-1.5-pro-latest",
    "models/gemini-1.5-flash-latest",
    "models/gemini-1.0-pro",
    "models/gemini-pro-vision",
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

def get_available_models() -> List[str]:
    """Fetches available Gemini models supporting 'generateContent'."""
    logger.debug("Attempting to fetch available Gemini models...")
    api_key_found = False
    try:
        load_dotenv(find_dotenv(), override=True)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not found. Cannot fetch dynamic model list.")
            return DEFAULT_MODELS
        api_key_found = True
        genai.configure(api_key=api_key)
        model_list: List[str] = [
            m.name for m in genai.list_models()
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
        log_func = logger.error if api_key_found else logger.warning
        log_func(f"Failed to fetch models from Gemini API: {e}. Using default list.", exc_info=True)
        return DEFAULT_MODELS

def configure_api_key() -> Optional[str]:
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

def initialize_model(model_name: str, safety_settings: List[Union[SafetySettingDict, dict]],
                      generation_config: Union[GenerationConfigDict, dict]) -> Union[GenerativeModel, Any]:
    """Initializes the GenerativeModel instance."""
    logger.info(f"Initializing Gemini model: {model_name}")
    try:
        # Pass configs directly, type safety handled by genai library internally
        # The actual type returned depends on whether the specific GenerativeModel was imported
        return genai.GenerativeModel(
            model_name=model_name,
            safety_settings=safety_settings,
            generation_config=generation_config
        )
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model '{model_name}': {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize Gemini model: {e}") from e

def prepare_content_parts(prompt: str, image_optional: Optional[torch.Tensor], model_name: str) -> Tuple[List[Any], Optional[str]]:
    """Prepares the list of content parts (text and optional image)."""
    content_parts: List[Any] = [prompt]
    error_msg: Optional[str] = None

    if image_optional is not None:
        logger.debug("Processing optional image input.")
        try:
            if callable(tensor_to_pil):
                pil_image: Optional[PilImageT] = tensor_to_pil(image_optional)
                if pil_image:
                    logger.debug("Successfully converted image tensor to PIL Image.")
                    content_parts.append(pil_image)
                else:
                    logger.warning("tensor_to_pil returned None. Check input tensor format.")
            else:
                logger.error("tensor_to_pil function is not available due to import errors.")
                error_msg = f"{ERROR_PREFIX} Image processing utility is unavailable."
        except Exception as img_e:
            logger.error(f"Failed to convert input tensor to PIL Image: {img_e}", exc_info=True)
            error_msg = f"{ERROR_PREFIX} Failed to process input image: {img_e}"
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
            prompt_feedback = getattr(response, 'prompt_feedback', None)
            if prompt_feedback and hasattr(prompt_feedback, 'block_reason'):
                block_reason_obj = getattr(prompt_feedback, 'block_reason', None)
                block_reason_code = getattr(block_reason_obj, 'name', str(block_reason_obj))

            error_msg = f"{ERROR_PREFIX} Blocked/Failed: Generation failed. Block Reason: {block_reason_code}."
            logger.error(error_msg + " Check prompt feedback for details.")
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
