"""
Utility functions and constants for the Google Gemini API node.
Includes functions for API key configuration, content preparation,
generation with advanced options, and model listing.
Uses the recommended google-genai library and follows GitHub README examples.
"""
import os
import logging
import io
from typing import Optional, Dict, Any, Tuple, List, Union, TypeAlias
from PIL import Image
import torch
from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions

# Import the new config manager
from ..shared_utils.config_manager import load_config
from ..shared_utils.image_conversion import tensor_to_pil # Assuming this utility is stable

# Setup logger for this module
logger = logging.getLogger(__name__)
# Ensure handler is configured if root logger isn't set up
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Type Aliases ---
PilImageT: TypeAlias = Image.Image

# --- Constants ---
# IMPORTANT: DO NOT DOWNGRADE THIS LIST. This list represents the preferred set of models,
# regardless of what the API might dynamically return or what documentation might suggest
# as a default. New models can be added, but existing ones should not be removed unless
# explicitly instructed by the user.
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

def configure_api_key(api_key_override: Optional[str] = None) -> Optional[str]:
    """
    Determines the Gemini API key to use (override > config.json > environment).
    Returns the key string.
    """
    logger.debug("configure_api_key called.")

    # 1. Check override first
    if api_key_override and api_key_override.strip():
        api_key = api_key_override.strip()
        logger.debug(f"Using API key from override: {api_key[:4]}...") # Log first few chars
        return api_key

    # 2. Load config from config.json
    config = load_config()
    api_key = config.get("GOOGLE_API_KEY")
    if api_key:
        logger.debug(f"GOOGLE_API_KEY found in config.json: {api_key[:4]}...") # Log first few chars
        return api_key

    # 3. Fallback to environment variable (less preferred)
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        logger.warning("Using GOOGLE_API_KEY from environment variable. Consider moving to config.json.")
        return api_key

    # 4. Fallback to old GEMINI_API_KEY environment variable (deprecated)
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        logger.warning(f"Using deprecated GEMINI_API_KEY from environment variable: {api_key[:4]}... Consider renaming to GOOGLE_API_KEY and moving to config.json.") # Log first few chars
        return api_key

    logger.error("GOOGLE_API_KEY or GEMINI_API_KEY not found in config.json or environment.")
    return None

def prepare_safety_settings(safety_harassment: str, safety_hate_speech: str,
                             safety_sexually_explicit: str, safety_dangerous_content: str) -> List[types.SafetySetting]:
    """Builds the safety settings list from node inputs using types.SafetySetting and strings."""
    logger.debug("Preparing safety settings.")
    settings = []
    # Map string inputs to the string values expected by types.SafetySetting
    category_map = {
        "Harassment": "HARM_CATEGORY_HARASSMENT",
        "Hate speech": "HARM_CATEGORY_HATE_SPEECH",
        "Sexually explicit": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "Dangerous": "HARM_CATEGORY_DANGEROUS_CONTENT",
    }
    threshold_map = {
        "Default (Unspecified)": "HARM_BLOCK_THRESHOLD_UNSPECIFIED",
        "Block Low & Above": "BLOCK_LOW_AND_ABOVE",
        "Block Medium & Above": "BLOCK_MEDIUM_AND_ABOVE",
        "Block High Only": "BLOCK_ONLY_HIGH",
        "Block None": "BLOCK_NONE",
    }

    # Use .get() with a default to handle potential missing keys gracefully
    settings.append(types.SafetySetting(
        category=category_map.get("Harassment", "HARM_CATEGORY_UNSPECIFIED"),
        threshold=threshold_map.get(safety_harassment, "HARM_BLOCK_THRESHOLD_UNSPECIFIED")
    ))
    settings.append(types.SafetySetting(
        category=category_map.get("Hate speech", "HARM_CATEGORY_UNSPECIFIED"),
        threshold=threshold_map.get(safety_hate_speech, "HARM_BLOCK_THRESHOLD_UNSPECIFIED")
    ))
    settings.append(types.SafetySetting(
        category=category_map.get("Sexually explicit", "HARM_CATEGORY_UNSPECIFIED"),
        threshold=threshold_map.get(safety_sexually_explicit, "HARM_BLOCK_THRESHOLD_UNSPECIFIED")
    ))
    settings.append(types.SafetySetting(
        category=category_map.get("Dangerous", "HARM_CATEGORY_UNSPECIFIED"),
        threshold=threshold_map.get(safety_dangerous_content, "HARM_BLOCK_THRESHOLD_UNSPECIFIED")
    ))

    return settings

def prepare_generation_config(temperature: float, top_p: float, top_k: int,
                               max_output_tokens: int) -> types.GenerateContentConfig:
    """Builds the generation configuration object using types.GenerateContentConfig."""
    logger.debug("Preparing generation config.")
    # Create types.GenerateContentConfig object directly
    return types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_output_tokens=max_output_tokens,
    )


def prepare_image_part(image_tensor: torch.Tensor) -> Tuple[Optional[types.Part], Optional[str]]:
    """
    Converts a ComfyUI image tensor to a Gemini API types.Part for inline use.
    Returns a tuple: (image_part, error_message).
    """
    logger.debug("Preparing image part from tensor.")
    try:
        pil_image: Optional[PilImageT] = tensor_to_pil(image_tensor)
        if pil_image:
            logger.debug("Successfully converted image tensor to PIL Image.")
            img_byte_arr = io.BytesIO()
            # Save as JPEG for common compatibility
            pil_image.save(img_byte_arr, format='JPEG', quality=90)
            img_bytes = img_byte_arr.getvalue()
            # Use types.Part.from_bytes as shown in documentation
            image_part = types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg')
            logger.debug("Created image types.Part.")
            return image_part, None
        else:
            error_msg = f"{ERROR_PREFIX} Image conversion failed: tensor_to_pil returned None. Check input tensor format."
            logger.error(error_msg)
            return None, error_msg
    except Exception as e:
        error_msg = f"{ERROR_PREFIX} Error during image processing for API: {type(e).__name__}: {e}"
        logger.error(error_msg, exc_info=True)
        return None, error_msg

def generate_content(
    api_key: str,
    model_name: str,
    prompt: str,
    image_part: Optional[types.Part] = None,
    generation_config: Optional[types.GenerateContentConfig] = None, # Expect types.GenerateContentConfig
    safety_settings: Optional[List[types.SafetySetting]] = None, # Expect List[types.SafetySetting]
) -> Tuple[str, Optional[str]]:
    """
    Generates content using genai.Client based on text and optional image parts,
    with optional generation config and safety settings.
    Returns a tuple: (generated_text_or_error, api_block_error_msg).
    """
    logger.info(f"Generating content for model: {model_name}")
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

        # Prepare contents list based on documentation examples
        contents: List[Any] = []
        if image_part:
            contents.append(image_part)
        contents.append(prompt) # Add prompt after image as per best practices

        # Create the single GenerateContentConfig object
        # Combine generation_config and safety_settings here
        full_config = types.GenerateContentConfig()
        if generation_config:
             # Copy parameters from the prepared generation_config object
             full_config.temperature = generation_config.temperature
             full_config.top_p = generation_config.top_p
             full_config.top_k = generation_config.top_k
             full_config.max_output_tokens = generation_config.max_output_tokens
             # Add other generation config parameters if needed
        if safety_settings:
             full_config.safety_settings = safety_settings # Assign the list of SafetySetting objects


        logger.debug(f"Sending request to Gemini API model '{model_name}'...")
        # Use client.models.generate_content with the single config argument
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=full_config, # Pass the combined config object
        )

        # Process Response
        # This part is adapted from the previous process_api_response logic
        generated_text: str = ""
        try:
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

                response_error_msg = f"{ERROR_PREFIX} Blocked/Failed: Generation failed. Reason: {block_reason_code}. {prompt_feedback_msg}"
                logger.error(response_error_msg)
                return response_error_msg, response_error_msg # Return error as both text and block message

            candidate = candidates[0]
            finish_reason_obj = getattr(candidate, 'finish_reason', None)
            finish_reason_name = getattr(finish_reason_obj, 'name', str(finish_reason_obj))

            if finish_reason_name == 'SAFETY':
                ratings = getattr(candidate, 'safety_ratings', [])
                ratings_str = ', '.join([f"{getattr(r.category, 'name', 'UNK')}: {getattr(r.probability, 'name', 'UNK')}" for r in ratings])
                response_error_msg = f"{ERROR_PREFIX} Blocked: Response stopped by safety settings. Ratings: [{ratings_str}]"
                logger.error(response_error_msg)
                return response_error_msg, response_error_msg # Return error as both text and block message
            elif finish_reason_name == 'RECITATION':
                response_error_msg = f"{ERROR_PREFIX} Blocked: Response stopped for potential recitation."
                logger.error(response_error_msg)
                return response_error_msg, response_error_msg # Return error as both text and block message
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
             response_error_msg = f"{ERROR_PREFIX} Error parsing API response: {type(e).__name__}."
             return response_error_msg, None # Return processing error, no specific API block

        final_output = generated_text if not response_error_msg else response_error_msg

    except google_exceptions.GoogleAPIError as e:
         if google_exceptions:
             error_msg = f"{ERROR_PREFIX} Google API Error - Status: {getattr(e, 'code', 'N/A')}, Message: {e}"
             logger.error(error_msg, exc_info=True)
             final_output = f"{ERROR_PREFIX} A Google API error occurred ({getattr(e, 'code', 'N/A')}). Check console logs."
         else:
             # Fallback if specific exception types aren't available
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

# Note: get_available_models is still needed for INPUT_TYPES in the node file.
# It should also use genai.Client(api_key=api_key).models.list()
def get_available_models(api_key: Optional[str]) -> List[str]:
    """Fetches available Gemini models supporting 'generateContent' using genai.Client."""
    logger.debug("Attempting to fetch available Gemini models...")
    if not api_key:
         logger.warning("API key not provided. Cannot fetch dynamic model list.")
         return DEFAULT_MODELS

    try:
        # Instantiate client with the provided API key
        client = genai.Client(api_key=api_key)
        logger.debug("genai.Client instantiated for model listing.")

        # Use client.models.list() as shown in documentation
        model_list: List[str] = [
            m.name for m in client.models.list()
            if 'generateContent' in m.supported_actions # Corrected attribute name
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
