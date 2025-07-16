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
        category=category_map.get("Sexually explicit", "HARM_CATEGORY_SEXUALLY_EXPLICIT"),
        threshold=threshold_map.get(safety_sexually_explicit, "HARM_BLOCK_THRESHOLD_UNSPECIFIED")
    ))
    settings.append(types.SafetySetting(
        category=category_map.get("Dangerous", "HARM_CATEGORY_DANGEROUS_CONTENT"),
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


# prepare_image_part function is removed as per refactoring plan

def prepare_content_parts(prompt: str) -> Tuple[List[Any], Optional[str]]:
    """
    Prepares the initial content parts list for the Gemini API generate_content method,
    primarily handling text. Image parts are expected to be added by the calling node.
    Returns a tuple: (content_parts_list, error_message).
    """
    logger.debug("Preparing content parts.")
    contents: List[Any] = []
    error_msg: Optional[str] = None

    # Add text prompt
    contents.append(prompt)

    logger.debug(f"Prepared initial contents: {contents}")
    return contents, error_msg

def generate_content(
    api_key: str,
    model_name: str,
    contents: List[Any], # Accept contents list directly
    generation_config: Optional[types.GenerateContentConfig] = None, # Expect types.GenerateContentConfig
    safety_settings: Optional[List[types.SafetySetting]] = None, # Expect List[types.SafetySetting]
    max_retries: int = 3, # New parameter for retry count
    retry_delay_seconds: int = 5 # New parameter for delay between retries
) -> Tuple[str, Optional[str]]:
    """
    Generates content using genai.Client based on a list of content parts,
    with optional generation config and safety settings, and retry logic.
    Returns a tuple: (generated_text_or_error, api_block_error_msg).
    """
    logger.info(f"Generating content for model: {model_name} with {max_retries} retries.")
    final_output = ""
    response_error_msg: Optional[str] = None
    current_retry = 0

    if not api_key:
        error_msg = f"{ERROR_PREFIX} API key not provided for generation."
        logger.error(error_msg)
        return error_msg, error_msg

    while current_retry <= max_retries:
        try:
            # Instantiate client with the provided API key
            client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
            logger.debug("genai.Client instantiated for content generation with v1alpha API.")

            # Create the single GenerateContentConfig object
            full_config = types.GenerateContentConfig()
            if generation_config:
                full_config.temperature = generation_config.temperature
                full_config.top_p = generation_config.top_p
                full_config.top_k = generation_config.top_k
                full_config.max_output_tokens = generation_config.max_output_tokens
            if safety_settings:
                full_config.safety_settings = safety_settings

            logger.debug(f"Sending request to Gemini API model '{model_name}' (Attempt {current_retry + 1}/{max_retries + 1})...")
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=full_config,
            )

            # Process Response
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
                    # This is a content block, not a transient API error, so no retry
                    return response_error_msg, response_error_msg

                candidate = candidates[0]
                finish_reason_obj = getattr(candidate, 'finish_reason', None)
                finish_reason_name = getattr(finish_reason_obj, 'name', str(finish_reason_obj))

                if finish_reason_name == 'SAFETY':
                    raw_ratings = getattr(candidate, 'safety_ratings', None) # Get raw value, could be None
                    ratings = raw_ratings if raw_ratings is not None else [] # Ensure it's a list, even if raw is None
                    try:
                        # Filter out None elements from ratings before joining
                        ratings_str = ', '.join([
                            f"{getattr(r.category, 'name', 'UNK')}: {getattr(r.probability, 'name', 'UNK')}"
                            for r in ratings if r is not None
                        ])
                    except (TypeError, AttributeError) as e:
                        ratings_str = f"Error parsing safety ratings: {type(e).__name__}: {e}"
                        logger.error(f"Failed to parse safety ratings: {e}", exc_info=True)
                    response_error_msg = f"{ERROR_PREFIX} Blocked: Response stopped by safety settings. Ratings: [{ratings_str}]"
                    logger.error(response_error_msg)
                    # Content block, no retry
                    return response_error_msg, response_error_msg
                elif finish_reason_name == 'RECITATION':
                    response_error_msg = f"{ERROR_PREFIX} Blocked: Response stopped for potential recitation."
                    logger.error(response_error_msg)
                    # Content block, no retry
                    return response_error_msg, response_error_msg
                elif finish_reason_name == 'MAX_TOKENS':
                    logger.warning("Generation stopped: Reached max_output_tokens limit.")
                elif finish_reason_name not in ['STOP', 'UNSPECIFIED', 'FINISH_REASON_UNSPECIFIED', None]:
                    logger.warning(f"Generation finished with reason: {finish_reason_name}")

                content = getattr(candidate, 'content', None)
                if content is None:
                    status_msg = f"Response received but candidate content is None. Finish Reason: {finish_reason_name}. Retrying..."
                    logger.warning(status_msg)
                    raise RuntimeError(status_msg) # Raise to trigger retry logic

                if not getattr(content, 'parts', None):
                    status_msg = f"Response received but no valid content parts found. Finish Reason: {finish_reason_name}. Retrying..."
                    logger.warning(status_msg)
                    raise RuntimeError(status_msg) # Raise to trigger retry logic

                parts_list = getattr(content, 'parts', [])
                generated_text = "".join(getattr(part, 'text', '') for part in parts_list)
                if generated_text:
                    logger.info(f"Successfully generated text (length: {len(generated_text)}). Finish Reason: {finish_reason_name}")
                    return generated_text, None # Success, return immediately
                else:
                    status_msg = f"Response received with parts, but no text extracted. Finish Reason: {finish_reason_name}. Retrying..."
                    logger.warning(status_msg)
                    raise RuntimeError(status_msg) # Raise to trigger retry logic

            except (IndexError, AttributeError, TypeError) as e:
                error_msg = f"{ERROR_PREFIX} Error parsing API response: {type(e).__name__}: {e}. Retrying..."
                logger.error(error_msg, exc_info=True)
                # This is a transient issue, retry
                raise RuntimeError(error_msg) # Raise to trigger retry logic

        except google_exceptions.GoogleAPIError as e:
            error_msg = f"{ERROR_PREFIX} Google API Error - Status: {getattr(e, 'code', 'N/A')}, Message: {e}. Retrying..."
            logger.error(error_msg, exc_info=True)
            # This is a transient issue, retry
            pass # Allow loop to continue for retry

        except Exception as e:
            error_details = str(e)
            if hasattr(e, 'message') and e.message: error_details = e.message
            elif hasattr(e, 'details') and callable(e.details) and e.details(): error_details = e.details()
            error_msg = f"{ERROR_PREFIX} Gemini Generation Error ({type(e).__name__}): {error_details}. Retrying..."
            logger.error(error_msg, exc_info=True)
            # This is a transient issue, retry
            pass # Allow loop to continue for retry

        current_retry += 1
        if current_retry <= max_retries:
            logger.info(f"Waiting {retry_delay_seconds} seconds before retry {current_retry}/{max_retries}...")
            import time
            time.sleep(retry_delay_seconds)

    # If loop finishes, all retries exhausted
    final_error_msg = f"{ERROR_PREFIX} All {max_retries} retries failed. Last error: {error_msg if 'error_msg' in locals() else 'Unknown error.'}"
    logger.error(final_error_msg)
    return final_error_msg, final_error_msg # Return final error after all retries


# This function is now designed to be called at node initialization (INPUT_TYPES)
# It attempts to fetch models if an API key is available, otherwise uses a default list.
def get_available_models_robust(api_key: Optional[str]) -> List[str]:
    """
    Fetches available Gemini models supporting 'generateContent' using genai.Client.
    If API key is invalid or fetching fails, it falls back to DEFAULT_MODELS.
    """
    logger.debug("Attempting to fetch available Gemini models robustly...")
    
    if not api_key:
        logger.info("No API key provided at startup. Using default model list.")
        return DEFAULT_MODELS

    try:
        # Configure the genai library with the API key and API version
        genai.configure(api_key=api_key, transport='rest', http_options={'api_version': 'v1alpha'})
        logger.debug("genai configured for model listing with v1alpha API.")

        # Use genai.list_models() to fetch models
        # Filter for models that support generateContent
        model_list: List[str] = [
            m.name for m in genai.list_models()
            if 'generateContent' in m.supported_actions
        ]
        
        if not model_list:
            logger.warning("No models supporting 'generateContent' found via API. Using default list.")
            return DEFAULT_MODELS
        
        # Sort models for consistent display, putting 'latest' versions first
        model_list.sort(key=lambda x: ('latest' not in x, x))
        logger.info(f"Successfully fetched {len(model_list)} models from Gemini API.")
        return model_list
    
    except google_exceptions.GoogleAPIError as e:
        error_msg = f"Error fetching Gemini models: {getattr(e, 'code', 'N/A')} {e.message}"
        logger.error(error_msg, exc_info=True)
        logger.warning("Failed to fetch models from Gemini API due to API error. Using default list.")
        return DEFAULT_MODELS
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching Gemini models: {e}", exc_info=True)
        logger.warning("Failed to fetch models from Gemini API due to unexpected error. Using default list.")
        return DEFAULT_MODELS
