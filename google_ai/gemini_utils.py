"""
Utility functions and constants for the Google Gemini API node.
Includes functions for API key configuration, content preparation,
generation with advanced options, and model listing.
Uses the recommended google-genai library and follows GitHub README examples.
"""
import os
import logging
import io
import hashlib
from typing import Optional, Dict, Any, Tuple, List, Union, TypeAlias
from PIL import Image
import torch
from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions
from google.generativeai import GenerativeModel # Added import for GenerativeModel

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
                               max_output_tokens: int,
                               thinking_config: Optional[types.ThinkingConfig] = None) -> types.GenerateContentConfig:
    """Builds the generation configuration object using types.GenerateContentConfig."""
    logger.debug("Preparing generation config.")
    # Create types.GenerateContentConfig object directly, including thinking_config
    return types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_output_tokens=max_output_tokens,
        thinking_config=thinking_config # Pass thinking_config here
    )

def prepare_thinking_config(extended_thinking: bool, thinking_budget: int, output_thoughts: bool) -> types.ThinkingConfig:
    """
    Prepares the thinking configuration object for the Gemini API.
    Controls whether the model performs internal thinking and whether those thoughts are included in the API response.
    """
    # 'include_thoughts' in the API request should be True only if both extended_thinking is True
    # AND output_thoughts is True. Otherwise, it should be False to prevent thoughts from being returned by the API.
    include_thoughts_in_api_request = extended_thinking and output_thoughts

    config = {"include_thoughts": include_thoughts_in_api_request}
    
    # If thinking_budget is -1, the model determines the budget.
    # If thinking_budget is 0, it implies no thinking, so include_thoughts should be False (handled above).
    # If thinking_budget is > 0, it sets a specific budget.
    if thinking_budget != -1:
        config["thinking_budget"] = thinking_budget

    thinking_config = types.ThinkingConfig(**config)
    logger.debug(f"Prepared thinking config: {thinking_config}")
    return thinking_config

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
    contents: List[Any],
    generation_config: Optional[types.GenerateContentConfig] = None,
    safety_settings: Optional[List[types.SafetySetting]] = None,
    thinking_config: Optional[types.ThinkingConfig] = None,
    max_retries: int = 3,
    retry_delay_seconds: int = 5,
    cached_context: str = "",
    output_thoughts: bool = False # Added new parameter
) -> Tuple[str, Optional[str], int, int, int]:
    """
    Generates content using genai.Client based on a list of content parts,
    with optional generation config and safety settings, and retry logic.
    This function now uses streaming to enforce max_output_tokens on the response text.
    Returns a tuple: (generated_text_or_error, api_block_error_msg, prompt_tokens, response_tokens, thoughts_tokens).
    """
    logger.info(f"Generating content for model: {model_name} with {max_retries} retries.")
    response_error_msg: Optional[str] = None
    current_retry = 0
    prompt_tokens = 0
    response_tokens = 0
    thoughts_tokens = 0

    if not api_key:
        error_msg = f"{ERROR_PREFIX} API key not provided for generation."
        logger.error(error_msg)
        return error_msg, error_msg, prompt_tokens, response_tokens, thoughts_tokens

    while current_retry <= max_retries:
        try:
            client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
            logger.debug("genai.Client instantiated for content generation with v1alpha API.")

            full_config = types.GenerateContentConfig()
            if generation_config:
                full_config.temperature = generation_config.temperature
                full_config.top_p = generation_config.top_p
                full_config.top_k = generation_config.top_k
                # max_output_tokens is now handled client-side with streaming
                # We still pass it to the API as a fallback/hint, but the client-side logic will enforce it.
                full_config.max_output_tokens = generation_config.max_output_tokens
            if safety_settings:
                full_config.safety_settings = safety_settings
            if thinking_config:
                full_config.thinking_config = thinking_config

            logger.debug(f"Sending request to Gemini API model '{model_name}' (Attempt {current_retry + 1}/{max_retries + 1})...")
            
            # Use streaming to control output tokens
            if cached_context:
                cache_display_name = hashlib.sha256(cached_context.encode()).hexdigest()
                cached_content = genai.CachedContent.get(f"cachedContents/{cache_display_name}")
                if not cached_content:
                    cached_content = genai.CachedContent.create(
                        model=model_name,
                        display_name=cache_display_name,
                        contents=[cached_context],
                    )
                response_stream = cached_content.generate_content(
                    contents=contents,
                    generation_config=full_config,
                    safety_settings=safety_settings,
                    stream=True
                )
            else:
                response_stream = client.models.generate_content_stream(
                    model=model_name,
                    contents=contents,
                    config=full_config,
                )

            full_response_text_list = []
            current_response_tokens = 0
            
            # Create a dummy model object to count tokens client-side
            # This is a lightweight way to access the tokenizer for accurate token counting
            # Note: This might not be perfectly aligned with API's internal tokenizer, but it's the best client-side approximation.
            token_counter_model = GenerativeModel(model_name=model_name)

            final_response_object = None # To store the last chunk which contains usage_metadata

            for chunk in response_stream:
                final_response_object = chunk # Keep track of the last chunk for usage_metadata
                
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    for part in chunk.candidates[0].content.parts:
                        if part.text:
                            # Count tokens of the current chunk's text
                            chunk_token_count = token_counter_model.count_tokens(part.text).total_tokens
                            
                            # Check if adding this chunk exceeds the max_output_tokens
                            if generation_config and (current_response_tokens + chunk_token_count > generation_config.max_output_tokens):
                                logger.warning(f"Max output tokens ({generation_config.max_output_tokens}) reached. Stopping stream.")
                                break # Stop processing further chunks
                            else:
                                full_response_text_list.append(part.text)
                                current_response_tokens += chunk_token_count
                        elif part.thought:
                            logger.debug("Received thought part (content suppressed from output).") # Log thought parts, but don't add to response
                        else:
                            logger.warning(f"Received part with no text or thought. Type: {type(part)}, Attributes: {dir(part)}")
                else:
                    logger.warning(f"Chunk has no candidates or candidates list is empty. Chunk: {chunk}")
                
                if generation_config and current_response_tokens >= generation_config.max_output_tokens:
                    break # Stop processing further chunks if max output tokens reached

            generated_text = "".join(full_response_text_list)

            # After the stream, extract final token counts from the last chunk's usage_metadata
            logger.debug(f"Final response object before usage_metadata extraction: {final_response_object}")
            if final_response_object and hasattr(final_response_object, 'usage_metadata'):
                usage_metadata = final_response_object.usage_metadata
                prompt_tokens = getattr(usage_metadata, 'prompt_token_count', 0)
                response_tokens = getattr(usage_metadata, 'response_token_count', 0) # Use API's response_token_count
                thoughts_tokens = getattr(usage_metadata, 'thoughts_token_count', 0)
                logger.info(f"Final Token Usage: Prompt={prompt_tokens}, API-Response={response_tokens}, API-Thoughts={thoughts_tokens}")
            else:
                logger.warning("Could not retrieve usage_metadata from the streamed response. Final response object might be incomplete or missing metadata.")
                # Fallback: estimate response tokens from generated_text if metadata is missing
                if generated_text:
                    response_tokens = token_counter_model.count_tokens(generated_text).total_tokens
                logger.info(f"Estimated Token Usage (no metadata): Client-Response={response_tokens}")


            # Check for safety blocks or other finish reasons from the last chunk
            finish_reason_str = "UNKNOWN"
            prompt_feedback = None
            if final_response_object and hasattr(final_response_object, 'candidates') and final_response_object.candidates:
                candidate = final_response_object.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    finish_reason_str = getattr(candidate.finish_reason, 'name', 'STOP')
                if hasattr(final_response_object, 'prompt_feedback'):
                    prompt_feedback = final_response_object.prompt_feedback

            if prompt_feedback and hasattr(prompt_feedback, 'block_reason') and prompt_feedback.block_reason:
                finish_reason_str = getattr(prompt_feedback.block_reason, 'name', 'OTHER')
            
            ratings = (getattr(prompt_feedback, 'safety_ratings', []) if prompt_feedback else []) or []
            ratings_str = ', '.join([f"{getattr(r.category, 'name', 'UNK')}: {getattr(r.probability, 'name', 'UNK')}" for r in ratings if r])

            if finish_reason_str == 'SAFETY':
                response_error_msg = f"{ERROR_PREFIX} Blocked: Response stopped by safety settings. Ratings: [{ratings_str}]"
                logger.error(response_error_msg)
                return response_error_msg, response_error_msg, prompt_tokens, current_response_tokens, thoughts_tokens
            elif finish_reason_str == 'RECITATION':
                response_error_msg = f"{ERROR_PREFIX} Blocked: Response stopped for potential recitation."
                logger.error(response_error_msg)
                return response_error_msg, response_error_msg, prompt_tokens, current_response_tokens, thoughts_tokens
            elif finish_reason_str == 'MAX_TOKENS':
                logger.warning("Generation stopped by API: Reached max_output_tokens limit.")
                # This case should be less frequent now due to client-side truncation,
                # but can still happen if API's internal tokenization differs or for other reasons.
            elif finish_reason_str not in ['STOP', 'UNSPECIFIED', 'FINISH_REASON_UNSPECIFIED', 'OTHER', 'UNKNOWN', None]:
                logger.warning(f"Generation finished with reason: {finish_reason_str}")

            if generated_text:
                logger.info(f"Successfully generated text (tokens: {current_response_tokens}, chars: {len(generated_text)}). Finish Reason: {finish_reason_str}")
                return generated_text, None, prompt_tokens, current_response_tokens, thoughts_tokens # Success, return immediately
            else:
                status_msg = f"Response received with parts, but no text extracted. Finish Reason: {finish_reason_str}. Retrying..."
                logger.warning(status_msg)
                raise RuntimeError(status_msg) # Raise to trigger retry logic

        except google_exceptions.GoogleAPIError as e:
            error_msg = f"{ERROR_PREFIX} Google API Error - Status: {getattr(e, 'code', 'N/A')}, Message: {e}. Retrying..."
            logger.error(error_msg, exc_info=True)
            pass # Allow loop to continue for retry

        except Exception as e:
            error_details = str(e)
            if hasattr(e, 'message') and e.message: error_details = e.message
            elif hasattr(e, 'details') and callable(e.details) and e.details(): error_details = e.details()
            error_msg = f"{ERROR_PREFIX} Gemini Generation Error ({type(e).__name__}): {error_details}. Retrying..."
            logger.error(error_msg, exc_info=True)
            pass # Allow loop to continue for retry

        current_retry += 1
        if current_retry <= max_retries:
            logger.info(f"Waiting {retry_delay_seconds} seconds before retry {current_retry}/{max_retries}...")
            import time
            time.sleep(retry_delay_seconds)

    # If loop finishes, all retries exhausted
    final_error_msg = f"{ERROR_PREFIX} All {max_retries} retries failed. Last error: {error_msg if 'error_msg' in locals() else 'Unknown error.'}"
    logger.error(final_error_msg)
    return final_error_msg, final_error_msg, prompt_tokens, response_tokens, thoughts_tokens # Return final error after all retries


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
