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
# Import the new config manager
from ..shared_utils.config_manager import load_config
from ..shared_utils.image_conversion import tensor_to_pil # Assuming this utility is stable
from ..shared_utils.logging_utils import SUCCESS_HIGHLIGHT # Import SUCCESS_HIGHLIGHT

# Setup logger for this module
logger = logging.getLogger(__name__)

# Suppress verbose HTTP request logging from google-api-core and its sub-loggers
# Setting the top-level 'google.api_core' logger to ERROR will silence most verbose HTTP/RPC logs.
logging.getLogger('google.api_core').setLevel(logging.ERROR)


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
        logger.info(f"🔑 Using API key from node override: {api_key[:4]}...") # Log first few chars
        return api_key

    # 2. Load config from config.json
    config = load_config()
    api_key = config.get("GOOGLE_API_KEY")
    if api_key:
        logger.info(f"🔑 Using API key from config.json: {api_key[:4]}...") # Log first few chars
        return api_key

    # 3. Fallback to environment variable (less preferred)
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        logger.warning("⚠️ Using GOOGLE_API_KEY from environment variable. For better security and management, consider adding it to `config.json` instead.")
        return api_key

    # 4. Fallback to old GEMINI_API_KEY environment variable (deprecated)
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        logger.warning(f"⚠️ Using deprecated GEMINI_API_KEY from environment variable. Please rename it to `GOOGLE_API_KEY` and ideally move it to `config.json`.")
        return api_key

    logger.error("❌ API Key Missing: No `GOOGLE_API_KEY` found in `config.json` or environment variables. Please set it up to use the Gemini Node.")
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
    output_thoughts: bool = False
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
        error_msg = f"{ERROR_PREFIX} API Key Missing: Cannot generate content without an API key."
        logger.error(error_msg)
        return error_msg, error_msg, prompt_tokens, response_tokens, thoughts_tokens

    while current_retry <= max_retries:
        try:
            client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
            logger.debug("✅ Gemini Client initialized for content generation.")

            # The config parameter is passed directly to generate_content_stream
            # It combines generation_config, safety_settings, and thinking_config
            full_config = types.GenerateContentConfig()
            if generation_config:
                full_config.temperature = generation_config.temperature
                full_config.top_p = generation_config.top_p
                full_config.top_k = generation_config.top_k
                full_config.max_output_tokens = generation_config.max_output_tokens
            if safety_settings:
                full_config.safety_settings = safety_settings
            if thinking_config:
                full_config.thinking_config = thinking_config

            logger.info(f"🚀 Sending request to Gemini model '{model_name}' (Attempt {current_retry + 1}/{max_retries + 1})...")
            
            response_stream = client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=full_config, # Pass the combined config here
            )

            full_response_text_list = []
            current_response_tokens = 0
            final_response_object = None

            for chunk in response_stream:
                final_response_object = chunk
                
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    candidate_content = chunk.candidates[0].content
                    if candidate_content is None:
                        logger.warning(f"⚠️ Candidate content is empty. This might indicate a safety block or a non-textual response. Checking feedback...")
                        continue
                    
                    for part in candidate_content.parts:
                        if part.text:
                            # Use client.models.count_tokens
                            chunk_token_count = client.models.count_tokens(model=model_name, contents=[part.text]).total_tokens
                            
                            if generation_config and (current_response_tokens + chunk_token_count > generation_config.max_output_tokens):
                                logger.warning(f"⚠️ Max output tokens ({generation_config.max_output_tokens}) reached. Truncating response.")
                                break
                            else:
                                full_response_text_list.append(part.text)
                                current_response_tokens += chunk_token_count
                        elif part.thought:
                            logger.debug("🐛 Received internal thought from model (not added to output).")
                        else:
                            logger.warning(f"⚠️ Received an unexpected part type with no text or thought. This chunk will be skipped.")
                else:
                    logger.warning(f"⚠️ Response chunk has no candidates. This might be an empty or malformed response.")
                
                if generation_config and current_response_tokens >= generation_config.max_output_tokens:
                    break

            generated_text = "".join(full_response_text_list)

            logger.debug(f"Final response object for token extraction: {final_response_object}")
            if final_response_object and hasattr(final_response_object, 'usage_metadata'):
                usage_metadata = final_response_object.usage_metadata
                prompt_tokens = getattr(usage_metadata, 'prompt_token_count', 0)
                response_tokens = getattr(usage_metadata, 'response_token_count', 0)
                thoughts_tokens = getattr(usage_metadata, 'thoughts_token_count', 0)
                logger.info(f"📊 Token Usage: Prompt={prompt_tokens}, Response={response_tokens}, Thoughts={thoughts_tokens}")
            else:
                logger.warning("⚠️ Could not retrieve detailed token usage metadata. Estimating response tokens from generated text.")
                if generated_text:
                    # Use client.models.count_tokens
                    response_tokens = client.models.count_tokens(model=model_name, contents=[generated_text]).total_tokens
                logger.info(f"📊 Estimated Token Usage (no metadata): Response={response_tokens}")

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
                response_error_msg = f"{ERROR_PREFIX} 🛑 Content Blocked: The response was stopped by safety settings. This might be due to sensitive content in your prompt or the model's output. Ratings: [{ratings_str}]. Please adjust your prompt or safety settings."
                logger.error(response_error_msg)
                return response_error_msg, response_error_msg, prompt_tokens, current_response_tokens, thoughts_tokens
            elif finish_reason_str == 'RECITATION':
                response_error_msg = f"{ERROR_PREFIX} 🚫 Content Blocked: The response was stopped due to potential recitation of copyrighted or sensitive material. Please try a different prompt."
                logger.error(response_error_msg)
                return response_error_msg, response_error_msg, prompt_tokens, current_response_tokens, thoughts_tokens
            elif finish_reason_str == 'MAX_TOKENS':
                logger.warning("⚠️ Generation stopped by API: The model reached its `max_output_tokens` limit. Consider increasing the limit or shortening your prompt.")
            elif finish_reason_str not in ['STOP', 'UNSPECIFIED', 'FINISH_REASON_UNSPECIFIED', 'OTHER', 'UNKNOWN', None]:
                logger.warning(f"⚠️ Generation finished with an unusual reason: {finish_reason_str}. The output might be incomplete.")

            if generated_text:
                logger.log(SUCCESS_HIGHLIGHT, f"🎉✨ Text generation complete! (Tokens: {current_response_tokens}, Chars: {len(generated_text)}). Finish Reason: {finish_reason_str}")
                return generated_text, None, prompt_tokens, current_response_tokens, thoughts_tokens
            else:
                status_msg = f"⚠️ Response received, but no text was extracted. Finish Reason: {finish_reason_str}. Retrying..."
                logger.warning(status_msg)
                raise RuntimeError(status_msg)

        except google_exceptions.GoogleAPIError as e:
            error_msg = f"{ERROR_PREFIX} ❌ Google API Error ({getattr(e, 'code', 'N/A')}): {e.message}. This could be an invalid API key, rate limit, or service issue. Retrying..."
            logger.error(error_msg, exc_info=True)
            pass

        except Exception as e:
            error_details = str(e)
            if hasattr(e, 'message') and e.message: error_details = e.message
            elif hasattr(e, 'details') and callable(e.details) and e.details(): error_details = e.details()
            error_msg = f"{ERROR_PREFIX} 💥 Unexpected Error during Gemini Generation ({type(e).__name__}): {error_details}. Please check your inputs and network connection. Retrying..."
            logger.error(error_msg, exc_info=True)
            pass

        current_retry += 1
        if current_retry <= max_retries:
            logger.info(f"⏳ Waiting {retry_delay_seconds} seconds before retry {current_retry}/{max_retries}...")
            import time
            time.sleep(retry_delay_seconds)

    final_error_msg = f"{ERROR_PREFIX} 🔴 All {max_retries} retries failed. Please check your API key, network, and the ComfyUI console for detailed errors."
    logger.error(final_error_msg)
    return final_error_msg, final_error_msg, prompt_tokens, response_tokens, thoughts_tokens


# This function is now designed to be called at node initialization (INPUT_TYPES)
# It attempts to fetch models if an API key is available, otherwise uses a default list.
def get_available_models_robust(api_key: Optional[str]) -> List[str]:
    """
    Fetches available Gemini models supporting 'generateContent' using genai.Client.
    If API key is invalid or fetching fails, it falls back to DEFAULT_MODELS.
    """
    logger.info("🔄 Attempting to fetch available Gemini models...")
    
    if not api_key:
        logger.warning("⚠️ No API key provided. Using a default list of models. Please configure your API key for dynamic model fetching.")
        return DEFAULT_MODELS

    try:
        genai.configure(api_key=api_key, transport='rest', http_options={'api_version': 'v1alpha'})
        logger.debug("✅ Gemini API configured for model listing.")

        model_list: List[str] = [
            m.name for m in genai.list_models()
            if 'generateContent' in m.supported_actions
        ]
        
        if not model_list:
            logger.warning("⚠️ No models supporting 'generateContent' were found via the API. Falling back to default model list. This might indicate an API issue or restricted access.")
            return DEFAULT_MODELS
        
        model_list.sort(key=lambda x: ('latest' not in x, x))
        logger.info(f"✅ Successfully fetched {len(model_list)} models from Gemini API.")
        return model_list
    
    except google_exceptions.GoogleAPIError as e:
        error_msg = f"❌ API Error fetching Gemini models ({getattr(e, 'code', 'N/A')}): {e.message}. Please check your API key and network connection."
        logger.error(error_msg, exc_info=True)
        logger.warning("⚠️ Failed to fetch models from Gemini API due to an API error. Using default model list.")
        return DEFAULT_MODELS
    except Exception as e:
        logger.error(f"💥 An unexpected error occurred while fetching Gemini models: {e}. Please report this issue.", exc_info=True)
        logger.warning("⚠️ Failed to fetch models from Gemini API due to an unexpected error. Using default model list.")
        return DEFAULT_MODELS
