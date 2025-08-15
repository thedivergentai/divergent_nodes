import sys
import re
import requests
import torch
import json
import logging
from typing import Optional, Dict, Any, Tuple, List

# Import ComfyUI types for better type hinting and autocomplete
from comfy.comfy_types import ComfyNodeABC, IO, InputTypeDict

# Use relative import from the parent directory's utils
try:
    # Assumes shared_utils is one level up from koboldcpp
    from ..shared_utils.image_conversion import tensor_to_pil, pil_to_base64
    from ..shared_utils.logging_utils import SUCCESS_HIGHLIGHT # Import custom log level
except ImportError:
    # Fallback for direct execution or different structure
    logging.warning("Could not perform relative import for shared_utils, attempting direct import.")
    from shared_utils.image_conversion import tensor_to_pil, pil_to_base64
    from shared_utils.logging_utils import SUCCESS_HIGHLIGHT # Import custom log level

# Setup logger for this module
logger = logging.getLogger(__name__)

# --- Basic API Connector Node ---

class KoboldCppApiNode(ComfyNodeABC): # Inherit from ComfyNodeABC
    """
    ComfyUI node to connect to an ALREADY RUNNING KoboldCpp instance via its API.

    This node sends generation requests (text and optional image) to a specified
    KoboldCpp API endpoint (e.g., http://127.0.0.1:5001). It does NOT launch
    or manage the KoboldCpp process itself, relying on the user to have it
    running independently.
    """
    # Define types using ComfyUI conventions
    RETURN_TYPES: Tuple[str] = (IO.STRING,)
    RETURN_NAMES: Tuple[str] = ("text",)
    FUNCTION: str = "execute"
    CATEGORY: str = "Divergent Nodes 👽/KoboldCpp" # Keep category consistent

    def __init__(self):
        """Initializes the API Connector node instance."""
        logger.debug("KoboldCppApiNode instance created.")
        # No instance-specific state needed

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict: # Use InputTypeDict for type hinting
        """
        Defines the input types for the ComfyUI node interface.

        Includes the API URL and standard generation parameters.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary defining required and optional inputs.
        """
        return {
            "required": {
                "api_url": (IO.STRING, {
                    "multiline": False,
                    "default": "http://127.0.0.1:5001", # Default to common local KoboldCpp port
                    "tooltip": "Base URL of the running KoboldCpp API (e.g., http://127.0.0.1:5001)."
                }),
                # --- Generation Args (Passed via API JSON) ---
                "prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "Describe the image.",
                    "tooltip": "The text prompt for generation."
                }),
                "max_length": (IO.INT, {"default": 512, "min": 1, "max": 16384, "step": 1, "tooltip": "Maximum number of tokens to generate."}),
                "temperature": (IO.FLOAT, {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Sampling temperature."}),
                "top_p": (IO.FLOAT, {"default": 0.92, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Nucleus sampling probability."}),
                "top_k": (IO.INT, {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "Top-K sampling (0 disables)."}),
                "rep_pen": (IO.FLOAT, {"default": 1.1, "min": 0.0, "max": 3.0, "step": 0.01, "tooltip": "Repetition penalty."}),
            },
            "optional": {
                 # --- Optional Generation Args (Passed via API JSON) ---
                "image_optional": (IO.IMAGE, {"tooltip": "Optional image input for multimodal models."}), # ComfyUI's IMAGE type
                "stop_sequence": (IO.STRING, {
                    "multiline": True,
                     "default": "", # Comma or newline separated
                     "tooltip": "Comma or newline-separated strings to stop generation at."
                }),
                # TODO: Add other relevant API params as optional inputs if desired (e.g., top_a, typical, mirostat)
            }
        }

    def _check_api_connection(self, api_url: str) -> bool:
        """Performs a quick check to see if the API endpoint is reachable."""
        version_url = f"{api_url}/api/extra/version"
        logger.debug(f"Checking API connection to: {version_url}")
        try:
            # Use a short timeout for the readiness check
            response = requests.get(version_url, timeout=3)
            response.raise_for_status() # Check for HTTP errors (4xx, 5xx)
            logger.info(f"Successfully connected to KoboldCpp API at {api_url}.")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to KoboldCpp API at {api_url}. Is it running? Details: {e}")
            return False
        except Exception as e:
             # Catch other potential errors during the check
             logger.error(f"Unexpected error checking API status at {api_url}: {e}", exc_info=True)
             return False

    def execute(self,
                api_url: str,
                prompt: str,
                max_length: int,
                temperature: float,
                top_p: float,
                top_k: int,
                rep_pen: float,
                image_optional: Optional[torch.Tensor] = None,
                stop_sequence: str = ""
                ) -> Tuple[str]:
        """
        Executes the KoboldCpp API Connector node logic.

        Validates the API URL, checks connection, prepares the payload (including
        optional image conversion), sends the generation request to the specified
        API endpoint, and returns the generated text or an error message.

        Args:
            api_url (str): The base URL of the running KoboldCpp API (e.g., "http://127.0.0.1:5001").
            prompt (str): The text prompt for generation.
            max_length (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling p value.
            top_k (int): Sampling top-k value (0 disables).
            rep_pen (float): Repetition penalty.
            image_optional (Optional[torch.Tensor]): Optional ComfyUI IMAGE tensor. Defaults to None.
            stop_sequence (str): Optional comma or newline-separated stop strings. Defaults to "".

        Returns:
            Tuple[str]: A tuple containing a single string: the generated text or an error message
                        prefixed with "ERROR:".
        """
        logger.info("KoboldCpp API Connector Node: Starting execution.")
        base64_image_string: Optional[str] = None
        generated_text: str = "ERROR: Node execution failed." # Default error message
        error_prefix = "ERROR:"

        # --- 1. Clean and Validate API URL ---
        api_url_cleaned: str = api_url.strip().rstrip('/')
        if not api_url_cleaned.startswith(("http://", "https://")):
            error_msg = f"{error_prefix} Invalid API URL format: '{api_url}'. Must start with http:// or https://."
            logger.error(error_msg)
            return (error_msg,)

        # --- 2. Check API Connection ---
        if not self._check_api_connection(api_url_cleaned):
            # Error logged within _check_api_connection
            return (f"{error_prefix} Failed to connect to KoboldCpp API at {api_url_cleaned}. Please ensure it is running and the URL is correct.",)

        # --- 3. Handle Image Input (Convert to Base64) ---
        if image_optional is not None:
             logger.debug("Processing optional image input.")
             try:
                 pil_image = tensor_to_pil(image_optional) # Use shared util
                 if pil_image:
                     base64_image_string = pil_to_base64(pil_image, format="jpeg") # Use shared util
                     if not base64_image_string:
                          logger.error("Failed to convert input image tensor to Base64 string.")
                          return (f"{error_prefix} Failed to convert input image to Base64.",)
                     logger.debug("Successfully converted image to Base64.")
                 else:
                     logger.warning("Could not convert input tensor to PIL image. Check tensor format/shape.")
             except Exception as e:
                  logger.error(f"Error during image processing: {e}", exc_info=True)
                  return (f"{error_prefix} Failed during image processing: {e}",)

        # --- 4. Prepare Stop Sequences (Split string into list) ---
        stop_sequence_list: Optional[List[str]] = None
        if stop_sequence and stop_sequence.strip():
             # Split by comma or newline, remove empty strings/whitespace
             stop_sequence_list = [seq.strip() for seq in re.split(r'[,\n]', stop_sequence) if seq.strip()]
             if not stop_sequence_list:
                 stop_sequence_list = None # Ensure it's None if list becomes empty
             logger.debug(f"Parsed stop sequences: {stop_sequence_list}")
        else:
             logger.debug("No stop sequences provided.")

        # --- 5. Prepare API Request Payload ---
        # Construct the prompt, potentially adding multimodal prefix
        final_prompt: str
        if base64_image_string:
            # Common format for multimodal prompts in KoboldCpp API
            # Adjust this format if the target KoboldCpp version uses a different convention
            final_prompt = f"\n(Attached Image)\n\n### Instruction:\n{prompt}\n### Response:\n"
            logger.debug("Using multimodal prompt format.")
        else:
            final_prompt = prompt
            logger.debug("Using standard text prompt format.")

        payload: Dict[str, Any] = {
            "prompt": final_prompt,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "rep_pen": rep_pen,
            "use_default_badwordsids": False, # Common default for API usage
            "n": 1, # Generate one candidate
            # Add other generation parameters here if needed (e.g., top_a, mirostat)
        }
        if base64_image_string:
            payload["images"] = [base64_image_string] # API expects a list
        if stop_sequence_list:
            payload["stop_sequence"] = stop_sequence_list

        # --- 6. Call KoboldCpp API ---
        generate_url: str = f"{api_url_cleaned}/api/v1/generate" # Standard endpoint
        logger.info(f"Sending generation request to {generate_url}")
        logger.debug(f"API Payload: {json.dumps(payload, indent=2)}")
        try:
            response = requests.post(generate_url, json=payload, timeout=300) # 5 min timeout
            response.raise_for_status() # Check for 4xx/5xx errors

            response_json: Dict[str, Any] = response.json()
            logger.debug(f"API Response JSON: {json.dumps(response_json, indent=2)}")

            results: Optional[List[Dict[str, Any]]] = response_json.get("results")
            if results and isinstance(results, list) and len(results) > 0:
                # Extract text from the first result
                generated_text = results[0].get("text")
                if generated_text is not None:
                     logger.log(SUCCESS_HIGHLIGHT, "API call successful, received generated text.") # Use SUCCESS_HIGHLIGHT
                     generated_text = str(generated_text) # Ensure string type
                else:
                     generated_text = f"{error_prefix} 'text' field not found in API response results."
                     logger.warning(f"Invalid API response structure ('text' missing): {response_json}")
            else:
                # Handle cases where 'results' is missing or empty
                generated_text = f"{error_prefix} 'results' field not found or invalid in API response: {response_json}"
                logger.warning(f"Invalid API response structure ('results' missing or empty): {response_json}")

        except requests.exceptions.Timeout:
            generated_text = f"{error_prefix} API request timed out after 300 seconds to {generate_url}."
            logger.error(generated_text)
        except requests.exceptions.RequestException as e:
            # Handles ConnectionError, HTTPError, etc.
            generated_text = f"{error_prefix} API request failed: {e}"
            logger.error(generated_text, exc_info=True)
        except json.JSONDecodeError as e:
             generated_text = f"{error_prefix} Failed to parse JSON response from API: {e}"
             logger.error(generated_text, exc_info=True)
             logger.debug(f"Raw API response content: {response.text if 'response' in locals() else 'N/A'}")
        except Exception as e:
             # Catch any other unexpected errors
             generated_text = f"{error_prefix} An unexpected error occurred during API call: {e}"
             logger.error(generated_text, exc_info=True)

        # --- 7. Return the result ---
        logger.info("KoboldCpp API Connector Node: Execution finished.")
        return (generated_text,)

# Note: NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS are defined
# in the koboldcpp/__init__.py file.
