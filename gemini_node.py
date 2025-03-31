import os
import torch
import numpy as np
from PIL import Image
import google.generativeai as genai
# Removed incorrect import: from google.generativeai.types import generation_types
from dotenv import load_dotenv, find_dotenv

# --- Constants ---
# More comprehensive default list based on documentation (Mar 2025)
# The dynamic fetch should override this if successful. Sorted newest -> oldest.
DEFAULT_MODELS = [
    # 2.5 Models
    "models/gemini-2.5-pro-exp-03-25",
    # 2.0 Models
    "models/gemini-2.0-flash", # latest
    "models/gemini-2.0-flash-lite", # latest
    "models/gemini-2.0-flash-001",
    "models/gemini-2.0-flash-lite-001",
    "models/gemini-2.0-flash-exp",
    "models/gemini-2.0-flash-thinking-exp-01-21",
    # 1.5 Models
    "models/gemini-1.5-pro-latest",
    "models/gemini-1.5-flash-latest",
    "models/gemini-1.5-flash-8b-latest",
    "models/gemini-1.5-pro", # latest stable
    "models/gemini-1.5-flash", # latest stable
    "models/gemini-1.5-flash-8b", # latest stable
    "models/gemini-1.5-pro-002",
    "models/gemini-1.5-flash-002",
    "models/gemini-1.5-pro-001",
    "models/gemini-1.5-flash-001",
    "models/gemini-1.5-flash-8b-001",
    # 1.0 Models
    "models/gemini-1.0-pro", # Older stable
    "models/gemini-pro-vision", # Older vision stable
]

SAFETY_SETTINGS_MAP = {
    "Default (Unspecified)": "HARM_BLOCK_THRESHOLD_UNSPECIFIED",
    "Block Low & Above": "BLOCK_LOW_AND_ABOVE",
    "Block Medium & Above": "BLOCK_MEDIUM_AND_ABOVE",
    "Block High Only": "BLOCK_ONLY_HIGH",
    "Block None": "BLOCK_NONE",
}
# Reverse map for finding default friendly name
SAFETY_THRESHOLD_TO_NAME = {v: k for k, v in SAFETY_SETTINGS_MAP.items()}

# Removed incorrect BLOCK_REASON_MAP


# --- Helper Functions ---

def tensor_to_pil(tensor):
    """Converts a torch tensor (B, H, W, C) to a list of PIL Images."""
    if tensor is None:
        return None # Return None if no tensor provided
    images = []
    for i in range(tensor.shape[0]):
        img_np = tensor[i].cpu().numpy()
        # Ensure conversion from [0,1] float to [0,255] uint8
        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
             img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        elif img_np.dtype != np.uint8: # Handle other potential types if necessary
             print(f"[GeminiNode] Warning: Unexpected tensor dtype {img_np.dtype}, attempting conversion.")
             img_np = img_np.astype(np.uint8) # Direct cast might not be ideal for all types

        try:
            pil_image = Image.fromarray(img_np)
            images.append(pil_image)
        except Exception as e:
            print(f"[GeminiNode] Error converting tensor slice to PIL Image: {e}")
            return None # Return None if conversion fails for any image in batch

    # Return only the first image if it's a single-image batch for Gemini
    return images[0] if images else None

def get_available_models():
    """Fetches available Gemini models supporting generateContent."""
    try:
        # Load API key specifically for listing models
        load_dotenv(find_dotenv())
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[GeminiNode] Warning: GEMINI_API_KEY not found in .env. Cannot fetch dynamic model list.")
            return DEFAULT_MODELS

        genai.configure(api_key=api_key)
        model_list = []
        for m in genai.list_models():
            # Check if 'generateContent' is a supported method
            if 'generateContent' in m.supported_generation_methods:
                 # Use the full name like 'models/gemini-1.5-flash-latest'
                 model_list.append(m.name)
        if not model_list:
             print("[GeminiNode] Warning: No models supporting generateContent found via API. Using default list.")
             return DEFAULT_MODELS
        # Sort for consistency, maybe put 'latest' versions first?
        model_list.sort(key=lambda x: ('latest' not in x, x))
        print(f"[GeminiNode] Fetched {len(model_list)} models from Gemini API.")
        return model_list
    except Exception as e:
        print(f"[GeminiNode] Warning: Failed to fetch models from Gemini API: {e}. Using default list.")
        return DEFAULT_MODELS

# --- ComfyUI Node Definition ---

class GeminiNode:
    """
    A ComfyUI node to interact with the Google Gemini API.
    Fetches available models dynamically.
    """
    AVAILABLE_MODELS = get_available_models() # Fetch models when class is loaded
    SAFETY_OPTIONS = list(SAFETY_SETTINGS_MAP.keys())

    def __init__(self):
        # Load API key from .env file when an instance is created
        load_dotenv(find_dotenv())
        self.api_key = os.getenv("GEMINI_API_KEY")

    @classmethod
    def INPUT_TYPES(s):
        # Find default friendly names, fallback if exact match not found
        default_harassment = SAFETY_THRESHOLD_TO_NAME.get("BLOCK_MEDIUM_AND_ABOVE", s.SAFETY_OPTIONS[0])
        default_hate = SAFETY_THRESHOLD_TO_NAME.get("BLOCK_MEDIUM_AND_ABOVE", s.SAFETY_OPTIONS[0])
        default_sexual = SAFETY_THRESHOLD_TO_NAME.get("BLOCK_MEDIUM_AND_ABOVE", s.SAFETY_OPTIONS[0])
        default_dangerous = SAFETY_THRESHOLD_TO_NAME.get("BLOCK_MEDIUM_AND_ABOVE", s.SAFETY_OPTIONS[0])
        # Removed civic integrity default calculation

        return {
            "required": {
                # Use the class variable holding the fetched models
                "model": (s.AVAILABLE_MODELS, {"default": s.AVAILABLE_MODELS[0] if s.AVAILABLE_MODELS else ""}),
                "prompt": ("STRING", {"multiline": True, "default": "Describe the image."}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "max_output_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192, "step": 1}),
                # Use user-friendly names for safety settings
                "safety_harassment": (s.SAFETY_OPTIONS, {"default": default_harassment}),
                "safety_hate_speech": (s.SAFETY_OPTIONS, {"default": default_hate}),
                "safety_sexually_explicit": (s.SAFETY_OPTIONS, {"default": default_sexual}),
                "safety_dangerous_content": (s.SAFETY_OPTIONS, {"default": default_dangerous}),
                # Removed civic integrity input
            },
            "optional": {
                 "image_optional": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "Divergent Nodes 👽/Gemini"

    def generate(self, model, prompt, temperature, top_p, top_k, max_output_tokens,
                 safety_harassment, safety_hate_speech, safety_sexually_explicit,
                 safety_dangerous_content, image_optional=None): # Removed safety_civic_integrity

        if not self.api_key:
            # Re-check API key in case it wasn't available during model list fetch
            load_dotenv(find_dotenv())
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                error_msg = "ERROR: GEMINI_API_KEY not found in .env file."
                print(f"[GeminiNode] {error_msg}")
                return (error_msg,)

        try:
            print(f"[GeminiNode] Configuring Gemini with API key...")
            genai.configure(api_key=self.api_key)

            # Map user-friendly safety names back to API constants
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": SAFETY_SETTINGS_MAP[safety_harassment]},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": SAFETY_SETTINGS_MAP[safety_hate_speech]},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": SAFETY_SETTINGS_MAP[safety_sexually_explicit]},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": SAFETY_SETTINGS_MAP[safety_dangerous_content]},
                # Removed civic integrity setting
            ]

            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_output_tokens,
            )

            # Extract the actual model ID if the full name 'models/...' is passed
            model_id = model.split('/')[-1] if '/' in model else model
            print(f"[GeminiNode] Using model: {model_id}")

            gemini_model = genai.GenerativeModel(
                model_name=model_id,
                safety_settings=safety_settings,
                generation_config=generation_config
            )

            content_parts = [prompt]
            pil_image = tensor_to_pil(image_optional)

            if pil_image:
                 print("[GeminiNode] Image provided, adding to content parts.")
                 content_parts.append(pil_image)
            elif "vision" in model_id: # Keep the warning for vision models without images
                 print(f"[GeminiNode] Warning: Vision model '{model_id}' selected, but no image provided.")

            print(f"[GeminiNode] Sending request to Gemini API...")
            response = gemini_model.generate_content(content_parts)

            # Handle potential blocks or empty responses
            if not response.candidates:
                 block_reason_code = generation_types.BlockReason.BLOCK_REASON_UNSPECIFIED
                 finish_reason = "Unknown"
                 if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                      block_reason_code = response.prompt_feedback.block_reason
                 if hasattr(response, 'finish_reason'): # Sometimes finish_reason is on the response itself
                      finish_reason = response.finish_reason

                 # Get block reason string name directly from the enum object
                 block_reason_str = "UNSPECIFIED" # Default fallback
                 if block_reason_code and hasattr(block_reason_code, 'name'):
                      block_reason_str = block_reason_code.name
                 elif block_reason_code: # If it doesn't have .name, use its string representation
                      block_reason_str = str(block_reason_code)

                 error_msg = f"Blocked/Failed: Generation failed. Block Reason: {block_reason_str}, Finish Reason: {finish_reason}"
                 print(f"[GeminiNode] {error_msg}")
                 return (error_msg,)


            # Accessing generated text safely
            generated_text = ""
            try:
                 candidate = response.candidates[0]
                 # Check finish reason first - Use direct comparison if possible, or string name
                 finish_reason_val = candidate.finish_reason
                 if hasattr(finish_reason_val, 'name') and finish_reason_val.name == 'SAFETY':
                      safety_ratings_str = ', '.join([f"{r.category.name}: {r.probability.name}" for r in candidate.safety_ratings])
                      error_msg = f"Blocked: Response stopped due to safety settings. Ratings: [{safety_ratings_str}]"
                      print(f"[GeminiNode] {error_msg}")
                      return (error_msg,)
                 elif candidate.content and candidate.content.parts:
                      generated_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                      print(f"[GeminiNode] Successfully generated text (length: {len(generated_text)}).")
                 else:
                      # Handle cases like recitation, other finish reasons
                      finish_reason_str = finish_reason_val.name if hasattr(finish_reason_val, 'name') else str(finish_reason_val)
                      status_msg = f"Response received but no text content. Finish Reason: {finish_reason_str}"
                      # Check by name if possible
                      if hasattr(finish_reason_val, 'name') and finish_reason_val.name == 'RECITATION':
                           status_msg += ". (Content may have been blocked due to recitation)"
                      print(f"[GeminiNode] {status_msg}")
                      generated_text = status_msg # Return the status message as output


            except (ValueError, IndexError, AttributeError) as e:
                 # Catch potential errors accessing response structure
                 error_msg = f"Error accessing response content: {type(e).__name__}. Raw response: {response}"
                 print(f"[GeminiNode] {error_msg}")
                 generated_text = f"ERROR: {type(e).__name__} accessing response." # Simplified return


            return (generated_text,)

        except Exception as e:
            error_details = str(e)
            # Try to extract more specific Google API error messages
            if hasattr(e, 'message'):
                error_details = e.message
            elif hasattr(e, 'details'):
                 error_details = e.details()
            error_msg = f"ERROR: Gemini API Error - {error_details}"
            print(f"[GeminiNode] {error_msg}") # Log the detailed error
            return (f"ERROR: An API error occurred. Check console logs.",) # Simplified return

# Note: NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
# will be handled in __init__.py
