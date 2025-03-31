import os
import torch
import numpy as np
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv

# --- Helper Functions ---

def tensor_to_pil(tensor):
    """Converts a torch tensor (B, H, W, C) to a list of PIL Images."""
    if tensor is None:
        return []
    images = []
    for i in range(tensor.shape[0]):
        img_np = tensor[i].cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8) # Convert from [0,1] float to [0,255] uint8
        pil_image = Image.fromarray(img_np)
        images.append(pil_image)
    # Return only the first image if it's a single-image batch for Gemini
    return images[0] if images else None

# --- ComfyUI Node Definition ---

class GeminiNode:
    """
    A ComfyUI node to interact with the Google Gemini API.
    """
    def __init__(self):
        # Load API key from .env file located in the same directory as this script
        load_dotenv(find_dotenv())
        self.api_key = os.getenv("GEMINI_API_KEY")

    @classmethod
    def INPUT_TYPES(s):
        safety_thresholds = [
            "HARM_BLOCK_THRESHOLD_UNSPECIFIED",
            "BLOCK_LOW_AND_ABOVE",
            "BLOCK_MEDIUM_AND_ABOVE",
            "BLOCK_ONLY_HIGH",
            "BLOCK_NONE",
        ]
        # List common models, user might need to adjust if new ones are released
        # Check https://ai.google.dev/models/gemini for available models
        models = [
            "gemini-1.5-flash-latest", # Alias for latest flash
            "gemini-1.5-pro-latest",   # Alias for latest pro
            "gemini-1.0-pro",          # Stable 1.0 pro
            # Add other models as needed, e.g., specific versions or vision models explicitly
            "gemini-pro-vision", # Required for image input
        ]
        return {
            "required": {
                "model": (models,),
                "prompt": ("STRING", {"multiline": True, "default": "Describe the image."}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}), # Often 1 for default Gemini Pro
                "max_output_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192, "step": 1}), # Check model limits
                "safety_harassment": (safety_thresholds, {"default": "BLOCK_MEDIUM_AND_ABOVE"}),
                "safety_hate_speech": (safety_thresholds, {"default": "BLOCK_MEDIUM_AND_ABOVE"}),
                "safety_sexually_explicit": (safety_thresholds, {"default": "BLOCK_MEDIUM_AND_ABOVE"}),
                "safety_dangerous_content": (safety_thresholds, {"default": "BLOCK_MEDIUM_AND_ABOVE"}),
                # Civic integrity default is often BLOCK_NONE or BLOCK_MOST depending on model/UI
                "safety_civic_integrity": (safety_thresholds, {"default": "HARM_BLOCK_THRESHOLD_UNSPECIFIED"}),
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
                 safety_dangerous_content, safety_civic_integrity, image_optional=None):

        if not self.api_key:
            return ("ERROR: GEMINI_API_KEY not found in .env file.",)

        try:
            genai.configure(api_key=self.api_key)

            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": safety_harassment},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": safety_hate_speech},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": safety_sexually_explicit},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": safety_dangerous_content},
                {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": safety_civic_integrity},
            ]

            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_output_tokens,
                # candidate_count=1 # Default is 1
            )

            gemini_model = genai.GenerativeModel(
                model_name=model,
                safety_settings=safety_settings,
                generation_config=generation_config
            )

            content_parts = [prompt]
            pil_image = tensor_to_pil(image_optional)

            if pil_image:
                 # Check if the selected model supports vision
                 if "vision" not in model and "1.5" not in model: # Basic check, might need refinement
                      return (f"ERROR: Model '{model}' might not support image input. Try a vision model like 'gemini-pro-vision' or a 1.5 model.",)
                 content_parts.append(pil_image)
            elif "vision" in model:
                 # Warn if a vision model is selected but no image provided
                 print(f"Warning: Vision model '{model}' selected, but no image provided.")


            response = gemini_model.generate_content(content_parts)

            # Handle potential blocks or empty responses
            if not response.candidates:
                 # Check prompt feedback for block reason
                 block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
                 return (f"Blocked: Generation failed due to safety settings or other reasons. Reason: {block_reason}",)

            # Accessing generated text safely
            generated_text = ""
            try:
                 # Check finish reason first
                 if response.candidates[0].finish_reason == 'SAFETY':
                      safety_ratings_str = ', '.join([f"{r.category.name}: {r.probability.name}" for r in response.candidates[0].safety_ratings])
                      return (f"Blocked: Response stopped due to safety settings. Ratings: [{safety_ratings_str}]",)
                 elif response.candidates[0].content and response.candidates[0].content.parts:
                      generated_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
                 else:
                      generated_text = f"Empty response received. Finish Reason: {response.candidates[0].finish_reason}"

            except ValueError as ve:
                 # Handle cases where accessing response parts might fail (e.g., unexpected structure)
                 generated_text = f"Error accessing response content: {ve}. Raw response: {response}"
            except IndexError:
                 generated_text = f"Error: No valid candidates found in response. Raw response: {response}"


            return (generated_text,)

        except Exception as e:
            print(f"Gemini API Error: {e}")
            # Attempt to get more specific error details if available
            error_details = str(e)
            if hasattr(e, 'message'): # Specific Google API errors might have this
                error_details = e.message
            return (f"ERROR: {error_details}",)

# Note: NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
# will be handled in __init__.py
