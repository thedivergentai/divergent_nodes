"""
GeminiNode (v3): ComfyUI node for Google Gemini API interaction.
Fully migrated to v3 schema with lazy evaluation, validation, and caching.
"""
import logging
import torch
import io as io_module
import hashlib
from typing import Optional, List, Any
from comfy_api.latest import io, ui

# Import Gemini utilities
from .gemini_utils import (
    configure_api_key,
    prepare_safety_settings,
    prepare_generation_config,
    prepare_content_parts,
    generate_content,
    prepare_thinking_config,
    SAFETY_SETTINGS_MAP,
    SAFETY_THRESHOLD_TO_NAME,
    ERROR_PREFIX
)

# Import shared utilities
from ..shared_utils.text_encoding_utils import ensure_utf8_friendly
from ..shared_utils.image_conversion import tensor_to_pil

# Import genai
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

logger = logging.getLogger(__name__)


class GeminiNode(io.ComfyNode):
    """
    Google Gemini API node for text generation with optional image input.
    Supports lazy evaluation, input validation, and intelligent caching.
    """
    
    AVAILABLE_MODELS = [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        # Get default safety setting
        default_safety = SAFETY_THRESHOLD_TO_NAME.get("BLOCK_MEDIUM_AND_ABOVE", "Block Medium and Above")
        safety_options = list(SAFETY_SETTINGS_MAP.keys())
       
        return io.Schema(
            node_id="DivergentGeminiNode",
            display_name="Divergent Gemini Node",
            category="Divergent Nodes 👽/Gemini",
            description="Generate text using Google Gemini API with optional image input for multimodal models.",
            inputs=[
                io.Combo.Input("model", options=cls.AVAILABLE_MODELS, default="gemini-1.5-flash", tooltip="Select the Gemini model to use"),
                io.String.Input("prompt", multiline=True, default="Describe the image.", tooltip="The text prompt for generation"),
                io.Float.Input("temperature", default=0.9, min=0.0, max=2.0, tooltip="Controls randomness. Higher = more creative"),
                io.Float.Input("top_p", default=1.0, min=0.0, max=1.0, tooltip="Nucleus sampling probability threshold"),
                io.Int.Input("top_k", default=1, min=1, max=100, tooltip="Top-K sampling threshold"),
                io.Int.Input("max_output_tokens", default=2048, min=1, max=8192, tooltip="Maximum tokens to generate"),
                io.Combo.Input("safety_harassment", options=safety_options, default=default_safety, tooltip="Safety threshold for harassment"),
                io.Combo.Input("safety_hate_speech", options=safety_options, default=default_safety, tooltip="Safety threshold for hate speech"),
                io.Combo.Input("safety_sexually_explicit", options=safety_options, default=default_safety, tooltip="Safety threshold for sexual content"),
                io.Combo.Input("safety_dangerous_content", options=safety_options, default=default_safety, tooltip="Safety threshold for dangerous content"),
                io.Image.Input("image_optional", optional=True, lazy=True, tooltip="Optional image for multimodal models"),
                io.String.Input("api_key_override", default="", optional=True, tooltip="Override API key for this node"),
                io.Int.Input("max_retries", default=3, min=0, max=10, optional=True, tooltip="Maximum retry attempts"),
                io.Int.Input("retry_delay_seconds", default=5, min=1, max=60, optional=True, tooltip="Delay between retries"),
                io.Boolean.Input("extended_thinking", default=True, optional=True, tooltip="Enable extended thinking if supported"),
                io.Int.Input("thinking_token_budget", default=-1, min=-1, max=8192, optional=True, tooltip="Token budget for thinking (-1=auto)"),
                io.Boolean.Input("output_thoughts", default=False, optional=True, tooltip="Include thought process in output"),
            ],
            outputs=[
                io.String.Output(display_name="text"),
            ],
        )
    
    @classmethod
    def validate_inputs(cls, api_key_override: str = "", model: str = "", **kwargs) -> bool | str:
        """Validate inputs before execution."""
        if api_key_override and not api_key_override.strip():
            return "API key override cannot be empty string"
        
        if model and model not in cls.AVAILABLE_MODELS:
            return f"Invalid model '{model}'. Must be one of: {', '.join(cls.AVAILABLE_MODELS[:3])}..."
        
        return True
    
    @classmethod
    def fingerprint_inputs(cls, prompt: str, model: str, temperature: float, image_optional=None, **kwargs) -> str:
        """Return fingerprint for caching. Only re-execute if inputs changed."""
        hash_input = f"{prompt}|{model}|{temperature}"
        
        if image_optional is not None and isinstance(image_optional, torch.Tensor):
            hash_input += f"|img_{image_optional.shape}_{image_optional.sum().item():.4f}"
        
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    @classmethod
    def check_lazy_status(cls, image_optional=None, **kwargs) -> List[str]:
        """Determine which lazy inputs to evaluate."""
        if image_optional is not None:
            return ["image_optional"]
        return []
    
    @classmethod
    def execute(
        cls,
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
        api_key_override: str = "",
        max_retries: int = 3,
        retry_delay_seconds: int = 5,
        extended_thinking: bool = True,
        thinking_token_budget: int = -1,
        output_thoughts: bool = False,
    ) -> io.NodeOutput:
        """Execute Gemini API generation."""
        logger.info(f"🚀 [GeminiNode] Starting generation with model: {model}")
        
        # Get API key
        api_key = api_key_override.strip() if api_key_override.strip() else configure_api_key()
        if not api_key:
            error =f"{ERROR_PREFIX} API key not found. Set GEMINI_API_KEY in environment or use api_key_override."
            logger.error(error)
            return io.NodeOutput(error)
        
        # Prepare content
        safe_prompt = ensure_utf8_friendly(prompt)
        content_parts, error_msg = prepare_content_parts(safe_prompt)
        
        if error_msg:
            logger.error(error_msg)
            return io.NodeOutput(error_msg)
        
        # Add image if provided
        if image_optional is not None:
            try:
                pil_image = tensor_to_pil(image_optional)
                if pil_image:
                    img_byte_arr = io_module.BytesIO()
                    pil_image.save(img_byte_arr, format='JPEG', quality=90)
                    img_bytes = img_byte_arr.getvalue()
                    image_part = types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg')
                    content_parts.append(image_part)
                    logger.debug("✅ [GeminiNode] Image added to content")
            except Exception as e:
                error = f"{ERROR_PREFIX} Error processing image: {e}"
                logger.error(error)
                return io.NodeOutput(error)
        
        # Prepare configs
        safety_settings = prepare_safety_settings(
            safety_harassment, safety_hate_speech,
            safety_sexually_explicit, safety_dangerous_content
        )
        generation_config = prepare_generation_config(
            temperature, top_p, top_k, max_output_tokens,
            prepare_thinking_config(extended_thinking, thinking_token_budget, output_thoughts)
        )
        
        # Generate content
        generated_text, response_error, _, _, _ = generate_content(
            api_key=api_key,
            model_name=model,
            contents=content_parts,
            generation_config=generation_config,
            safety_settings=safety_settings,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
        )
        
        if response_error:
            logger.error(response_error)
            return io.NodeOutput(response_error)
        
        logger.info(f"✅ [GeminiNode] Successfully generated {len(generated_text)} characters")
        return io.NodeOutput(generated_text)
