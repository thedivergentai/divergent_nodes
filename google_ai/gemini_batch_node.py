"""
GeminiBatchProcess (v3): Batch process multiple images/prompts with Gemini API.
Migrated to ComfyUI v3 schema with INPUT_IS_LIST support.
"""
import logging
import torch
import io as io_module
from typing import Optional, List, Tuple
from comfy_api.latest import io

# Import Gemini utilities
from .gemini_utils import (
    configure_api_key,
    prepare_safety_settings,
    prepare_generation_config,
    prepare_content_parts,
    generate_content,
    SAFETY_SETTINGS_MAP,
    SAFETY_THRESHOLD_TO_NAME,
    ERROR_PREFIX
)

# Import shared utilities
from ..shared_utils.text_encoding_utils import ensure_utf8_friendly
from ..shared_utils.image_conversion import tensor_to_pil
from ..shared_utils.batch_utils import pad_list_to_length

# Import genai types
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiBatchProcess(io.ComfyNode):
    """
    Process multiple images/prompts in batch with Google Gemini API.
    Uses list inputs for efficient batch processing.
    """
    
    DEFAULT_MODELS = [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        default_safety = SAFETY_THRESHOLD_TO_NAME.get("BLOCK_MEDIUM_AND_ABOVE", "Block Medium and Above")
        safety_options = list(SAFETY_SETTINGS_MAP.keys())
        
        return io.Schema(
            node_id="GeminiBatchProcess",
            display_name="Gemini Batch Process",
            category="Divergent Nodes 👽/Gemini",
            description="Process multiple images/prompts with Gemini API in batch mode.",
            inputs=[
                io.Combo.Input("model", options=cls.DEFAULT_MODELS, default="gemini-1.5-flash"),
                io.String.Input("prompts", tooltip="List of prompts (one per image)"),
                io.Float.Input("temperature", default=0.9, min=0.0, max=2.0),
                io.Int.Input("max_output_tokens", default=2048, min=1, max=8192),
                io.Combo.Input("safety_harassment", options=safety_options, default=default_safety),
                io.Combo.Input("safety_hate_speech", options=safety_options, default=default_safety),
                io.Combo.Input("safety_sexually_explicit", options=safety_options, default=default_safety),
                io.Combo.Input("safety_dangerous_content", options=safety_options, default=default_safety),
                io.Image.Input("images", optional=True, tooltip="Optional list of images"),
                io.String.Input("api_key_override", default="", optional=True),
                io.Int.Input("max_retries", default=3, min=0, max=10, optional=True),
            ],
            outputs=[
                io.String.Output(display_name="texts"),
                io.String.Output(display_name="errors"),
            ],
        )
    
    @classmethod
    def execute(
        cls,
        model: str,
        prompts: List[str],
        temperature: float,
        max_output_tokens: int,
        safety_harassment: str,
        safety_hate_speech: str,
        safety_sexually_explicit: str,
        safety_dangerous_content: str,
        images: Optional[List[torch.Tensor]] = None,
        api_key_override: str = "",
        max_retries: int = 3,
    ) -> io.NodeOutput:
        """Process batch of prompts and images."""
        logger.info(f"🚀 [GeminiBatch] Processing {len(prompts)} prompts in batch mode")
        
        # Get API key
        api_key = api_key_override.strip() if api_key_override.strip() else configure_api_key()
        if not api_key:
            error = f"{ERROR_PREFIX} API key not found"
            logger.error(error)
            return io.NodeOutput([error] * len(prompts), [error] * len(prompts))
        
        # Pad images list
        if images and len(images) < len(prompts):
            logger.warning(f"⚠️ Padding {len(images)} images to {len(prompts)}")
            images = pad_list_to_length(images, len(prompts))
        elif images and len(images) > len(prompts):
            images = images[:len(prompts)]
        
        # Prepare configs
        safety_settings = prepare_safety_settings(
            safety_harassment, safety_hate_speech,
            safety_sexually_explicit, safety_dangerous_content
        )
        generation_config = prepare_generation_config(
            temperature, 1.0, 1, max_output_tokens, None
        )
        
        # Process each prompt
        results = []
        errors = []
        
        for i, prompt in enumerate(prompts):
            safe_prompt = ensure_utf8_friendly(prompt)
            content_parts, error_msg = prepare_content_parts(safe_prompt)
            
            if error_msg:
                results.append("")
                errors.append(error_msg)
                continue
            
           # Add image if available
            if images and i < len(images):
                try:
                    pil_image = tensor_to_pil(images[i])
                    if pil_image:
                        img_byte_arr = io_module.BytesIO()
                        pil_image.save(img_byte_arr, format='JPEG', quality=90)
                        img_bytes = img_byte_arr.getvalue()
                        image_part = types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg')
                        content_parts.append(image_part)
                except Exception as e:
                    logger.error(f"Error processing image {i}: {e}")
            
            # Generate
            generated_text, response_error, _, _, _ = generate_content(
                api_key=api_key,
                model_name=model,
                contents=content_parts,
                generation_config=generation_config,
                safety_settings=safety_settings,
                max_retries=max_retries,
                retry_delay_seconds=5,
            )
            
            results.append(generated_text if not response_error else "")
            errors.append(response_error if response_error else "")
        
        logger.info(f"✅ [GeminiBatch] Completed {len(results)} generations")
        return io.NodeOutput(results, errors)
