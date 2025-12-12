"""
CLIPTokenCounter (v3): Count CLIP tokens with caching for tokenizer optimization.
Migrated to ComfyUI v3 schema.
"""
import logging
from typing import List, Optional
from comfy_api.latest import io

logger = logging.getLogger(__name__)


class CLIPTokenCounter(io.ComfyNode):
    """
    Counts tokens in a text prompt using CLIP tokenizer.
    Useful for ensuring prompts fit within model token limits.
    """
    
    _tokenizer_cache = {}
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CLIPTokenCounter",
            display_name="CLIP Token Counter",
            category="Divergent Nodes 👽/Utils",
            description="Count tokens in a text prompt using CLIP tokenizer to ensure it fits within model limits.",
            inputs=[
                io.Clip.Input("clip", tooltip="CLIP model to use for tokenization"),
                io.String.Input("text", multiline=True, default="", tooltip="Text to count tokens for"),
                io.Boolean.Input("show_breakdown", default=False, optional=True, tooltip="Show per-word token breakdown"),
            ],
            outputs=[
                io.Int.Output(display_name="token_count"),
                io.String.Output(display_name="analysis"),
            ],
        )
    
    @classmethod
    def execute(cls, clip, text: str, show_breakdown: bool = False) -> io.NodeOutput:
        """Count tokens in text using CLIP tokenizer."""
        if not text:
            logger.warning("Empty text provided to CLIPTokenCounter")
            return io.NodeOutput(0, "Empty text - 0 tokens")
        
        try:
            # Get tokenizer from CLIP
            tokenizer = clip.tokenizer if hasattr(clip, 'tokenizer') else clip.cond_stage_model.tokenizer
            
            # Tokenize
            tokens = tokenizer.tokenize(text)
            token_count = len(tokens)
            
            # Build analysis
            analysis = f"Token count: {token_count}\n"
            analysis += f"Max tokens (CLIP): 77\n"
            analysis += f"Remaining: {77 - token_count}\n"
            
            if token_count > 77:
                analysis += f"⚠️ WARNING: Exceeds CLIP limit by {token_count - 77} tokens!\n"
            
            if show_breakdown:
                words = text.split()
                analysis += f"\nWord breakdown ({len(words)} words):\n"
                for word in words[:10]:  # Show first 10 words
                    word_tokens = tokenizer.tokenize(word)
                    analysis += f"  '{word}': {len(word_tokens)} tokens\n"
                if len(words) > 10:
                    analysis += f"  ... and {len(words) - 10} more words\n"
            
            logger.info(f"📊 [CLIPTokenCounter] Counted {token_count} tokens")
            return io.NodeOutput(token_count, analysis)
            
        except Exception as e:
            error = f"Error counting tokens: {e}"
            logger.error(error)
            return io.NodeOutput(0, error)
