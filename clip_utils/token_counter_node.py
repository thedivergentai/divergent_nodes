import torch
from transformers import CLIPTokenizer, PreTrainedTokenizerBase
from typing import Tuple, Dict, Any, List, Optional
import logging

# Setup logger for this module
logger = logging.getLogger(__name__)


class CLIPTokenCounter:
    """
    A ComfyUI node for counting CLIP tokens in a given text string using
    a specified Hugging Face tokenizer.
    """
    # Define available tokenizers - consider making this dynamic or configurable if needed
    TOKENIZER_NAMES: List[str] = [
        "openai/clip-vit-base-patch32",
        "stabilityai/stable-diffusion-clip-vit-large-patch14",
        # Add other relevant CLIP tokenizers here
    ]

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """
        Defines the input types for the CLIPTokenCounter node.

        Returns:
            A dictionary specifying the required input types:
              - text (STRING): The text string to analyze.
              - tokenizer_name (COMBO): The name of the CLIP tokenizer to use.
        """
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "tooltip": "The text string to count tokens for."}),
                "tokenizer_name": (cls.TOKENIZER_NAMES, {"default": cls.TOKENIZER_NAMES[0], "tooltip": "Select the Hugging Face CLIP tokenizer to use."}),
            },
        }

    RETURN_TYPES: Tuple[str] = ("INT",)
    RETURN_NAMES: Tuple[str] = ("token_count",)
    FUNCTION: str = "count_tokens"
    CATEGORY: str = "Divergent Nodes 👽/Text Utils" # Updated category slightly

    # Removed __init__ as tokenizer loading is now handled per execution

    def count_tokens(self, text: str, tokenizer_name: str) -> Tuple[int]:
        """
        Counts the number of CLIP tokens in the given text using the specified tokenizer.

        Loads the tokenizer on demand. Handles empty input and tokenization errors.

        Args:
            text: The text string to tokenize.
            tokenizer_name: The Hugging Face name of the tokenizer to use
                            (e.g., "openai/clip-vit-base-patch32").

        Returns:
            A tuple containing the integer token count. Returns (0,) if the input
            text is empty, not a string, or if an error occurs during tokenization.
        """
        if not text or not isinstance(text, str):
            logger.warning("⚠️ [TokenCounter] Input text is empty or not a string. Cannot count tokens. Returning 0.")
            return (0,)

        try:
            # Load tokenizer - consider caching if performance becomes an issue
            # for frequently reused tokenizers within a single workflow execution.
            # For now, load fresh each time for simplicity.
            logger.debug(f"Loading tokenizer: {tokenizer_name}")
            tokenizer: PreTrainedTokenizerBase = CLIPTokenizer.from_pretrained(tokenizer_name)

            # Tokenize the input text
            # padding=True ensures consistent shape if needed elsewhere,
            # truncation=True prevents errors with overly long text.
            # return_tensors="pt" returns PyTorch tensors.
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

            # The token count is the sequence length (dimension 1) of the input_ids tensor
            # It includes special tokens (like BOS/EOS) added by the tokenizer.
            token_count: int = inputs['input_ids'].shape[1]
            logger.info(f"✅ [TokenCounter] Tokenized text with '{tokenizer_name}'. Token count: {token_count}")
            return (token_count,)

        except Exception as e:
            # Log the error and return 0
            logger.error(f"❌ [TokenCounter] Failed to load tokenizer or tokenize text with '{tokenizer_name}'. Please check the tokenizer name and your internet connection. Error: {e}", exc_info=True)
            return (0,)

# Note: Mappings are handled in clip_utils/__init__.py
