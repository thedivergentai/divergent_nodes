import torch
from transformers import CLIPTokenizer


class CLIPTokenCounter:
    """
    A ComfyUI node for counting CLIP tokens in a given text string.
    """
    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types for the CLIPTokenCounter node.

        Returns:
            dict: A dictionary specifying the input types, including:
              - text (STRING): The text string to analyze.
              - tokenizer_name (STRING): The name of the CLIP tokenizer to use.
        """
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "tokenizer_name": (["openai/clip-vit-base-patch32", "stabilityai/stable-diffusion-clip-vit-large-patch14"], {"default": "openai/clip-vit-base-patch32"}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("token_count",)
    FUNCTION = "count_tokens"
    CATEGORY = "Divergent Nodes 👽/Text"

    def __init__(self, tokenizer_name="openai/clip-vit-base-patch32"):
        """
        Initializes the CLIPTokenCounter.

        Args:
            tokenizer_name (str): The name of the CLIP tokenizer to use
                (e.g., "openai/clip-vit-base-patch32").
        """
        self.tokenizer_name = tokenizer_name
        self.tokenizer = None


    def count_tokens(self, text, tokenizer_name):
        """
        Counts the number of CLIP tokens in the given text.

        Args:
            text (str): The text to tokenize.
            tokenizer_name (str): The name of the tokenizer to use (redundant, kept for compatibility).

        Returns:
            tuple: A tuple containing the token count (int). Returns (0,) if the input
            text is empty or not a string, or if an error occurs during tokenization.
        """
        # Initialize tokenizer if it hasn't been initialized yet
        if self.tokenizer is None:
            try:
                self.tokenizer = CLIPTokenizer.from_pretrained(self.tokenizer_name)
            except Exception as e:
                print(f"Error loading tokenizer: {e}")
                return (0,)

        try:
            if not text or not isinstance(text, str):
                return (0,)

            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

            token_count = inputs['input_ids'].shape[1]
            return (token_count,)

        except Exception as e:
            print(f"Error tokenizing text: {e}")
            return (0,)
