import torch
from transformers import CLIPTokenizer
from comfy.sd import CLIP


class DivergentCLIPTokenCounter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("token_count",)
    FUNCTION = "count_tokens"
    CATEGORY = "Divergent Nodes 👽/Text"

    def __init__(self):
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def count_tokens(self, text):
        try:
            if not text or not isinstance(text, str):
                return (0,)
                
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            token_count = inputs['input_ids'].shape[1]
            return (token_count,)
            
        except Exception as e:
            print(f"Error in token counting: {e}")
            return (0,)
