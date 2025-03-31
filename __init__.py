from .clip_token_counter import CLIPTokenCounter
from .gemini_node import GeminiNode

NODE_CLASS_MAPPINGS = {
    "CLIPTokenCounter": CLIPTokenCounter,
    "GeminiNode": GeminiNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTokenCounter": "CLIP Token Counter",
    "GeminiNode": "Gemini API Node",
}
