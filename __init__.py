from .clip_token_counter import CLIPTokenCounter
from .gemini_node import GeminiNode
from .gemma3_vision_node import Gemma3VisionNode
from .koboldcpp_node import KoboldCppNode # Added import

NODE_CLASS_MAPPINGS = {
    "CLIPTokenCounter": CLIPTokenCounter,
    "GeminiNode": GeminiNode,
    "Gemma3VisionNode": Gemma3VisionNode,
    "KoboldCppNode": KoboldCppNode, # Added mapping
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTokenCounter": "CLIP Token Counter",
    "GeminiNode": "Gemini API Node",
    "Gemma3VisionNode": "Gemma3 Vision Node",
    "KoboldCppNode": "KoboldCpp Node", # Added display name
}
