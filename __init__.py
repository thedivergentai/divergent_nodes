from .clip_token_counter import CLIPTokenCounter
from .dolphin_vision_node import DolphinVisionNode

NODE_CLASS_MAPPINGS = {
    "CLIPTokenCounter": CLIPTokenCounter,
    "DolphinVisionNode": DolphinVisionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTokenCounter": "CLIP Token Counter",
    "DolphinVisionNode": "Dolphin Vision"
}
