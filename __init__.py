from .dolphin_vision_node import DolphinVision
from .clip_token_counter import CLIPTokenCounter

NODE_CLASS_MAPPINGS = {
    "DolphinVision": DolphinVision,
    "CLIPTokenCounter": CLIPTokenCounter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DolphinVision": "DolphinVision",
    "CLIPTokenCounter": "CLIP Token Counter"
}
