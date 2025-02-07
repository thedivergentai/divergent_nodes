from .clip_token_counter import DivergentCLIPTokenCounter
from .deepseek_vl2_node import DeepSeekVL2Node

NODE_CLASS_MAPPINGS = {
    "CLIP Token Counter": DivergentCLIPTokenCounter,
    "DeepSeekVL2Node": DeepSeekVL2Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIP Token Counter": "Divergent CLIP Token Counter",
    "DeepSeekVL2Node": "DeepSeek VL2 Node",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
