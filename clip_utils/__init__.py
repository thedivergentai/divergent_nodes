# Expose node classes from this module
from .token_counter_node import CLIPTokenCounter
from ..shared_utils.logging_utils import setup_node_logging

# Setup logging for this node package
setup_node_logging()

# Define mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "CLIPTokenCounter": CLIPTokenCounter,
}

# Optional: Define display names if different from class names
NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTokenCounter": "CLIP Token Counter",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
