# Expose node classes from this module
from .gemini_api_node import GeminiNode
# Removed logging setup from __init__.py to prevent potential import issues during ComfyUI startup.
# Logging is handled within individual node files or via ComfyUI's default logging.

# Define mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeminiNode": GeminiNode,
}

# Optional: Define display names if different from class names
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiNode": "Gemini API Node",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
