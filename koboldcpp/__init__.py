# Expose node classes from this module
from .api_connector_node import KoboldCppApiNode

# Define mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "KoboldCppApiNode": KoboldCppApiNode,
}

# Optional: Define display names if different from class names
NODE_DISPLAY_NAME_MAPPINGS = {
    "KoboldCppApiNode": "KoboldCpp API Connector (Basic)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
