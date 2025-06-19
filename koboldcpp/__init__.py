# Expose node classes from this module
from .api_connector_node import KoboldCppApiNode
from ..shared_utils.logging_utils import setup_node_logging

# Setup logging for this node package
setup_node_logging()

# Define mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "KoboldCppApiNode": KoboldCppApiNode,
}

# Optional: Define display names if different from class names
NODE_DISPLAY_NAME_MAPPINGS = {
    "KoboldCppApiNode": "KoboldCpp API Connector (Basic)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
