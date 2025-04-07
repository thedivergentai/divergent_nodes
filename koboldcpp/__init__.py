# Expose node classes from this module
from .launcher_node import KoboldCppLauncherNode
from .api_connector_node import KoboldCppApiNode

# Define mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "KoboldCppLauncherNode": KoboldCppLauncherNode,
    "KoboldCppApiNode": KoboldCppApiNode,
}

# Optional: Define display names if different from class names
NODE_DISPLAY_NAME_MAPPINGS = {
    "KoboldCppLauncherNode": "KoboldCpp Launcher (Advanced)",
    "KoboldCppApiNode": "KoboldCpp API Connector (Basic)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
