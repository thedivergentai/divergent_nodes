from .clip_token_counter import CLIPTokenCounter
from .gemini_node import GeminiNode
# from .gemma3_vision_node import Gemma3VisionNode # Keep or remove? Assuming keep for now
from .gemma3_vision_node import Gemma3VisionNode
from .koboldcpp_node import KoboldCppLauncherNode, KoboldCppApiNode # Import both new nodes

NODE_CLASS_MAPPINGS = {
    "CLIPTokenCounter": CLIPTokenCounter,
    "GeminiNode": GeminiNode,
    "Gemma3VisionNode": Gemma3VisionNode, # Keep or remove?
    "KoboldCppLauncherNode": KoboldCppLauncherNode, # Register Launcher node
    "KoboldCppApiNode": KoboldCppApiNode, # Register API node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTokenCounter": "CLIP Token Counter",
    "GeminiNode": "Gemini API Node",
    "Gemma3VisionNode": "Gemma3 Vision Node", # Keep or remove?
    "KoboldCppLauncherNode": "KoboldCpp Launcher (Advanced)", # Display name for Launcher
    "KoboldCppApiNode": "KoboldCpp API Connector (Basic)", # Display name for API node
}
