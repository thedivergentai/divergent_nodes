# Centralized Node Registration for Divergent Nodes

# Import all node classes
from .clip_utils.token_counter_node import CLIPTokenCounter
from .google_ai.gemini_api_node import GeminiNode
from .xy_plotting.lora_strength_plot_node import LoraStrengthXYPlot
from .koboldcpp.api_connector_node import KoboldCppApiNode
from .image_utils.save_image_enhanced_node import SaveImageEnhancedNode
from .musiq_utils.musiq_node import MusiQNode

# Define the master NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "CLIPTokenCounter": CLIPTokenCounter,
    "DivergentGeminiNode": GeminiNode,
    "LoraStrengthXYPlot": LoraStrengthXYPlot,
    "KoboldCppApiNode": KoboldCppApiNode,
    "SaveImageEnhancedNode": SaveImageEnhancedNode,
    "MusiQNode": MusiQNode,
}

# Define the master NODE_DISPLAY_NAME_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTokenCounter": "CLIP Token Counter",
    "DivergentGeminiNode": "Divergent Gemini Node",
    "LoraStrengthXYPlot": "XY Plot: LoRA/Strength",
    "KoboldCppApiNode": "KoboldCpp API Connector (Basic)",
    "SaveImageEnhancedNode": "Save Image Enhanced (DN)",
    "MusiQNode": "MusiQ Image Scorer",
}

# Optional: Define a common category for all nodes if desired, or keep individual categories in node classes
CATEGORY = "Divergent AI 👽"
