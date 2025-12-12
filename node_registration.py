# Centralized Node Registration for Divergent Nodes

# Import all node classes
from .clip_utils.token_counter_node import CLIPTokenCounter
from .google_ai.gemini_api_node import GeminiNode
from .google_ai.gemini_batch_node import GeminiBatchProcess
from .xy_plotting.lora_strength_plot_node import LoraStrengthXYPlot
from .image_utils.save_image_enhanced_node import SaveImageEnhancedNode
from .image_utils.image_info_node import ImageInfo
from .musiq_utils.musiq_node import MusiQNode
from .text_utils.prompt_combiner_node import PromptCombiner
from .text_utils.string_list_node import StringToList
from .text_utils.preview_text_node import PreviewText
# Note: Logging is configured in __init__.py

# Define the master NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "CLIPTokenCounter": CLIPTokenCounter,
    "DivergentGeminiNode": GeminiNode,
    "GeminiBatchProcess": GeminiBatchProcess,
    "LoraStrengthXYPlot": LoraStrengthXYPlot,
    "SaveImageEnhancedNode": SaveImageEnhancedNode,
    "ImageInfo": ImageInfo,
    "MusiQNode": MusiQNode,
    "PromptCombiner": PromptCombiner,
    "StringToList": StringToList,
    "PreviewText": PreviewText,
}

# Define the master NODE_DISPLAY_NAME_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTokenCounter": "CLIP Token Counter",
    "DivergentGeminiNode": "Divergent Gemini Node",
    "GeminiBatchProcess": "Gemini Batch Process",
    "LoraStrengthXYPlot": "XY Plot: LoRA/Strength",
    "SaveImageEnhancedNode": "Save Image Enhanced (DN)",
    "ImageInfo": "Image Info",
    "MusiQNode": "MusiQ Image Scorer",
    "PromptCombiner": "Prompt Combiner",
    "StringToList": "String to List",
    "PreviewText": "Preview Text",
}

# Optional: Define a common category for all nodes if desired, or keep individual categories in node classes
CATEGORY = "Divergent AI 👽"
