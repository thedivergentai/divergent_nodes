"""
ComfyUI v3 Extension Entry Point for Divergent Nodes.
Registers all v3-compatible nodes with ComfyUI.
"""
from comfy_api.latest import ComfyExtension, io
from typing import List

# Import all v3 nodes
from .text_utils.preview_text_node import PreviewText
from .text_utils.prompt_combiner_node import PromptCombiner
from .text_utils.string_list_node import StringToList
from .image_utils.image_info_node import ImageInfo
from .image_utils.save_image_enhanced_node import SaveImageEnhancedNode
from .google_ai.gemini_api_node import GeminiNode
from .google_ai.gemini_batch_node import GeminiBatchProcess
from .clip_utils.token_counter_node import CLIPTokenCounter
from .musiq_utils.musiq_node import MusiQNode
from .xy_plotting.lora_strength_plot_node import LoraStrengthXYPlot


class DivergentNodesExtension(ComfyExtension):
    """
    Divergent Nodes custom node pack for ComfyUI v3.
    Provides AI integration (Gemini), image utilities, and workflow helpers.
    """
    
    async def get_node_list(self) -> List[type[io.ComfyNode]]:
        """
        Returns list of all v3 node classes in this extension.
        """
        return [
            # Text utilities
            PreviewText,
            PromptCombiner,
            StringToList,
            
            # Image utilities
            ImageInfo,
            SaveImageEnhancedNode,
            
            # Google Gemini
            GeminiNode,
            GeminiBatchProcess,
            
            # Other utilities
            CLIPTokenCounter,
            MusiQNode,
            LoraStrengthXYPlot,
        ]


async def comfy_entrypoint() -> DivergentNodesExtension:
    """
    ComfyUI v3 entry point.
    This function is called by ComfyUI to initialize the extension.
    """
    return DivergentNodesExtension()
