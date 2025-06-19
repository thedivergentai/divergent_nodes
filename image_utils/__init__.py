# Node mappings for the image_utils package

from .save_image_enhanced_node import SaveImageEnhancedNode
from ..shared_utils.logging_utils import setup_node_logging

# Setup logging for this node package
setup_node_logging()

NODE_CLASS_MAPPINGS = {
    "SaveImageEnhancedNode": SaveImageEnhancedNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageEnhancedNode": "Save Image Enhanced (DN)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
