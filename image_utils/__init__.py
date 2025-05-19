# Node mappings for the image_utils package

from .save_image_enhanced_node import SaveImageEnhancedNode

NODE_CLASS_MAPPINGS = {
    "SaveImageEnhancedNode": SaveImageEnhancedNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageEnhancedNode": "Save Image Enhanced (DN)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
