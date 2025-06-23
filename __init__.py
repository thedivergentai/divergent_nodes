# Import mappings from each node package
# Using direct import and dictionary unpacking for robustness

# Import all necessary mappings from sub-packages
from .clip_utils import NODE_CLASS_MAPPINGS as clip_utils_class_mappings, NODE_DISPLAY_NAME_MAPPINGS as clip_utils_display_mappings
from .google_ai import NODE_CLASS_MAPPINGS as google_ai_class_mappings, NODE_DISPLAY_NAME_MAPPINGS as google_ai_display_mappings
from .xy_plotting import NODE_CLASS_MAPPINGS as xy_plotting_class_mappings, NODE_DISPLAY_NAME_MAPPINGS as xy_plotting_display_mappings
from .koboldcpp import NODE_CLASS_MAPPINGS as kobold_class_mappings, NODE_DISPLAY_NAME_MAPPINGS as kobold_display_mappings
from .image_utils import NODE_CLASS_MAPPINGS as image_utils_class_mappings, NODE_DISPLAY_NAME_MAPPINGS as image_utils_display_mappings
from .musiq_utils import NODE_CLASS_MAPPINGS as musiq_utils_class_mappings, NODE_DISPLAY_NAME_MAPPINGS as musiq_utils_display_mappings

# Combine all node mappings into a single dictionary
NODE_CLASS_MAPPINGS = {
    **clip_utils_class_mappings,
    **google_ai_class_mappings,
    **xy_plotting_class_mappings,
    **kobold_class_mappings,
    **image_utils_class_mappings,
    **musiq_utils_class_mappings,
}

# Combine all display name mappings into a single dictionary
NODE_DISPLAY_NAME_MAPPINGS = {
    **clip_utils_display_mappings,
    **google_ai_display_mappings,
    **xy_plotting_display_mappings,
    **kobold_display_mappings,
    **image_utils_display_mappings,
    **musiq_utils_display_mappings,
}

# Expose the aggregated mappings for ComfyUI
WEB_DIRECTORY = "./js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
