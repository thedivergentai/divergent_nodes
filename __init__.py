# Import mappings from each node package
# Using try-except blocks for robustness in case a package fails to load
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .clip_utils import NODE_CLASS_MAPPINGS as clip_utils_class_mappings, NODE_DISPLAY_NAME_MAPPINGS as clip_utils_display_mappings
    NODE_CLASS_MAPPINGS.update(clip_utils_class_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(clip_utils_display_mappings)
except ImportError as e:
    print(f"[WARN] Failed to import clip_utils nodes: {e}")

try:
    from .google_ai import NODE_CLASS_MAPPINGS as google_ai_class_mappings, NODE_DISPLAY_NAME_MAPPINGS as google_ai_display_mappings
    NODE_CLASS_MAPPINGS.update(google_ai_class_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(google_ai_display_mappings)
except ImportError as e:
    print(f"[WARN] Failed to import google_ai nodes: {e}")

try:
    from .xy_plotting import NODE_CLASS_MAPPINGS as xy_plotting_class_mappings, NODE_DISPLAY_NAME_MAPPINGS as xy_plotting_display_mappings
    NODE_CLASS_MAPPINGS.update(xy_plotting_class_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(xy_plotting_display_mappings)
except ImportError as e:
    print(f"[WARN] Failed to import xy_plotting nodes: {e}")

try:
    from .koboldcpp import NODE_CLASS_MAPPINGS as kobold_class_mappings, NODE_DISPLAY_NAME_MAPPINGS as kobold_display_mappings
    NODE_CLASS_MAPPINGS.update(kobold_class_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(kobold_display_mappings)
except ImportError as e:
    print(f"[WARN] Failed to import koboldcpp nodes: {e}")

try:
    from .image_utils import NODE_CLASS_MAPPINGS as image_utils_class_mappings, NODE_DISPLAY_NAME_MAPPINGS as image_utils_display_mappings
    NODE_CLASS_MAPPINGS.update(image_utils_class_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(image_utils_display_mappings)
except ImportError as e:
    print(f"[WARN] Failed to import image_utils nodes: {e}")


# Expose the aggregated mappings for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
