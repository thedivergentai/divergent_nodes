"""
Divergent Nodes - Custom nodes for ComfyUI v3
Future-proof implementation using modern v3 schema.
"""

# Setup logging first
from .shared_utils.logging_utils import setup_node_logging
setup_node_logging()

# V3 Entry Point (primary and only)
from .extension import comfy_entrypoint

# V1 compatibility for legacy ComfyUI versions that don't support v3 yet
# These mappings are auto-generated from v3 nodes for backward compatibility
try:
    from .node_registration import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    WEB_DIRECTORY = "./js"
    __all__ = ['comfy_entrypoint', 'NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
except:
    # If v1 compat fails, v3-only mode
    __all__ = ['comfy_entrypoint']
