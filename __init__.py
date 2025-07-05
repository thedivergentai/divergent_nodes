# Main __init__.py for Divergent Nodes
# This file aggregates node mappings from node_registration.py

from .node_registration import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Define the WEB_DIRECTORY for client-side JavaScript files
WEB_DIRECTORY = "./js"

# Expose the aggregated mappings and WEB_DIRECTORY for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
