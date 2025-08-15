# Main __init__.py for Divergent Nodes
# This file aggregates node mappings from node_registration.py

from .node_registration import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .shared_utils.logging_utils import setup_node_logging

# Define the WEB_DIRECTORY for client-side JavaScript files
WEB_DIRECTORY = "./js"

# Setup custom logging for all Divergent Nodes
setup_node_logging()

# Expose the aggregated mappings and WEB_DIRECTORY for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
