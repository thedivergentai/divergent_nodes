# Expose node classes from this module
from .token_counter_node import CLIPTokenCounter

# This __init__.py no longer defines NODE_CLASS_MAPPINGS or NODE_DISPLAY_NAME_MAPPINGS
# as they are now centralized in source/node_registration.py.
# The setup_node_logging() call has also been removed from here.

__all__ = ['CLIPTokenCounter']
