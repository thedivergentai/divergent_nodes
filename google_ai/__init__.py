# Expose node classes from this module
from .gemini_api_node import GeminiNode

# This __init__.py no longer defines NODE_CLASS_MAPPINGS or NODE_DISPLAY_NAME_MAPPINGS
# as they are now centralized in source/node_registration.py.

__all__ = ['GeminiNode']
