from .musiq_node import MusiQNode
from ..shared_utils.logging_utils import setup_node_logging

# Setup logging for this node package
setup_node_logging()

NODE_CLASS_MAPPINGS = {
    "MusiQNode": MusiQNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MusiQNode": "MusiQ Image Scorer"
}
