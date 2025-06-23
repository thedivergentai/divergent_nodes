import os
import sys
import logging

# Setup a basic logger for the main __init__.py
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Define the base directory for custom nodes
# This assumes the __init__.py is directly inside the custom node folder
NODE_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# List of node sub-packages to import
NODE_PACKAGES = [
    "clip_utils",
    "google_ai",
    "xy_plotting",
    "koboldcpp",
    "image_utils",
    "musiq_utils",
]

# Dynamically import mappings from each node package
for package_name in NODE_PACKAGES:
    try:
        # Construct the full module path
        module_path = f".{package_name}"
        
        # Import the module
        module = __import__(module_path, fromlist=['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'])
        
        # Update the global mappings
        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
            logger.info(f"Successfully loaded NODE_CLASS_MAPPINGS from {package_name}")
        else:
            logger.warning(f"Module {package_name} does not expose NODE_CLASS_MAPPINGS.")

        if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            logger.info(f"Successfully loaded NODE_DISPLAY_NAME_MAPPINGS from {package_name}")
        else:
            logger.warning(f"Module {package_name} does not expose NODE_DISPLAY_NAME_MAPPINGS.")

    except ImportError as e:
        logger.error(f"Failed to import nodes from package '{package_name}': {e}", exc_info=True)
    except Exception as e:
        logger.critical(f"An unexpected error occurred while loading nodes from package '{package_name}': {e}", exc_info=True)

# Expose the aggregated mappings for ComfyUI
WEB_DIRECTORY = "./js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
