import json
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Assuming config.json is in the source/ directory, relative to shared_utils/
CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

_config_cache = None # Cache the config once loaded

def load_config() -> Dict[str, Any]:
    """
    Loads the configuration from config.json. Caches the result.
    Returns an empty dictionary if the file is not found or invalid.
    """
    global _config_cache
    if _config_cache is not None:
        logger.debug("Returning cached config.")
        return _config_cache

    logger.debug(f"Attempting to load config from: {CONFIG_FILE_PATH}")
    config_data = {}
    try:
        with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        logger.info("Config loaded successfully.")
        _config_cache = config_data # Cache the loaded config
    except FileNotFoundError:
        logger.warning(f"Config file not found at: {CONFIG_FILE_PATH}. Using empty configuration.")
        _config_cache = {} # Cache empty config
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding config file at {CONFIG_FILE_PATH}: {e}. Using empty configuration.", exc_info=True)
        _config_cache = {} # Cache empty config
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading config from {CONFIG_FILE_PATH}: {e}. Using empty configuration.", exc_info=True)
        _config_cache = {} # Cache empty config

    return _config_cache

# Example usage (for testing/demonstration, not part of the utility itself)
# if __name__ == "__main__":
#     config = load_config()
#     print("Loaded Config:", config)
#     print("Google API Key:", config.get("GOOGLE_API_KEY"))
