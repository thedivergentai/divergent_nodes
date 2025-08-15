import logging
import sys
from colorama import Fore, Style, init

# Initialize colorama for cross-platform ANSI escape code support
init(autoreset=True)

# Define custom log level for special success messages
SUCCESS_HIGHLIGHT = 25
logging.addLevelName(SUCCESS_HIGHLIGHT, "SUCCESS")

class ColoredFormatter(logging.Formatter):
    """
    A custom formatter that adds ANSI escape codes for colored console output
    and prepends emojis and node names.
    """
    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
        "SUCCESS": Fore.MAGENTA + Style.BRIGHT, # For SUCCESS_HIGHLIGHT
    }

    EMOJIS = {
        "DEBUG": "🐛",
        "INFO": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "CRITICAL": "💥",
        "SUCCESS": "🎉✨",
    }

    def format(self, record):
        # Get the node name from the logger name (e.g., 'source.google_ai.GeminiNode' -> 'GeminiNode')
        # If the logger name is just 'source', use 'Divergent Nodes'
        node_name = record.name.split('.')[-1] if '.' in record.name else "Divergent Nodes"
        
        # Temporarily add 'node_name' to the record so it can be used in the format string
        record.node_name = node_name
        
        # Apply color and emoji to the levelname
        color = self.COLORS.get(record.levelname, Style.RESET_ALL)
        emoji = self.EMOJIS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        
        # Apply color to the message itself
        record.msg = f"{color}{emoji} {record.msg}{Style.RESET_ALL}"
        
        # Let the base formatter handle the rest of the formatting
        return super().format(record)

def setup_node_logging():
    """
    Configures logging for Divergent Nodes to use colored console output.
    This function should be called once per node package (e.g., in __init__.py).
    """
    # Get the logger for 'source' (our custom nodes root)
    divergent_nodes_logger = logging.getLogger('source')
    divergent_nodes_logger.setLevel(logging.DEBUG) # Set default level to DEBUG to capture all messages

    # Remove any existing handlers from the 'source' logger to avoid duplicate messages
    # and ensure our custom handler is the only one for this hierarchy.
    for handler in list(divergent_nodes_logger.handlers):
        divergent_nodes_logger.removeHandler(handler)

    # Prevent logs from propagating to the root logger, which might have default handlers
    # that strip our custom formatting.
    divergent_nodes_logger.propagate = False

    # Add a StreamHandler for direct console output
    console_handler = logging.StreamHandler(sys.stdout)
    
    # The formatter string now includes %(node_name)s which is dynamically added in ColoredFormatter.format
    # The levelname and message will be colored/emojified by ColoredFormatter.format
    formatter = ColoredFormatter('[👽 %(node_name)s] %(levelname)s: %(message)s')
    
    console_handler.setFormatter(formatter)
    divergent_nodes_logger.addHandler(console_handler)

    # It's generally good practice to ensure the root logger doesn't have a default handler
    # that might interfere, but setting propagate=False on 'source' logger is the primary fix.
    # This part is less critical if propagate=False is correctly applied.
    # However, if ComfyUI's root logger is already configured, we should not override it.
    # The key is that our 'source' logger's messages don't reach it.
