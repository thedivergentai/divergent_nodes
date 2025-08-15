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
        # Create a copy to avoid modifying the original record for other handlers if any
        record_copy = logging.makeLogRecord(record.__dict__)
        
        # Get the node name from the logger name (e.g., 'source.GeminiNode' -> 'GeminiNode')
        node_name = record_copy.name.split('.')[-1] if '.' in record_copy.name else "Divergent Nodes"
        
        color = self.COLORS.get(record_copy.levelname, Style.RESET_ALL)
        emoji = self.EMOJIS.get(record_copy.levelname, "")

        # Format for console output
        record_copy.message = f"{color}[👽 {node_name}] {emoji} {record_copy.levelname}: {record_copy.getMessage()}{Style.RESET_ALL}"
        
        return super().format(record_copy)

def setup_node_logging():
    """
    Configures logging for Divergent Nodes to use colored console output.
    This function should be called once per node package (e.g., in __init__.py).
    """
    # Get the logger for 'source' (our custom nodes root)
    divergent_nodes_logger = logging.getLogger('source')
    divergent_nodes_logger.setLevel(logging.DEBUG) # Set default level to DEBUG to capture all messages

    # Remove any existing handlers to avoid duplicate messages
    for handler in list(divergent_nodes_logger.handlers):
        divergent_nodes_logger.removeHandler(handler)

    # Add a StreamHandler for direct console output
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = ColoredFormatter('%(message)s') # Formatter will handle prefixing and coloring
    console_handler.setFormatter(formatter)
    divergent_nodes_logger.addHandler(console_handler)

    # Ensure the root logger's level is set appropriately if it's not already.
    root_logger = logging.getLogger()
    if not root_logger.level:
        root_logger.setLevel(logging.INFO) # Set a reasonable default for the root logger
