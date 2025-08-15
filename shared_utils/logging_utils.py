import logging
import json
import sys
from colorama import Fore, Style, init

# Initialize colorama for cross-platform ANSI escape code support
init(autoreset=True)

# Try to import ComfyUI's server, which is needed to send messages to the frontend
try:
    from comfy.server import PromptServer
except ImportError:
    PromptServer = None
    print(f"{Fore.YELLOW}[WARN] ComfyUI's PromptServer not found. Toast logging will be disabled.{Style.RESET_ALL}")

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
        log_message = super().format(record)
        
        # Get the node name from the logger name (e.g., 'source.GeminiNode' -> 'GeminiNode')
        node_name = record.name.split('.')[-1] if '.' in record.name else "Divergent Nodes"
        
        color = self.COLORS.get(record.levelname, Style.RESET_ALL)
        emoji = self.EMOJIS.get(record.levelname, "")

        # Format for console output
        formatted_message = f"{color}[👽 {node_name}] {emoji} {record.levelname}: {record.message}{Style.RESET_ALL}"
        
        # Store the original message for the toast handler's detail field
        record.original_message = record.message
        record.message = formatted_message # Overwrite record.message for console output

        return formatted_message

class ComfyUIToastHandler(logging.Handler):
    """
    A custom logging handler that sends log records to the ComfyUI frontend
    as toast notifications via WebSocket.
    """
    def emit(self, record):
        if PromptServer is None:
            # Fallback to console if PromptServer is not available
            sys.stdout.write(record.message + '\n') # Use the already formatted message
            return

        try:
            # Determine severity for toast
            if record.levelno >= logging.ERROR:
                severity = "error"
            elif record.levelno >= logging.WARNING:
                severity = "warn"
            elif record.levelno == SUCCESS_HIGHLIGHT:
                severity = "success" # Custom severity for toast if supported
            elif record.levelno >= logging.INFO:
                severity = "info"
            else:
                severity = "info" # Default for debug/notset

            # Get the node name from the logger name
            node_name = record.name.split('.')[-1] if '.' in record.name else "Divergent Nodes"
            emoji = ColoredFormatter.EMOJIS.get(record.levelname, "")

            # Prepare the message data for toast
            message_data = {
                "severity": severity,
                "summary": f"[👽 {node_name}] {emoji} {record.levelname}",
                "detail": record.original_message, # Use the original message for toast detail
                "life": 5000 if severity in ["error", "warn"] else 3000 # Longer life for errors/warnings
            }

            # Send the message to the frontend via a custom WebSocket event
            # The 'divergent_nodes_toast' event name must match the one in the JS extension
            PromptServer.instance.send_sync("divergent_nodes_toast", message_data)
        except Exception as e:
            # Fallback to console logging if sending to frontend fails
            sys.stderr.write(f"{Fore.RED}Failed to send log to ComfyUI Toast API: {e}{Style.RESET_ALL}\n")
            sys.stderr.write(record.message + '\n') # Use the already formatted message


def setup_node_logging():
    """
    Configures logging for Divergent Nodes to use the ComfyUI Toast API and colored console output.
    This function should be called once per node package (e.g., in __init__.py).
    """
    # Get the logger for 'source' (our custom nodes root)
    divergent_nodes_logger = logging.getLogger('source')
    divergent_nodes_logger.setLevel(logging.DEBUG) # Set default level to DEBUG to capture all messages

    # Remove any existing handlers to avoid duplicate messages
    for handler in list(divergent_nodes_logger.handlers):
        divergent_nodes_logger.removeHandler(handler)

    # Add our custom ComfyUIToastHandler
    toast_handler = ComfyUIToastHandler()
    # Use the ColoredFormatter for console output (which is the fallback for toast)
    # and for formatting the summary/detail for the toast itself.
    formatter = ColoredFormatter('%(message)s') # Formatter will handle prefixing and coloring
    toast_handler.setFormatter(formatter)
    divergent_nodes_logger.addHandler(toast_handler)

    # Also add a StreamHandler for direct console output, in case PromptServer is None
    # This ensures messages are always visible in the console, even if toast fails.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    divergent_nodes_logger.addHandler(console_handler)

    # Ensure the root logger's level is set appropriately if it's not already.
    root_logger = logging.getLogger()
    if not root_logger.level:
        root_logger.setLevel(logging.INFO) # Set a reasonable default for the root logger
