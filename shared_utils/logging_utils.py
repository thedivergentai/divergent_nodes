import logging
import json
import sys

# Try to import ComfyUI's server, which is needed to send messages to the frontend
try:
    from comfy.server import PromptServer
except ImportError:
    PromptServer = None
    print("[WARN] ComfyUI's PromptServer not found. Toast logging will be disabled.")

class ComfyUIToastHandler(logging.Handler):
    """
    A custom logging handler that sends log records to the ComfyUI frontend
    as toast notifications via WebSocket.
    """
    def emit(self, record):
        if PromptServer is None:
            # Fallback to console if PromptServer is not available
            sys.stdout.write(self.format(record) + '\n')
            return

        try:
            # Determine severity for toast
            if record.levelno >= logging.ERROR:
                severity = "error"
            elif record.levelno >= logging.WARNING:
                severity = "warn"
            elif record.levelno >= logging.INFO:
                severity = "info"
            else:
                severity = "info" # Default for debug/notset

            # Prepare the message data
            message_data = {
                "severity": severity,
                "summary": f"Divergent Nodes: {record.levelname}",
                "detail": self.format(record), # Use the formatted message as detail
                "life": 5000 if severity in ["error", "warn"] else 3000 # Longer life for errors/warnings
            }

            # Send the message to the frontend via a custom WebSocket event
            # The 'divergent_nodes_toast' event name must match the one in the JS extension
            PromptServer.instance.send_sync("divergent_nodes_toast", message_data)
        except Exception as e:
            # Fallback to console logging if sending to frontend fails
            sys.stderr.write(f"Failed to send log to ComfyUI Toast API: {e}\n")
            sys.stderr.write(self.format(record) + '\n')


def setup_node_logging():
    """
    Configures logging for Divergent Nodes to use the ComfyUI Toast API.
    This function should be called once per node package (e.g., in __init__.py).
    """
    # Get the logger for 'source' (our custom nodes root)
    divergent_nodes_logger = logging.getLogger('source')
    divergent_nodes_logger.setLevel(logging.INFO) # Set default level for our nodes

    # Remove any existing handlers that might be sending to console
    # to avoid duplicate messages if we're fully switching to toast
    for handler in list(divergent_nodes_logger.handlers):
        if isinstance(handler, logging.StreamHandler):
            divergent_nodes_logger.removeHandler(handler)

    # Add our custom ComfyUIToastHandler if it's not already added
    if not any(isinstance(h, ComfyUIToastHandler) for h in divergent_nodes_logger.handlers):
        toast_handler = ComfyUIToastHandler()
        # A simple formatter for the handler, the JS will handle the prefix
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        toast_handler.setFormatter(formatter)
        divergent_nodes_logger.addHandler(toast_handler)

    # Ensure the root logger's level is set appropriately if it's not already.
    # This is important because child loggers propagate messages to parent loggers.
    root_logger = logging.getLogger()
    if not root_logger.level: # If level is not explicitly set, it's NOTSET (0)
        root_logger.setLevel(logging.INFO) # Set a reasonable default for the root logger
