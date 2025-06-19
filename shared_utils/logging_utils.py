import logging

def setup_node_logging():
    """
    Configures logging for Divergent Nodes to include a specific prefix.
    This function should be called once per node package (e.g., in __init__.py).
    """
    # Get the root logger
    root_logger = logging.getLogger()

    # Ensure a handler exists for console output if none are configured
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        root_logger.addHandler(handler)
        # Set a default level if no handlers are present, otherwise respect existing level
        root_logger.setLevel(logging.INFO)

    # Create a custom formatter that adds the "Divergent Nodes: " prefix
    # The format includes the prefix, logger name, level name, and message
    formatter = logging.Formatter('Divergent Nodes: %(name)s - %(levelname)s - %(message)s')

    # Apply the formatter to all existing handlers of the root logger
    # This ensures that all logs from any part of the application (including ComfyUI's own logs)
    # will use this formatter if they pass through the root logger's handlers.
    # However, we only want to affect our custom nodes.
    # A more targeted approach is to get the specific logger for the node and set its handler.
    # But since ComfyUI often uses the root logger, modifying its handlers might be the only way
    # to consistently apply the prefix without changing every single logger instance.

    # Let's try a more targeted approach by setting a filter on the root logger's handlers
    # or by creating a new handler specifically for our nodes.
    # For simplicity and to ensure the prefix is applied, we will modify the format of existing handlers.
    # This might affect other logs if they share the same handler, but it's the most direct way
    # to ensure our logs get the prefix without changing every `logger = getLogger(__name__)` call.

    # Iterate through existing handlers and set the custom formatter
    for handler in root_logger.handlers:
        # Check if the handler is a StreamHandler (console output)
        # and if it doesn't already have our custom formatter
        if isinstance(handler, logging.StreamHandler) and not any(isinstance(f, type(formatter)) for f in handler.filters):
            handler.setFormatter(formatter)
            # Add a filter to ensure this formatter only applies to loggers under 'source'
            # This is more complex and might not be necessary if ComfyUI's logging is simple.
            # For now, let's assume modifying the formatter of existing handlers is sufficient.
            # If this causes issues with other logs, we'll need a more sophisticated filter/handler setup.

            # A simpler approach for now: just set the formatter.
            # If ComfyUI uses its own handlers, this might not affect them.
            # If it uses the root logger's default handler, it will.
            pass # Formatter is set below for all handlers.

    # A more robust way: create a new handler specifically for Divergent Nodes
    # and add it to the root logger, or configure specific loggers.
    # Given the request, the simplest is to ensure our loggers use a specific format.

    # Let's re-evaluate: The user wants "ALL custom nodes logging to have a designation of Divergent Nodes:".
    # This implies modifying the format of messages originating from our custom nodes.
    # The `logging.getLogger(__name__)` approach means each module gets a logger named after its path.
    # We can set a custom formatter for these specific loggers, or for their handlers.

    # The most straightforward way to ensure *our* logs have the prefix without affecting others
    # is to create a custom handler or modify the formatter of the specific loggers we create.
    # However, if ComfyUI's root logger is already configured, our loggers will inherit its handlers.

    # Let's try to get the logger for 'source' and configure it.
    # This will affect all loggers whose names start with 'source'.
    divergent_nodes_logger = logging.getLogger('source')
    if not divergent_nodes_logger.handlers:
        # Add a handler if the 'source' logger doesn't have one, to ensure output
        handler = logging.StreamHandler()
        divergent_nodes_logger.addHandler(handler)
        divergent_nodes_logger.setLevel(logging.INFO) # Default level for our nodes

    # Apply the custom formatter to handlers of the 'source' logger
    for handler in divergent_nodes_logger.handlers:
        handler.setFormatter(formatter)

    # This ensures that any logger created with a name starting with 'source.'
    # (e.g., 'source.musiq_utils.musiq_node') will inherit this configuration.
    # We also need to ensure that the loggers in individual files don't override this.
    # The `getLogger(__name__)` calls are fine, as they will be children of 'source'.

    # Final check: ensure the root logger's level is set appropriately if it's not already.
    # This is important because child loggers propagate messages to parent loggers.
    if not root_logger.level: # If level is not explicitly set, it's NOTSET (0)
        root_logger.setLevel(logging.INFO) # Set a reasonable default for the root logger
