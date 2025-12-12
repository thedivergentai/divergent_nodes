"""
PreviewText node (v3): Display text outputs in the ComfyUI interface.
Migrated to ComfyUI v3 schema for future-proofing.
"""
import logging
from typing import Any
from comfy_api.latest import io, ui

logger = logging.getLogger(__name__)


class PreviewText(io.ComfyNode):
    """
    Displays text content in the ComfyUI interface.
    Useful for previewing generated text, prompts, or other string outputs.
    """
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PreviewText",
            display_name="Preview Text",
            category="Divergent Nodes 👽/Utils",
            description="Display text output in the UI for preview and debugging.",
            inputs=[
                io.String.Input("text", tooltip="Text to display"),
                io.String.Input("prefix", default="", optional=True, tooltip="Optional prefix label"),
            ],
            outputs=[
                io.String.Output()
            ],
            is_output_node=True,
        )
    
    @classmethod
    def execute(cls, text: Any, prefix: str = "") -> io.NodeOutput:
        """
        Display text in UI and pass it through.
        
        Args:
            text: Text to display (will be converted to string)
            prefix: Optional prefix for the display
        
        Returns:
            NodeOutput with text and UI preview
        """
        # Convert to string if needed
        text_str = str(text) if text is not None else ""
        
        # Log for console output
        if prefix:
            logger.info(f"📝 [PreviewText] {prefix}: {text_str[:100]}{'...' if len(text_str) > 100 else ''}")
        else:
            logger.info(f"📝 [PreviewText] {text_str[:100]}{'...' if len(text_str) > 100 else ''}")
        
        # Return with UI preview
        return io.NodeOutput(
            text_str,
            ui=ui.Text(text_str)
        )
