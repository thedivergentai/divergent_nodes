"""
StringToList node (v3): Split a single string into a list of strings by delimiter.
Migrated to ComfyUI v3 schema.
"""
import logging
from typing import List
from comfy_api.latest import io

logger = logging.getLogger(__name__)


class StringToList(io.ComfyNode):
    """
    Splits a single string into a list of strings using a delimiter.
    Useful for creating batch inputs from text files or multi-line prompts.
    """
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="StringToList",
            display_name="String to List",
            category="Divergent Nodes 👽/Text Utils",
            description="Split a string into a list of strings using a delimiter.",
            inputs=[
                io.String.Input("text", multiline=True, default="", tooltip="Text to split into list"),
                io.String.Input("delimiter", default="\\n", tooltip="Delimiter to split on (\\n for newlines)"),
                io.Boolean.Input("strip_whitespace", default=True, tooltip="Remove leading/trailing whitespace from each item"),
                io.Boolean.Input("remove_empty", default=True, tooltip="Remove empty strings from result"),
            ],
            outputs=[
                io.String.Output(display_name="strings"),
            ],
        )
    
    @classmethod
    def execute(
        cls,
        text: str,
        delimiter: str = "\\n",
        strip_whitespace: bool = True,
        remove_empty: bool = True
    ) -> io.NodeOutput:
        """
        Split text into a list of strings.
        
        Args:
            text: Input text
            delimiter: String to split on
            strip_whitespace: Whether to strip whitespace from each item
            remove_empty: Whether to remove empty strings
        
        Returns:
            NodeOutput containing list of strings
        """
        if not text:
            logger.warning("Empty text provided to StringToList")
            return io.NodeOutput([])
        
        # Handle escaped newlines
        if delimiter == "\\n":
            delimiter = "\n"
        elif delimiter == "\\t":
            delimiter = "\t"
        elif delimiter == "\\r":
            delimiter = "\r"
        
        # Split the text
        items = text.split(delimiter)
        
        # Process items
        if strip_whitespace:
            items = [item.strip() for item in items]
        
        if remove_empty:
            items = [item for item in items if item]
        
        logger.info(f"✅ Split text into {len(items)} strings using delimiter: '{delimiter}'")
        return io.NodeOutput(items)
