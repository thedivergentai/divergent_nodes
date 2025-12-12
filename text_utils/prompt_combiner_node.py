"""
PromptCombiner node (v3): Combine multiple prompts with a customizable separator.
Migrated to ComfyUI v3 schema.
"""
import logging
from typing import List
from comfy_api.latest import io

# Import shared utilities
from ..shared_utils.prompt_utils import combine_prompts

logger = logging.getLogger(__name__)


class PromptCombiner(io.ComfyNode):
    """
    Combines multiple text prompts into a single string with a separator.
    Useful for merging prompt fragments from different sources.
    """
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PromptCombiner",
            display_name="Prompt Combiner",
            category="Divergent Nodes 👽/Text Utils",
            description="Combine multiple text prompts into a single prompt with a customizable separator.",
            inputs=[
                io.String.Input("prompts", tooltip="List of prompts to combine"),
                io.String.Input("separator", default=", ", tooltip="Separator to join prompts with"),
            ],
            outputs=[
                io.String.Output(display_name="combined_prompt"),
            ],
        )
    
    @classmethod
    def execute(cls, prompts: List[str], separator: str = ", ") -> io.NodeOutput:
        """
        Combine multiple prompts into one.
        
        Args:
            prompts: List of prompt strings
            separator: String to join prompts with
        
        Returns:
            NodeOutput containing combined prompt
        """
        logger.debug(f"Combining {len(prompts)} prompts with separator: '{separator}'")
        
        combined = combine_prompts(prompts, separator)
        
        logger.info(f"✅ Combined {len(prompts)} prompts into text of length {len(combined)}")
        return io.NodeOutput(combined)
