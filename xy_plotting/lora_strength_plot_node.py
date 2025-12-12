"""
LoraStrengthXYPlot (v3): Generate XY plots for LoRA strength testing.
Migrated to ComfyUI v3 schema. Note: This is a complex plotting node.
"""
import logging
from typing import List
from comfy_api.latest import io

logger = logging.getLogger(__name__)


class LoraStrengthXYPlot(io.ComfyNode):
    """
    Generates XY plots for testing different LoRA strengths across multiple LoRAs.
    Useful for finding optimal LoRA strength combinations.
    """
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LoraStrengthXYPlot",
            display_name="XY Plot: LoRA/Strength",
            category="Divergent Nodes 👽/Utils",
            description="Generate XY plots for testing LoRA strengths combinations.",
            inputs=[
                io.String.Input("lora_names", multiline=True, tooltip="LoRA names (one per line)"),
                io.String.Input("strength_values", default="0.0, 0.5, 1.0", tooltip="Strength values to test (comma-separated)"),
                io.Int.Input("grid_columns", default=3, min=1, max=10, tooltip="Number of columns in grid"),
                io.Boolean.Input("annotate_images", default=True, optional=True, tooltip="Add text annotations to images"),
            ],
            outputs=[
                io.String.Output(display_name="plot_data"),
                io.Int.Output(display_name="total_combinations"),
            ],
        )
    
    @classmethod
    def execute(
        cls,
        lora_names: str,
        strength_values: str,
        grid_columns: int,
        annotate_images: bool = True
    ) -> io.NodeOutput:
        """Generate XY plot configuration for LoRA strength testing."""
        try:
            # Parse LoRA names
            loras = [name.strip() for name in lora_names.split('\n') if name.strip()]
            
            # Parse strength values
            strengths = [float(s.strip()) for s in strength_values.split(',') if s.strip()]
            
            if not loras:
                return io.NodeOutput("Error: No LoRA names provided", 0)
            
            if not strengths:
                return io.NodeOutput("Error: No strength values provided", 0)
            
            # Calculate combinations
            total_combinations = len(loras) * len(strengths)
            
            # Build plot data
            plot_data = f"XY Plot Configuration:\n"
            plot_data += f"LoRAs ({len(loras)}): {', '.join(loras)}\n"
            plot_data += f"Strengths ({len(strengths)}): {', '.join(map(str, strengths))}\n"
            plot_data += f"Total combinations: {total_combinations}\n"
            plot_data += f"Grid layout: {grid_columns} columns\n"
            plot_data += f"Annotations: {'Enabled' if annotate_images else 'Disabled'}\n"
            
            logger.info(f"📊 [LoraStrengthXYPlot] Generated plot for {total_combinations} combinations")
            return io.NodeOutput(plot_data, total_combinations)
            
        except Exception as e:
            error = f"Error generating plot configuration: {e}"
            logger.error(error)
            return io.NodeOutput(error, 0)
