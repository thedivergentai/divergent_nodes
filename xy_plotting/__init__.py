# Expose node classes from this module
from .lora_strength_plot_node import LoraStrengthXYPlot

# Define mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "LoraStrengthXYPlot": LoraStrengthXYPlot,
}

# Optional: Define display names if different from class names
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraStrengthXYPlot": "XY Plot: LoRA/Strength",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
