"""
ImageInfo node (v3): Extract metadata and dimensions from image tensors.
Migrated to ComfyUI v3 schema.
"""
import logging
import torch
from comfy_api.latest import io

logger = logging.getLogger(__name__)


class ImageInfo(io.ComfyNode):
    """
    Extracts metadata from an image tensor including dimensions, batch size, and aspect ratio.
    """
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ImageInfo",
            display_name="Image Info",
            category="Divergent Nodes 👽/Image",
            description="Extract metadata from an image tensor including dimensions and aspect ratio.",
            inputs=[
                io.Image.Input("image", tooltip="Image tensor to analyze"),
            ],
            outputs=[
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Int.Output(display_name="batch_size"),
                io.Int.Output(display_name="channels"),
                io.Float.Output(display_name="aspect_ratio"),
                io.Boolean.Output(display_name="is_landscape"),
            ],
        )
    
    @classmethod
    def execute(cls, image: torch.Tensor) -> io.NodeOutput:
        """
        Extract image information.
        
        Args:
            image: Image tensor with shape [B, H, W, C]
        
        Returns:
            NodeOutput with (width, height, batch_size, channels, aspect_ratio, is_landscape)
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(image)}")
        
        if image.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B, H, W, C], got shape {image.shape}")
        
        batch_size, height, width, channels = image.shape
        aspect_ratio = width / height if height > 0 else 0.0
        is_landscape = width > height
        
        logger.debug(f"Image info: {width}x{height}, batch={batch_size}, aspect={aspect_ratio:.2f}")
        
        return io.NodeOutput(width, height, batch_size, channels, aspect_ratio, is_landscape)
