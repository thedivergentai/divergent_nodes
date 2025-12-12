"""
SaveImageEnhancedNode (v3): Enhanced image saving with caption support.
Migrated to ComfyUI v3 schema. Note: This is a larger node due to file I/O complexity.
"""
import logging
import os
import torch
from comfy_api.latest import io

# Import shared utilities
from ..shared_utils.image_conversion import tensor_to_pil
from ..shared_utils.text_encoding_utils import ensure_utf8_friendly

logger = logging.getLogger(__name__)


class SaveImageEnhancedNode(io.ComfyNode):
    """
    Saves images with enhanced options including caption files and custom naming.
    """
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SaveImageEnhancedNode",
            display_name="Save Image Enhanced (DN)",
            category="Divergent Nodes 👽/Image",
            description="Save images with enhanced options including captions, custom paths, and date formatting.",
            inputs=[
                io.Image.Input("images", tooltip="Images to save"),
                io.String.Input("output_folder", default="output", tooltip="Output folder path"),
                io.String.Input("filename_prefix", default="ComfyUI_DN_%date:yyyy-MM-dd%", tooltip="Filename prefix with date placeholders"),
                io.Boolean.Input("add_counter_suffix", default=True, tooltip="Add counter suffix to prevent overwriting"),
                io.String.Input("caption", optional=True, lazy=True, multiline=True, tooltip="Optional caption to save with images"),
                io.String.Input("caption_file_extension", default=".txt", optional=True, tooltip="Extension for caption files"),
            ],
            outputs=[
                io.String.Output(display_name="saved_paths"),
            ],
            is_output_node=True,
        )
    
    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        output_folder: str,
        filename_prefix: str,
        add_counter_suffix: bool,
        caption: str = "",
        caption_file_extension: str = ".txt",
    ) -> io.NodeOutput:
        """Save images with enhanced options."""
        try:
            # Validate inputs
            if images is None or images.shape[0] == 0:
                error = "No images provided to save"
                logger.error(error)
                return io.NodeOutput(error)
            
            # Create output directory
            os.makedirs(output_folder, exist_ok=True)
            logger.info(f"📁 [SaveImageEnhanced] Saving to: {output_folder}")
            
            saved_paths = []
            batch_size = images.shape[0]
            
            for i in range(batch_size):
                # Get single image
                image_tensor = images[i:i+1]
                pil_image = tensor_to_pil(image_tensor)
                
                if pil_image is None:
                    logger.error(f"Failed to convert image {i} to PIL")
                    continue
                
                # Build filename
                filename = f"{filename_prefix}"
                if add_counter_suffix:
                    filename += f"_{i:05d}"
                filename += ".png"
                
                # Full path
                full_path = os.path.join(output_folder, filename)
                
                # Save image
                pil_image.save(full_path, format='PNG')
                saved_paths.append(full_path)
                logger.debug(f"✅ Saved: {full_path}")
                
                # Save caption if provided
                if caption and caption_file_extension:
                    caption_path = full_path.replace('.png', caption_file_extension)
                    safe_caption = ensure_utf8_friendly(caption)
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(safe_caption)
                    logger.debug(f"📄 Saved caption: {caption_path}")
            
            result = f"Saved {len(saved_paths)} images to {output_folder}"
            logger.info(f"✅ [SaveImageEnhanced] {result}")
            return io.NodeOutput(result)
            
        except Exception as e:
            error = f"Error saving images: {e}"
            logger.error(error, exc_info=True)
            return io.NodeOutput(error)
