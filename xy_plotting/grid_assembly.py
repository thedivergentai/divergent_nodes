"""
Utility functions for assembling XY plot image grids and drawing labels using PIL/Pillow.
"""
import logging
import os
from typing import List, Tuple, Optional, Union, Sequence, TypeAlias
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageColor

# Define type aliases for clarity
PilImageT: TypeAlias = Image.Image
PilDrawT: TypeAlias = ImageDraw.ImageDraw
PilFontT: TypeAlias = Union[ImageFont.FreeTypeFont, ImageFont.ImageFont] # Allow both truetype and default bitmap
RgbTupleT: TypeAlias = Tuple[int, int, int]
TensorHWC: TypeAlias = torch.Tensor # Expected shape [H, W, C]
TensorGridHWC: TypeAlias = torch.Tensor # Expected shape [H_grid, W_grid, C]

# Setup logger for this module
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Grid Assembly ---

def assemble_image_grid(
    images: List[TensorHWC],
    rows: int,
    cols: int,
    row_gap: int = 0,
    col_gap: int = 0,
    background_color: Union[float, Tuple[float, float, float]] = 1.0 # White background (float for tensor)
) -> TensorGridHWC:
    """
    Assembles a list of image tensors (H, W, C) into a single grid tensor.

    Handles potential inconsistencies in image shapes by using the shape of the
    first image and logging a warning. Fills the grid background and places
    images according to the specified rows, columns, and gaps.

    Args:
        images (List[TensorHWC]): A list of image tensors, each expected to be [H, W, C].
                                   Assumes all images *should* have the same shape and dtype.
        rows (int): Number of rows in the desired grid.
        cols (int): Number of columns in the desired grid.
        row_gap (int): Gap between rows in pixels (default: 0).
        col_gap (int): Gap between columns in pixels (default: 0).
        background_color (Union[float, Tuple[float, float, float]]):
            Background color for the grid and gaps. Provide a float (e.g., 1.0 for white)
            or an RGB tuple (e.g., (1.0, 1.0, 1.0)). Defaults to white (1.0).

    Returns:
        TensorGridHWC: A single tensor representing the assembled grid [H_grid, W_grid, C].

    Raises:
        ValueError: If the input image list is empty.
    """
    if not images:
        logger.error("Cannot assemble grid: Input image list is empty.")
        raise ValueError("Input image list cannot be empty for grid assembly.")

    if rows <= 0 or cols <= 0:
        logger.error(f"Invalid grid dimensions: rows={rows}, cols={cols}. Must be positive.")
        raise ValueError("Grid rows and columns must be positive integers.")

    # --- Determine Grid Geometry and Validate Image Consistency ---
    try:
        # Use the first image to determine H, W, C and dtype
        first_image = images[0]
        H, W, C = first_image.shape
        dtype = first_image.dtype
        device = first_image.device # Keep grid on the same device if possible
        logger.debug(f"Base image properties: H={H}, W={W}, C={C}, dtype={dtype}, device={device}")

        # Check if all images have the same shape (optional but recommended)
        if not all(img.shape == (H, W, C) for img in images):
            logger.warning(f"Grid assembly: Images have inconsistent shapes. Using H={H}, W={W}, C={C} from the first image. Grid might look incorrect.")
            # Future enhancement: Option to resize images to fit?

    except (AttributeError, IndexError, ValueError) as e:
        logger.error(f"Cannot assemble grid: Invalid image data in list. Error: {e}", exc_info=True)
        raise ValueError("Invalid image data provided for grid assembly.") from e

    # --- Calculate Grid Dimensions ---
    grid_height = H * rows + max(0, row_gap * (rows - 1))
    grid_width = W * cols + max(0, col_gap * (cols - 1))
    logger.debug(f"Calculated grid dimensions: {grid_height}H x {grid_width}W")

    # --- Prepare Background Tensor ---
    try:
        # Determine background fill value based on channels and input type
        if C == 1 and isinstance(background_color, tuple) and len(background_color) >= 1:
            bg_value = background_color[0] # Use first value for grayscale
            logger.debug(f"Using grayscale background value: {bg_value}")
        elif C > 1 and isinstance(background_color, (int, float)):
            bg_value = (background_color,) * C # Repeat scalar for RGB/RGBA
            logger.debug(f"Using uniform multichannel background value: {bg_value}")
        elif isinstance(background_color, tuple) and len(background_color) == C:
             bg_value = background_color # Use provided tuple directly
             logger.debug(f"Using provided tuple background value: {bg_value}")
        else:
             logger.warning(f"Background color '{background_color}' incompatible with image channels C={C}. Defaulting to 1.0 (white).")
             bg_value = (1.0,) * C if C > 1 else 1.0

        # Create background grid tensor on the same device as input images
        grid_tensor = torch.full((grid_height, grid_width, C), fill_value=bg_value[0] if C==1 else 0.0, dtype=dtype, device=device)
        # Fill with tuple if multichannel
        if C > 1:
             for i in range(C):
                  grid_tensor[..., i] = bg_value[i]

    except Exception as e:
        logger.error(f"Failed to create background grid tensor: {e}", exc_info=True)
        raise RuntimeError("Failed to initialize grid tensor.") from e

    # --- Place Images onto Grid ---
    logger.info(f"Placing {len(images)} images onto {rows}x{cols} grid...")
    current_y = 0
    img_idx = 0
    for r in range(rows):
        current_x = 0
        for c in range(cols):
            if img_idx < len(images):
                try:
                    img_tensor = images[img_idx].to(device) # Ensure image is on the correct device
                    # Check shape again just before pasting (if warning was issued)
                    if img_tensor.shape == (H, W, C):
                        grid_tensor[current_y:current_y + H, current_x:current_x + W, :] = img_tensor
                    else:
                         logger.warning(f"Skipping image {img_idx} at grid pos ({r},{c}) due to shape mismatch ({img_tensor.shape} vs {(H,W,C)}).")
                         # Leave background color in this cell
                except Exception as e:
                    logger.error(f"Error placing image {img_idx} at grid pos ({r},{c}): {e}", exc_info=True)
                    # Leave background color, continue to next cell
            else:
                # Handle cases where there are fewer images than grid cells
                logger.debug(f"Grid cell ({r},{c}) left blank (Image index {img_idx} >= {len(images)}).")

            current_x += W + col_gap
            img_idx += 1
        current_y += H + row_gap

    logger.info("Grid assembly complete.")
    return grid_tensor

# --- Label Drawing ---

def _get_pil_font(font_size: int = 20) -> Optional[PilFontT]:
    """
    Attempts to load a usable font (Arial, then default PIL bitmap font).

    Args:
        font_size (int): Desired font size (primarily for TrueType).

    Returns:
        Optional[PilFontT]: A loaded PIL font object, or None if no font could be loaded.
    """
    font: Optional[PilFontT] = None
    font_paths_to_try = ["arial.ttf", "Arial.ttf"] # Common names
    # Add more common system font paths if needed, e.g., from matplotlib
    # font_paths_to_try.extend([...])

    # Try loading TrueType fonts first
    for font_name in font_paths_to_try:
        try:
            font = ImageFont.truetype(font_name, font_size)
            logger.debug(f"Loaded TrueType font: {font_name} at size {font_size}")
            return font # Success
        except IOError:
            logger.debug(f"TrueType font '{font_name}' not found.")
            continue # Try next font

    # If TrueType fails, try the default PIL bitmap font
    logger.warning(f"Common TrueType fonts not found. Attempting default PIL font.")
    try:
        font = ImageFont.load_default()
        logger.debug("Loaded default PIL bitmap font.")
        return font # Success
    except IOError as e:
        logger.error(f"Could not load any font for label drawing. PIL default font failed: {e}", exc_info=True)
        return None # Failure

def _calculate_required_padding(draw: PilDrawT, font: PilFontT,
                                x_labels: List[str], y_labels: List[str],
                                x_axis_label: str, y_axis_label: str
                                ) -> Tuple[int, int]:
    """Calculates the padding needed around the grid for labels."""
    left_padding = 0 # For Y labels
    top_padding = 0  # For X labels
    label_margin = 10 # Extra space around labels

    try:
        # Calculate padding for Y labels (left side)
        if y_axis_label:
            bbox = draw.textbbox((0, 0), y_axis_label, font=font)
            left_padding = bbox[2] - bbox[0] + label_margin * 2 # Width + margins
        elif y_labels:
            max_width = 0
            for label in y_labels:
                bbox = draw.textbbox((0, 0), label, font=font)
                max_width = max(max_width, bbox[2] - bbox[0])
            left_padding = max_width + label_margin # Max width + one margin

        # Calculate padding for X labels (top side)
        if x_axis_label:
            bbox = draw.textbbox((0, 0), x_axis_label, font=font)
            top_padding = bbox[3] - bbox[1] + label_margin * 2 # Height + margins
        elif x_labels:
            max_height = 0
            for label in x_labels:
                bbox = draw.textbbox((0, 0), label, font=font)
                max_height = max(max_height, bbox[3] - bbox[1])
            top_padding = max_height + label_margin # Max height + one margin

    except Exception as e:
        logger.error(f"Error calculating label padding: {e}", exc_info=True)
        # Return zero padding on error to avoid breaking layout entirely
        return 0, 0

    logger.debug(f"Calculated padding: Top={top_padding}, Left={left_padding}")
    return int(top_padding), int(left_padding)


def draw_labels_on_grid(
    grid_tensor: TensorGridHWC,
    x_labels: List[str],
    y_labels: List[str],
    x_axis_label: str = "",
    y_axis_label: str = "",
    font_size: int = 20,
    text_color: RgbTupleT = (0, 0, 0), # Black
    bg_color: RgbTupleT = (255, 255, 255) # White
) -> TensorGridHWC:
    """
    Draws X and Y axis labels onto a grid image tensor using PIL.

    Creates a new image with padding, pastes the original grid, and draws
    individual cell labels or overall axis labels.

    Args:
        grid_tensor (TensorGridHWC): The assembled grid image tensor [H_grid, W_grid, C].
                                     Assumed to be in CPU memory.
        x_labels (List[str]): List of labels for columns. Length should match grid columns.
        y_labels (List[str]): List of labels for rows. Length should match grid rows.
        x_axis_label (str): Optional overall label for the X axis (drawn at top).
        y_axis_label (str): Optional overall label for the Y axis (drawn at left).
        font_size (int): Font size for labels (default: 20).
        text_color (RgbTupleT): RGB tuple for label text color (default: black).
        bg_color (RgbTupleT): RGB tuple for the background/padding color (default: white).

    Returns:
        TensorGridHWC: A tensor representing the grid with labels drawn [H_new, W_new, C].
                       Returns the original tensor if font loading or drawing fails.
    """
    logger.info("Attempting to draw labels on grid...")
    if grid_tensor.dim() != 3:
         logger.error(f"Cannot draw labels: Input tensor has incorrect dimensions ({grid_tensor.dim()}). Expected 3 (H, W, C).")
         return grid_tensor
    if grid_tensor.shape[2] != 3:
         logger.warning(f"Drawing labels on non-RGB tensor (Channels={grid_tensor.shape[2]}). Color conversion might occur.")
         # PIL might handle some conversions, but best practice is RGB input

    try:
        # --- Font Loading ---
        font = _get_pil_font(font_size)
        if not font:
            logger.warning("No font loaded, cannot draw labels. Returning original grid.")
            return grid_tensor

        # --- Image Conversion (Tensor -> PIL) ---
        # Ensure tensor is on CPU and in uint8 format for PIL
        logger.debug("Converting grid tensor to PIL Image...")
        try:
            grid_np_uint8 = np.clip(grid_tensor.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            grid_pil: PilImageT = Image.fromarray(grid_np_uint8)
            # Ensure image mode is RGB for color drawing consistency
            if grid_pil.mode != 'RGB':
                logger.debug(f"Converting PIL image mode from {grid_pil.mode} to RGB.")
                grid_pil = grid_pil.convert('RGB')
        except Exception as e:
            logger.error(f"Failed to convert grid tensor to PIL Image: {e}", exc_info=True)
            return grid_tensor # Return original on conversion failure

        draw = ImageDraw.Draw(grid_pil)

        # --- Calculate Padding & Create Labeled Image Canvas ---
        top_padding, left_padding = _calculate_required_padding(
            draw, font, x_labels, y_labels, x_axis_label, y_axis_label
        )

        if top_padding == 0 and left_padding == 0:
             logger.info("No labels provided or padding calculation failed; skipping label drawing.")
             # No labels to draw if padding is zero (or calculation failed)
             # This check handles cases where only axis labels are given but are empty strings
             if not (x_labels or y_labels or x_axis_label or y_axis_label):
                  return grid_tensor # Nothing to draw

        new_width = left_padding + grid_pil.width
        new_height = top_padding + grid_pil.height
        logger.debug(f"Creating new labeled canvas: {new_width}W x {new_height}H")

        try:
            # Use the validated bg_color tuple
            labeled_pil = Image.new(grid_pil.mode, (new_width, new_height), color=bg_color)
            labeled_pil.paste(grid_pil, (left_padding, top_padding))
            # Get new draw context for the padded image
            draw = ImageDraw.Draw(labeled_pil)
        except Exception as e:
             logger.error(f"Failed to create padded image canvas: {e}", exc_info=True)
             return grid_tensor # Return original if canvas creation fails

        # --- Draw Labels ---
        grid_rows = len(y_labels)
        grid_cols = len(x_labels)
        if grid_rows == 0 or grid_cols == 0:
             logger.warning("Cannot draw labels: x_labels or y_labels list is empty.")
             # Return the padded image without labels if padding was added but lists are empty
             # Convert back to tensor
             labeled_np = np.array(labeled_pil).astype(np.float32) / 255.0
             return torch.from_numpy(labeled_np)


        # Estimate cell height/width (assuming uniform grid *without* gaps for label positioning)
        # This calculation is approximate and assumes the input grid_tensor did NOT include gaps.
        # If the grid assembly included gaps, this positioning will be slightly off.
        # A more robust solution would pass cell dimensions or calculate based on grid_tensor size and gaps.
        H_cell = grid_pil.height // grid_rows
        W_cell = grid_pil.width // grid_cols
        logger.debug(f"Estimated cell size for label positioning: {H_cell}H x {W_cell}W")

        # Draw Y-axis labels (Rows) - Placed to the left
        current_y_pos = top_padding # Starting Y position for the first cell's top edge
        if y_axis_label: # Draw single overall label centered vertically
             bbox = draw.textbbox((0, 0), y_axis_label, font=font)
             label_h = bbox[3] - bbox[1]
             # Center vertically within the entire grid height area
             text_y = top_padding + (grid_pil.height // 2) - (label_h // 2)
             # Position horizontally within the left padding area
             text_x = max(5, (left_padding - (bbox[2]-bbox[0])) // 2) # Center in padding
             draw.text((text_x, text_y), y_axis_label, fill=text_color, font=font)
             logger.debug(f"Drawing Y-axis label '{y_axis_label}' at ({text_x}, {text_y})")
        elif y_labels: # Draw individual row labels
             for r_idx, label in enumerate(y_labels):
                  bbox = draw.textbbox((0, 0), label, font=font)
                  label_h = bbox[3] - bbox[1]
                  label_w = bbox[2] - bbox[0]
                  # Calculate vertical center of the current cell
                  cell_center_y = current_y_pos + (H_cell // 2)
                  text_y = cell_center_y - (label_h // 2)
                  # Right-align text within the left padding area
                  text_x = max(5, left_padding - label_w - 5) # 5px margin from grid edge
                  draw.text((text_x, text_y), label, fill=text_color, font=font)
                  logger.debug(f"Drawing Y label '{label}' for row {r_idx} at ({text_x}, {text_y})")
                  current_y_pos += H_cell # Move to next row's estimated top edge

        # Draw X-axis labels (Columns) - Placed at the top
        current_x_pos = left_padding # Starting X position for the first cell's left edge
        if x_axis_label: # Draw single overall label centered horizontally
             bbox = draw.textbbox((0, 0), x_axis_label, font=font)
             label_w = bbox[2] - bbox[0]
             # Center horizontally within the entire grid width area
             text_x = left_padding + (grid_pil.width // 2) - (label_w // 2)
             # Position vertically within the top padding area
             text_y = max(5, (top_padding - (bbox[3]-bbox[1])) // 2) # Center in padding
             draw.text((text_x, text_y), x_axis_label, fill=text_color, font=font)
             logger.debug(f"Drawing X-axis label '{x_axis_label}' at ({text_x}, {text_y})")
        elif x_labels: # Draw individual column labels
             for c_idx, label in enumerate(x_labels):
                  bbox = draw.textbbox((0, 0), label, font=font)
                  label_w = bbox[2] - bbox[0]
                  label_h = bbox[3] - bbox[1]
                  # Calculate horizontal center of the current cell
                  cell_center_x = current_x_pos + (W_cell // 2)
                  text_x = cell_center_x - (label_w // 2)
                  # Position text vertically centered within the top padding
                  text_y = max(5, (top_padding - label_h) // 2)
                  draw.text((text_x, text_y), label, fill=text_color, font=font)
                  logger.debug(f"Drawing X label '{label}' for col {c_idx} at ({text_x}, {text_y})")
                  current_x_pos += W_cell # Move to next col's estimated left edge

        # --- Convert Labeled Image Back to Tensor ---
        logger.debug("Converting labeled PIL image back to tensor...")
        labeled_np = np.array(labeled_pil).astype(np.float32) / 255.0
        final_labeled_tensor = torch.from_numpy(labeled_np)
        logger.info("Label drawing complete.")
        return final_labeled_tensor

    except Exception as e:
        logger.error(f"Unexpected error during label drawing process: {e}", exc_info=True)
        return grid_tensor # Return original grid tensor on any unforeseen error
