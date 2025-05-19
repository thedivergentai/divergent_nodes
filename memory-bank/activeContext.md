# Active Context: SaveImageEnhancedNode and Node Registration

## Current Focus
- Development and integration of the `SaveImageEnhancedNode` to replace `SaveImageKJ` for improved image and caption saving.
- Correcting node registration strategy and category string to ensure nodes appear in the correct ComfyUI category.

## Immediate Next Steps
- Update Memory Bank files to reflect the completed work on the `SaveImageEnhancedNode`, the node registration fix, and the category string correction.
- Commit and push the latest changes.

## Decisions Made
- A new node, `SaveImageEnhancedNode`, was created in `source/image_utils/save_image_enhanced_node.py`.
- This node explicitly uses UTF-8 encoding for caption files and sanitizes caption text using `ensure_utf8_friendly`.
- An `add_counter_suffix` boolean input was added to control the filename numbering.
- Node mappings for the `image_utils` package were moved from the individual node file to `source/image_utils/__init__.py`.
- The main `source/__init__.py` was updated to import mappings from the `image_utils` package's `__init__.py`.
- The `ensure_utf8_friendly` utility function was copied into `save_image_enhanced_node.py` for easier access within the custom node environment.
- The `CATEGORY` string in `SaveImageEnhancedNode` was updated to "👽 Divergent Nodes/Image" to match the format of other nodes in the project.

## Patterns & Learnings
- ComfyUI node mappings should typically be defined in the `__init__.py` file of a custom node sub-package for correct discovery and categorization.
- Consistency in category strings, including emojis, is important for proper grouping in the ComfyUI node menu.
- Copying small, self-contained utility functions into node files can sometimes simplify imports in the custom node environment, though shared modules are preferred when import paths are manageable.
- Maintaining accurate Memory Bank documentation is crucial for tracking project state and decisions.
