# Save Image Enhanced

This node saves input images to a specified directory with enhanced options, including custom filename prefixes, optional caption saving, and an incrementing counter suffix.

## Parameters

- **images** (`IMAGE`): The images to save.
- **filename_prefix** (`STRING`, default: `ComfyUI_DN_%date:yyyy-MM-dd%`): The prefix for the saved file. This supports formatting information such as `%date:yyyy-MM-dd%` (for current date), `%Empty Latent Image.width%` (to include values from other nodes), and `%batch_num%` (for the current image's batch number).
- **output_folder** (`STRING`, default: `output`): The folder to save the images to. This can be a path relative to ComfyUI's output directory or an absolute path.
- **add_counter_suffix** (`BOOLEAN`, default: `True`): If `True`, an incrementing numerical suffix (e.g., `_00001`) is added to the filename to prevent overwriting existing files.
- **caption_file_extension** (`STRING`, optional, default: `.txt`): The file extension for the optional caption file (e.g., `.txt`, `.caption`).
- **caption** (`STRING`, optional, forceInput: `True`): A string to save as a separate text file alongside the image. This can be used for image descriptions or metadata.

## Outputs

- **last_filename_saved** (`STRING`): The full path of the last image file that was saved.

## Usage

Connect images to the `images` input. Configure the `filename_prefix` and `output_folder` as desired. You can also provide a `caption` string to be saved in a separate text file with the image. The `add_counter_suffix` option helps manage file versions.
