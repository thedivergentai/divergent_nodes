# LoRA Strength Plot

This node generates an XY plot grid comparing different LoRA models (X-axis) against varying LoRA strengths (Y-axis). It uses provided Model, CLIP, and VAE inputs to generate images for each combination, then assembles them into a single grid.

## Parameters

- **model** (`MODEL`): The input base model.
- **clip** (`CLIP`): The input CLIP model.
- **vae** (`VAE`): The input VAE model.
- **lora_folder_path** (`STRING`, default: `loras/`): Path to the folder containing LoRA files (can be relative to `ComfyUI/models/loras` or an absolute path).
- **positive** (`CONDITIONING`): Positive conditioning for image generation.
- **negative** (`CONDITIONING`): Negative conditioning for image generation.
- **latent_image** (`LATENT`): The initial latent image for sampling.
- **seed** (`INT`, default: `0`): The starting seed for image generation. Each subsequent image in the plot will use an incremented seed.
- **steps** (`INT`, default: `20`, range: `1` to `10000`): Number of sampling steps.
- **cfg** (`FLOAT`, default: `7.0`, range: `0.0` to `100.0`): Classifier Free Guidance scale.
- **sampler_name** (`COMBO`): The sampler algorithm to use (e.g., `euler`, `dpmpp_2m`).
- **scheduler** (`COMBO`): The scheduler to use with the sampler (e.g., `normal`, `karras`).
- **x_lora_steps** (`INT`, default: `3`, range: `0` to `100`): Number of LoRAs to plot on the X-axis. `0` means all detected LoRAs, `1` means only the last detected LoRA.
- **y_strength_steps** (`INT`, default: `3`, range: `1` to `100`): Number of strength steps to plot on the Y-axis.
- **max_strength** (`FLOAT`, default: `1.0`, range: `0.0` to `5.0`): The maximum LoRA strength to test on the Y-axis. Strengths will be evenly distributed from 0.0 to this value.
- **save_individual_images** (`BOOLEAN`, default: `False`): If `True`, saves each individual image generated for the grid to the output folder.
- **display_last_image** (`BOOLEAN`, default: `False`): If `True`, outputs the last generated image as a separate preview.
- **output_folder_name** (`STRING`, default: `XYPlot_LoRA-Strength`): The name of the subfolder within ComfyUI's output directory where the plot and individual images (if enabled) will be saved.
- **row_gap** (`INT`, default: `10`, range: `0` to `200`): Pixel gap between rows in the assembled plot.
- **col_gap** (`INT`, default: `10`, range: `0` to `200`): Pixel gap between columns in the assembled plot.
- **draw_labels** (`BOOLEAN`, default: `True`): If `True`, draws labels for LoRA names and strengths on the plot.
- **x_axis_label** (`STRING`, default: `LoRA`): Custom label for the X-axis.
- **y_axis_label** (`STRING`, default: `Strength`): Custom label for the Y-axis.

## Outputs

- **xy_plot_image** (`IMAGE`): The assembled grid image of all generated variations.
- **last_generated_image** (`IMAGE`): The last individual image generated during the plot process (useful for previewing).

## Usage

Connect your base `MODEL`, `CLIP`, `VAE`, `positive` and `negative` conditionings, and a `latent_image`. Specify the `lora_folder_path` and configure the number of LoRAs and strength steps for the plot. The node will generate a grid of images, allowing you to visually compare the effects of different LoRAs at various strengths.
