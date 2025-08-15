# Divergent Nodes - Custom ComfyUI Nodes

This repository contains a collection of custom nodes for ComfyUI designed to integrate external AI models, provide utilities, and enable advanced workflows.

## Installation

1.  **Clone Repository:**
    Navigate to your `ComfyUI/custom_nodes/` directory and clone this repository:
    ```bash
    cd ComfyUI/custom_nodes/
    git clone https://github.com/thedivergentai/divergent_nodes.git divergent_nodes
    ```
    *(Note: If you cloned previously, you can update with `git pull` inside the `divergent_nodes` directory)*

2.  **Install Dependencies:**
    Install the required Python packages:
    ```bash
    cd divergent_nodes
    pip install -r requirements.txt
    ```

3.  **Set up API Key (for Gemini Node):**
    *   **Recommended:** Create or update `config.json` in the `divergent_nodes` directory (this directory) with your Google AI Studio API key:
        ```json
        {
          "GOOGLE_API_KEY": "YOUR_API_KEY_HERE"
        }
        ```
    *   **Alternatively (less preferred):** Create a file named `.env` in the `divergent_nodes` directory (this directory).
    *   Add your Google AI Studio API key to the `.env` file in the following format:
        ```
        GOOGLE_API_KEY=YOUR_API_KEY_HERE
        ```
    *   Refer to the `example_config.json` or `.env.example` file for guidance.
    *   The `config.json` and `.env` files are included in `.gitignore` and will not be tracked by Git.

4.  **Restart ComfyUI:**
    Ensure you fully restart the ComfyUI server after installation/updates.

The nodes should now appear in the ComfyUI node menu under their respective categories.

## Enhanced Logging

Divergent Nodes features enhanced logging in your ComfyUI console for better readability and a more enjoyable experience.

*   **Clear Identification:** All log messages from Divergent Nodes are prefixed with `[👽 NodeName]` (e.g., `[👽 GeminiNode]`) to easily identify their source.
*   **Visual Cues:** Log messages are colored and include emojis to quickly convey their status:
    *   **INFO:** ✅ Green - For successful operations and general information.
    *   **WARNING:** ⚠️ Yellow - For non-critical issues or important notices.
    *   **ERROR:** ❌ Red - For failures and critical problems.
    *   **DEBUG:** 🐛 Cyan - For detailed debugging information (usually hidden).
    *   **SPECIAL SUCCESS:** 🎉✨ Bold Bright Magenta - For major milestones like successful image generation or model downloads.

For a detailed key of all log messages and their meanings, please refer to the [Divergent Nodes Wiki Logging section](divergent_nodes.wiki/Logging.md).

## Included Nodes

This pack currently includes the following nodes:

*   **CLIP Token Counter** (`Divergent AI 👽/Text Utils`): Counts tokens for given text using a selected CLIP tokenizer.
*   **Divergent Gemini Node** (`Divergent AI 👽/Gemini`): Generates text (optionally using image input) via the Google Gemini API. Requires API key. Supports extended thinking and thinking token budget.
*   **KoboldCpp API Connector (Basic)** (`Divergent AI 👽/KoboldCpp`): Connects to an *already running* KoboldCpp instance for text generation.
*   **LoRA Strength XY Plot** (`Divergent AI 👽/XY Plots`): Generates an image grid comparing different LoRAs (X-axis) against varying model strengths (Y-axis).
*   **MusiQ Image Scorer** (`Divergent AI 👽/MusiQ`): Scores images based on aesthetic and technical quality using Google's MusiQ models.
*   **Save Image Enhanced (DN)** (`Divergent AI 👽/Image`): Saves images with enhanced options including custom output folder, filename prefixing, and optional caption saving.

---

## Example Workflows

Example workflows demonstrating node usage can be found in the [Divergent Nodes Wiki Examples section](divergent_nodes.wiki/Examples.md).

## Contributing

Contribution guidelines can be found in the [Divergent Nodes Wiki Contributing section](divergent_nodes.wiki/Contributing.md).

## License

License information for Divergent Nodes can be found in the project's GitHub repository.
