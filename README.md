# Divergent Nodes - Custom ComfyUI Nodes

This repository contains custom nodes for ComfyUI.

## Installation

1.  Clone this repository into your `ComfyUI/custom_nodes/` directory:
    ```bash
    cd ComfyUI/custom_nodes/
    git clone https://github.com/thedivergentai/divergent_nodes.git divergent_nodes
    ```
2.  **Set up API Key (for Gemini Node):**
    *   Create a `.env` file in the `divergent_nodes` directory (this directory).
    *   Add your Google AI Studio API key to the `.env` file like this: `GEMINI_API_KEY=YOUR_API_KEY_HERE`
    *   See the `.env.example` file for the format.
    *   The `.env` file is automatically ignored by git via `.gitignore`.
3.  Install/update the required Python dependencies:
    ```bash
    cd divergent_nodes
    pip install -r requirements.txt
    ```
4.  Restart ComfyUI.

The nodes should now be available in their respective categories when you right-click on the ComfyUI canvas.

## Nodes

### CLIP Token Counter

Counts the number of tokens generated by a CLIP tokenizer for the input text.

**Inputs:**

*   `text` (STRING): The text string you want to analyze.
*   `tokenizer_name` (STRING): The name of the Hugging Face CLIP tokenizer model to use (e.g., `openai/clip-vit-base-patch32`, `stabilityai/stable-diffusion-clip-vit-large-patch14`). Defaults to `openai/clip-vit-base-patch32`.

**Outputs:**

*   `token_count` (INT): The total number of tokens generated for the input text by the selected tokenizer.

**Category:** `Divergent Nodes 👽/Text`

### Gemini API Node

Connects to the Google Gemini API to generate text based on a prompt and optional image input.

**Inputs:**

*   `model` (COMBO): Select the Gemini model to use. The list is dynamically fetched from the API if your API key is configured, otherwise defaults are shown.
*   `prompt` (STRING): The text prompt for the model.
*   `image_optional` (IMAGE): An optional image input for multimodal prompts.
*   `temperature` (FLOAT): Controls randomness (0.0-1.0). Lower values are more deterministic.
*   `top_p` (FLOAT): Nucleus sampling parameter (0.0-1.0).
*   `top_k` (INT): Top-k sampling parameter.
*   `max_output_tokens` (INT): Maximum number of tokens to generate.
*   `safety_harassment` (COMBO): Block threshold for harassment content (Options: "Default (Unspecified)", "Block Low & Above", "Block Medium & Above", "Block High Only", "Block None").
*   `safety_hate_speech` (COMBO): Block threshold for hate speech content (Options: "Default (Unspecified)", "Block Low & Above", "Block Medium & Above", "Block High Only", "Block None").
*   `safety_sexually_explicit` (COMBO): Block threshold for sexually explicit content (Options: "Default (Unspecified)", "Block Low & Above", "Block Medium & Above", "Block High Only", "Block None").
*   `safety_dangerous_content` (COMBO): Block threshold for dangerous content (Options: "Default (Unspecified)", "Block Low & Above", "Block Medium & Above", "Block High Only", "Block None").

**Outputs:**

*   `text` (STRING): The generated text response from the Gemini API.

**Category:** `Divergent Nodes 👽/Gemini`

### KoboldCpp Launcher (Advanced)

Launches and manages a KoboldCpp instance (`koboldcpp_cu12.exe` or similar) in the background for text generation. This node provides full control over the KoboldCpp launch parameters and uses its API for generation requests. It caches running instances based on setup parameters for efficiency.

**Prerequisites:**

*   You **must** have a working KoboldCpp executable (e.g., `koboldcpp_cu12.exe`). Download it from the [KoboldCpp releases page](https://github.com/LostRuins/koboldcpp/releases/latest).
*   The node needs the correct path to this executable via the `koboldcpp_path` input.
*   You need the `.gguf` model file(s) you intend to use.
*   For image input, you need the corresponding multimodal projector (`.gguf`) file specified in `mmproj_path`.
*   The `requests` Python library must be installed (`pip install -r requirements.txt`).

**How it Works (Hybrid Launch + API + Caching):**

1.  **Launch/Cache:** When executed, the node checks its internal cache for a running KoboldCpp instance matching the exact *Setup Arguments* provided.
    *   **Cache Hit:** If found and responsive, it reuses the existing instance.
    *   **Cache Miss:** If not found, it terminates any *other* cached instance and launches a new KoboldCpp process using the provided *Setup Arguments*. It finds a free port, waits for the API to be ready, and caches the new process.
2.  **API Call:** It prepares a JSON payload with the *Generation Arguments* (prompt, image, sampling settings).
3.  **Image Handling:** If an `image_optional` is provided, it's converted to Base64 and included in the payload. The prompt is formatted to indicate an image is present.
4.  **Request/Response:** Sends the payload to the `/api/v1/generate` endpoint of the managed KoboldCpp instance and returns the extracted text response.
5.  **Cleanup:** Attempts to terminate cached processes when ComfyUI exits normally (may fail on crash/force-kill).

**Inputs:**

*   **Setup Arguments (Used for Launching/Caching):**
    *   `koboldcpp_path` (STRING): Full path to your `koboldcpp_cu12.exe`.
    *   `model_path` (STRING): Full path to the primary `.gguf` model file.
    *   `gpu_acceleration` (COMBO): GPU backend ("None", "CuBLAS", "CLBlast", "Vulkan"). Default: "CuBLAS".
    *   `n_gpu_layers` (INT): Layers to offload (-1=auto). Default: -1.
    *   `context_size` (INT): Model context size. Default: 4096.
    *   `mmproj_path` (STRING): Optional path to `.gguf` multimodal projector. **Required for image input.**
    *   `threads` (INT): CPU threads (0=auto). Default: 0.
    *   `use_mmap` (BOOLEAN): Enable memory mapping. Default: True.
    *   `use_mlock` (BOOLEAN): Lock model in RAM. Default: False.
    *   `flash_attention` (BOOLEAN): Enable Flash Attention. Default: False.
    *   `quant_kv` (COMBO): KV cache quantization ("0: f16", "1: q8", "2: q4"). Default: "0: f16".
    *   `extra_cli_args` (STRING): Optional *setup* flags (e.g., `--useclblast 1 0`). **Do not** include generation or API flags.
*   **Generation Arguments (Passed via API JSON):**
    *   `prompt` (STRING): The text prompt.
    *   `max_length` (INT): Max tokens to generate. Default: 512.
    *   `temperature` (FLOAT): Sampling temperature. Default: 0.7.
    *   `top_p` (FLOAT): Nucleus sampling p. Default: 0.92.
    *   `top_k` (INT): Top-k sampling (0=disable). Default: 0.
    *   `rep_pen` (FLOAT): Repetition penalty. Default: 1.1.
    *   `image_optional` (IMAGE): Optional image input. Requires `mmproj_path`.
    *   `stop_sequence` (STRING): Optional comma/newline separated stop sequences.

**Outputs:**

*   `text` (STRING): The generated text response from KoboldCpp. Errors are returned here too.

**Category:** `Divergent Nodes 👽/KoboldCpp`

### KoboldCpp API Connector (Basic)

Connects to an **already running** KoboldCpp instance via its API to perform text generation. This node **does not** launch or manage the KoboldCpp process itself.

**Prerequisites:**

*   You must have a KoboldCpp instance running separately, accessible via the network.
*   The `requests` Python library must be installed (`pip install -r requirements.txt`).

**How it Works:**

1.  **Check Connection:** Verifies it can reach the API at the specified `api_url`.
2.  **Prepare Payload:** Creates a JSON payload with the prompt, image (if provided, converted to Base64), and generation settings.
3.  **API Call:** Sends the payload via HTTP POST to the `/api/v1/generate` endpoint of the running KoboldCpp instance.
4.  **Response:** Parses the API response and returns the generated text.

**Inputs:**

*   `api_url` (STRING): The base URL of the running KoboldCpp instance (e.g., `http://localhost:5001`). Default: `http://localhost:5001`.
*   `prompt` (STRING): The text prompt.
*   `max_length` (INT): Max tokens to generate. Default: 512.
*   `temperature` (FLOAT): Sampling temperature. Default: 0.7.
*   `top_p` (FLOAT): Nucleus sampling p. Default: 0.92.
*   `top_k` (INT): Top-k sampling (0=disable). Default: 0.
*   `rep_pen` (FLOAT): Repetition penalty. Default: 1.1.
*   `image_optional` (IMAGE): Optional image input. Requires the connected KoboldCpp instance to have loaded the appropriate model and mmproj file.
*   `stop_sequence` (STRING): Optional comma/newline separated stop sequences.

**Outputs:**

*   `text` (STRING): The generated text response from the KoboldCpp API. Errors are returned here too.

**Category:** `Divergent Nodes 👽/KoboldCpp`
