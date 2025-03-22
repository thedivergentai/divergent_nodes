# 👽 Divergent Nodes for ComfyUI

**✨ Enhance your ComfyUI workflows with Divergent Nodes – a growing collection of custom nodes designed to expand your creative possibilities!**

This repository offers a curated set of custom nodes that introduce new functionalities and streamline your ComfyUI experience.

**For detailed documentation, visit the [Divergent Nodes Wiki](https://github.com/thedivergentai/divergent_nodes/wiki).**

**Nodes Currently Included:**

*   **✨ Divergent CLIP Token Counter**: Precisely count CLIP tokens in your text prompts.
*   **✨ UTF8 Encoder**: Ensures text is encoded in UTF-8 format to prevent workflow errors.
*   **💾 Data Store**: Stores any type of data locally within a specified directory, using a memory stack system.
*   **✨ Gemma Multimodal Ollama**: Integrates the Gemma model via Ollama for multimodal tasks, allowing you to process images and prompts.

---

## 🛠️ Installation

1.  **Navigate to your ComfyUI installation's `custom_nodes` directory.** This directory is located within your ComfyUI installation folder.

2.  **Clone the Repository:**

    ```bash
    git clone https://github.com/thedivergentai/divergent_nodes.git
    ```
   (Or download as a ZIP and extract to the `custom_nodes` directory.)

3.  **Install Dependencies:**

    ```bash
    cd divergent_nodes
    pip install -r requirements.txt
    ```

4.  **Restart ComfyUI:** Restart to load the new nodes.

---

## 🧰 Node Details

1.  **Navigate to your ComfyUI installation's `custom_nodes` directory.** This directory is located within your ComfyUI installation folder.

2.  **Clone the Repository:**

    ```bash
    git clone https://github.com/thedivergentai/divergent_nodes.git
    ```
   (Or download as a ZIP and extract to the `custom_nodes` directory.)

3.  **Install Dependencies:**

    ```bash
    cd divergent_nodes
    pip install -r requirements.txt
    ```

4.  **Restart ComfyUI:** Restart to load the new nodes.

---

## 🧰 Node Details

### Divergent CLIP Token Counter

**Description:** Accurately count CLIP tokens in a text string. Essential for workflows with token limits.

**Inputs:** `text` (STRING, Multiline)
**Outputs:** `token_count` (INT)

**Key Features:** Zero token handling, robust text processing, CLIP standard compliance.

### ✨ UTF8 Encoder

**Description:** Ensures text is encoded in UTF-8 format to prevent workflow errors caused by incorrect character encoding.

**Inputs:** `text` (STRING)
**Outputs:** `text` (STRING)

**Key Features:** Handles potential decoding issues, ensures consistent text encoding.

### 💾 Data Store

**Description:** Stores any type of data locally within a specified directory, using a memory stack system.

**Inputs:**

*   `data` (*): Any type of data to be stored.
*   `directory` (STRING): The directory to store the data in. Defaults to "store" in the repository root.
*   `memory_stack_size` (INT): The number of memory stacks to use. Minimum is 1.

**Outputs:**

*   None

**Key Features:** Stores any data type, uses a memory stack system, and saves data to text files.

### ✨ Gemma Multimodal Ollama

**Description:** Integrates the Gemma model via Ollama for multimodal tasks, allowing you to process images and prompts.

**Inputs:**

*   `ollama_base_url` (STRING): Base URL of the Ollama server (default: http://localhost:11434).
*   `model_name` (STRING): Name of the Gemma model in Ollama (default: gemma-3-27b-it-abliterated-IQ2_S.gguf).
*   `mmproj_file_url` (STRING): URL for the mmproj file (default: https://huggingface.co/bartowski/mlabonne_gemma-3-27b-it-abliterated-GGUF/blob/main/mmproj-mlabonne_gemma-3-27b-it-abliterated-f32.gguf).
*   `image` (IMAGE): ComfyUI Image object.
*   `prompt` (STRING): Text prompt.

**Outputs:**

*   `response` (STRING): The generated text response from Gemma.

**Key Features:** Multimodal processing with Gemma and Ollama, image and text input, easy integration.

---

## ❓ Troubleshooting

See the [Troubleshooting/FAQ](https://github.com/thedivergentai/divergent_nodes/wiki/Troubleshooting) page on the Wiki.

---

## 🚀 Future Development

*   Additional Nodes
*   User Interface Enhancements
*   Community Contributions (see the [Contributing](https://github.com/thedivergentai/divergent_nodes/wiki/Contributing) page on the Wiki)

---

## 🔑 Key Dependencies

*   ComfyUI
*   PyTorch
*   Transformers
*   Hugging Face Hub
*   GitPython

---

## ✅ ComfyUI Compatibility

Designed for the latest version of ComfyUI. Report compatibility issues on the GitHub repository.

---

## 📜 License

Divergent Nodes for ComfyUI is released under the [MIT License](LICENSE). See the `LICENSE` file for details.
