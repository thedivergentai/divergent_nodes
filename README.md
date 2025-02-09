# 👽 Divergent Nodes for ComfyUI

**✨ Enhance your ComfyUI workflows with Divergent Nodes – a growing collection of custom nodes designed to expand your creative possibilities!**

This repository offers a curated set of custom nodes that introduce new functionalities and streamline your ComfyUI experience.

**For detailed documentation, visit the [Divergent Nodes Wiki](https://github.com/thedivergentai/divergent_nodes/wiki).**

**Nodes Currently Included:**

*   **✨ Divergent CLIP Token Counter**: Precisely count CLIP tokens in your text prompts.
*   **🐬 DolphinVision Node**: Generate text descriptions of images using the DolphinVision 7b multimodal model.

---

## 🛠️ Installation Guide

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/thedivergentai/divergent_nodes.git custom_nodes/divergent_nodes
    ```
   (Or download as a ZIP and extract to `custom_nodes`.)

2.  **Install Dependencies:**

    ```bash
    cd custom_nodes/divergent_nodes
    pip install -r requirements.txt
    ```

3.  **Restart ComfyUI:** Restart to load the new nodes.

---

## 🧰 Node Details

### Divergent CLIP Token Counter

**Description:** Accurately count CLIP tokens in a text string. Essential for workflows with token limits.

**Inputs:** `text` (STRING, Multiline)
**Outputs:** `token_count` (INT)

**Key Features:** Zero token handling, robust text processing, CLIP standard compliance.

### 🐬 DolphinVision Node

**Description:** Generate text descriptions of images using the [DolphinVision 7b](https://huggingface.co/cognitivecomputations/dolphin-2_6-phi-2) model (hosted on Hugging Face).

**Inputs:** `image` (IMAGE TENSOR), `prompt` (STRING)
**Outputs:** `text` (STRING)

**Model Loading:**

*   `load_model(cache=False)`: Loads the model.
    *   `cache`: Uses a cached version if available.
*   `IS_CHANGED(image, prompt, **kwargs)`: Optimizes performance by checking for input changes.
*   `unload()`: Unloads the model from memory.

**Key Features:** Multimodal generation, flexible prompting, automatic model download.

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
