# 👽 Divergent Nodes for ComfyUI 

**✨ Enhance your ComfyUI workflows with Divergent Nodes – a growing collection of custom nodes designed to expand your creative possibilities!**

This repository offers a curated set of custom nodes that introduce new functionalities and streamline your ComfyUI experience, making it even more powerful and versatile. 

**Nodes Currently Included:**

*   **✨ Divergent CLIP Token Counter**:  Precisely count CLIP tokens in your text prompts, ensuring you stay within token limits.
*   **🐬 DolphinVision Node**: Generate text descriptions of images using the DolphinVision 7b multimodal model.

---

## 🛠️ Installation Guide

Getting started with Divergent Nodes is straightforward. Follow these simple steps to integrate them into your ComfyUI setup:

1.  **Clone the Repository:**

    Open your ComfyUI `custom_nodes` directory and clone the Divergent Nodes repository:

    ```bash
    git clone https://github.com/your-github-username/divergent_nodes.git custom_nodes/divergent_nodes
    ```
    Alternatively, you can download the repository as a ZIP file and extract its contents into the `custom_nodes` directory.

2.  **Install Dependencies:**

    Navigate into the `divergent_nodes` directory within `custom_nodes` and install the required Python packages using pip:

    ```bash
    cd custom_nodes/divergent_nodes
    pip install -r requirements.txt
    ```

3.  **Restart ComfyUI:**

    To ensure ComfyUI recognizes and loads the newly installed Divergent Nodes, restart the ComfyUI application.

---

## 🧰 Node Details: Divergent CLIP Token Counter

**Description:**

The **Divergent CLIP Token Counter** node provides a utility to accurately count the number of CLIP tokens in a given text string. This is invaluable for workflows where managing token counts is crucial, such as when working with language models that have token limits. By using the CLIP tokenizer, this node ensures precise token counting, mirroring how CLIP models process text.

**Inputs:**

*   **`text` (STRING, Multiline)**:  The text string you want to analyze and count tokens for. Supports multiline input for convenience.

**Outputs:**

*   **`token_count` (INT)**:  The total number of CLIP tokens identified in the input text.

**Usage Tips:**

1.  In your ComfyUI workflow, add the "Divergent CLIP Token Counter" node.
2.  Connect a text-providing node (e.g., a `TextArea` or `Load Text File` node) to the `text` input of the **Divergent CLIP Token Counter**.
3.  The node will automatically process the text and output the `token_count`. You can then use this count for various purposes within your workflow, such as displaying it or using it in conditional logic.

**💡 Key Features:**

*   **Zero Token Handling**:  Correctly returns `0` tokens when an empty string is provided as input.
*   **Robust Text Processing**:  Accurately handles special characters and multilingual text, ensuring reliable token counts across diverse text inputs.
*   **CLIP Standard Compliance**:  Utilizes the CLIP tokenizer to provide token counts that are consistent with CLIP model expectations, respecting the 77-token limit of CLIP's context window.

---

## 🐬 DolphinVision Node

**Description:**

The **DolphinVision** node allows you to generate text descriptions of images using the DolphinVision 7b multimodal model.

**Inputs:**

*   **`image` (IMAGE TENSOR)**: The input image tensor.
*   **`prompt` (STRING)**: The text prompt to guide the image description.

**Outputs:**

*   **`text` (STRING)**: The generated text description of the image.

**Usage Tips:**

1.  Add the "DolphinVision" node to your ComfyUI workflow.
2.  Connect an image-providing node (e.g., `Load Image`) to the `image` input.
3.  Connect a text prompt node (e.g., `Text Input`) to the `prompt` input.
4.  The node will generate a text description based on the image and prompt.

**💡 Key Features:**

*   **Multimodal Generation**: Leverages the DolphinVision 7b model for image-to-text generation.
*   **Flexible Prompting**: Allows you to customize the image description using text prompts.
*   **Automatic Model Download**: Downloads the DolphinVision 7b model files automatically.

---

## 📜 License

Divergent Nodes for ComfyUI is released under the [MIT License](LICENSE).  See the `LICENSE` file for full details.
