# Divergent Nodes - Custom Nodes for ComfyUI

This repository contains a collection of custom nodes for ComfyUI, designed to enhance your workflows with new functionalities. Currently, it includes nodes for:

- **Divergent CLIP Token Counter**:  Counts CLIP tokens in text input.
- **DeepSeek VL2 Node**: Runs the DeepSeek VL2 model for visual language tasks.

Each node is designed to be modular and easy to integrate into your existing ComfyUI setups. Refer to the individual node sections below for detailed usage instructions.

## Installation

1. Clone or download this repository into your `custom_nodes` directory
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Restart ComfyUI

## Node: Divergent CLIP Token Counter

### Description
Counts the number of CLIP tokens in a given text string using the CLIP tokenizer.

### Inputs
- **text**: Input text string to tokenize

### Outputs  
- **token_count**: Number of CLIP tokens in the input text

### Usage
1. Add the "Divergent CLIP Token Counter" node to your workflow
2. Connect a text input to the node
3. The node will output the token count which can be used in your workflow

### Notes
- Empty strings will return 0 tokens
- The node handles special characters and multilingual text
- Maximum token length is limited by CLIP's tokenizer (77 tokens)

## Node: DeepSeek VL2 Node

### Description
A ComfyUI custom node for running the DeepSeek VL2 model for visual language tasks.

### Inputs
- **prompt**: Input text prompt for the model.
- **quantization**: Quantization method for the model (bf16, nf4).
- **model_variant**: Variant of the DeepSeek VL2 model (base, small, tiny).
- **image**: Optional input image for visual language tasks.

### Outputs
- **output**: Text output from the DeepSeek VL2 model.

### Usage
1. Add the "DeepSeek VL2 Node" to your workflow.
2. Connect a text prompt to the "prompt" input.
3. Select the desired quantization and model variant.
4. Optionally connect an image to the "image" input for visual tasks.
5. The node will output the text response from the DeepSeek VL2 model.

### Notes
- Requires significant VRAM, especially for the base model and bf16 quantization.
- Supports both text-only and visual-language tasks.
- Model variants allow for trade-offs between performance and resource usage.

## License
MIT License
