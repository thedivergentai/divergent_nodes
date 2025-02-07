# Divergent Nodes - CLIP Token Counter

A ComfyUI custom node for counting CLIP tokens in text input.

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

## License
MIT License
