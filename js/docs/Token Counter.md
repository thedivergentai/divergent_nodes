# Token Counter

This node counts the number of CLIP tokens in a given text string using a specified Hugging Face tokenizer.

## Parameters

- **text** (`STRING`, multiline): The text string for which to count tokens.
- **tokenizer_name** (`COMBO`): Select the Hugging Face CLIP tokenizer to use (e.g., "openai/clip-vit-base-patch32").

## Outputs

- **token_count** (`INT`): The total number of tokens in the input text, including special tokens.

## Usage

Provide a text string and select a CLIP tokenizer. The node will output the total token count. This is useful for understanding the length of prompts or other text inputs in terms of tokens, which can be relevant for models with token limits.
