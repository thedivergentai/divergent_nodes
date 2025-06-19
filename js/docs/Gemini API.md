# Gemini API

This node interacts with the Google Gemini API for text generation, optionally using an image input for multimodal models. It requires a `GEMINI_API_KEY` environment variable to be set.

## Parameters

- **model** (`COMBO`): Select the Gemini model to use (e.g., `gemini-pro`, `gemini-pro-vision`, `gemini-1.5-flash-latest`).
- **prompt** (`STRING`, multiline): The text prompt for generation.
- **temperature** (`FLOAT`, default: `0.9`, range: `0.0` to `2.0`): Controls randomness. Higher values are more creative, lower values are more deterministic.
- **top_p** (`FLOAT`, default: `1.0`, range: `0.0` to `1.0`): Nucleus sampling probability threshold (e.g., 0.95). 1.0 disables.
- **top_k** (`INT`, default: `1`, range: `1` to `100`): Top-K sampling threshold (consider probability of token).
- **max_output_tokens** (`INT`, default: `2048`, range: `1` to `8192`): Maximum number of tokens to generate in the response.
- **safety_harassment** (`COMBO`): Safety threshold for harassment content.
- **safety_hate_speech** (`COMBO`): Safety threshold for hate speech content.
- **safety_sexually_explicit** (`COMBO`): Safety threshold for sexually explicit content.
- **safety_dangerous_content** (`COMBO`): Safety threshold for dangerous content.
- **image_optional** (`IMAGE`, optional): Optional image input for multimodal models (e.g., `gemini-pro-vision`, `gemini-1.5-*`).

## Outputs

- **text** (`STRING`): The generated text response from the Gemini API.

## Usage

Connect a text prompt and optionally an image. Select the desired Gemini model and configure generation and safety settings. The node will output the generated text. Ensure your `GEMINI_API_KEY` is set as an environment variable.
