# KoboldCPP API Connector

This node connects to an already running KoboldCpp instance via its API to send generation requests (text and optional image). It does NOT launch or manage the KoboldCpp process itself.

## Parameters

- **api_url** (`STRING`, default: `http://127.0.0.1:5001`): The base URL of the running KoboldCpp API.
- **prompt** (`STRING`, multiline): The text prompt for generation.
- **max_length** (`INT`, default: `512`, range: `1` to `16384`): Maximum number of tokens to generate.
- **temperature** (`FLOAT`, default: `0.7`, range: `0.0` to `2.0`): Sampling temperature. Higher values lead to more random outputs.
- **top_p** (`FLOAT`, default: `0.92`, range: `0.0` to `1.0`): Nucleus sampling probability.
- **top_k** (`INT`, default: `0`, range: `0` to `1000`): Top-K sampling (0 disables).
- **rep_pen** (`FLOAT`, default: `1.1`, range: `0.0` to `3.0`): Repetition penalty. Higher values discourage repetition.
- **image_optional** (`IMAGE`, optional): Optional image input for multimodal models.
- **stop_sequence** (`STRING`, optional, multiline): Comma or newline-separated strings that will stop the generation when encountered.

## Outputs

- **text** (`STRING`): The generated text response from the KoboldCpp API, or an error message if the request fails.

## Usage

Ensure your KoboldCpp instance is running and accessible at the specified `api_url`. Provide a `prompt` and configure the generation parameters. Optionally, connect an image for multimodal models or specify `stop_sequence` to control generation length.
