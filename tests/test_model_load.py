import os
import sys
from huggingface_hub import hf_hub_download
import llama_cpp

model_name = "mlabonne_gemma-3-27b-it-abliterated-Q4_K_M.gguf"
repo_id = "bartowski/mlabonne_gemma-3-27b-it-abliterated-GGUF"

try:
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=model_name,
    )

    print(f"Model downloaded to: {model_path}")

    # Print llama_cpp version
    print(f"llama_cpp version: {llama_cpp.__version__}")

    # Try loading the model with llama_cpp
    try:
        llm = llama_cpp.Llama(model_path=model_path, n_gpu_layers=1, n_ctx=2048, verbose=False)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model with llama_cpp: {e}")

except Exception as e:
    print(f"Error downloading or processing model: {e}")
