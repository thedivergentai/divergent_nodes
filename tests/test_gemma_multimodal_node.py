import sys
sys.path.append("D:/DivergentAI/divergent_nodes")
import torch
import numpy as np
from PIL import Image
from source.gemma_multimodal_node import GemmaMultimodal

# Load test image
try:
    image = Image.open("tests/ferrets.jpeg")
except FileNotFoundError:
    print("Error: ferrets.jpeg not found. Place a test image in the tests directory.")
    raise

# Create test prompt
prompt = "What are these animals?"

# Instantiate the node
node = GemmaMultimodal()

# Set dummy values for model paths
gemma_llm_url = "https://huggingface.co/bartowski/mlabonne_gemma-3-27b-it-abliterated-GGUF/resolve/main/mlabonne_gemma-3-27b-it-abliterated-Q4_K_M.gguf"
gemma_mmproj_url = "https://huggingface.co/bartowski/mlabonne_gemma-3-27b-it-abliterated-GGUF/resolve/main/mmproj-mlabonne_gemma-3-27b-it-abliterated-f32.gguf"


# Convert PIL Image to ComfyUI Image format (numpy array)
image = np.array(image.convert("RGB"))
image = torch.from_numpy(image).float()
image = image.unsqueeze(0)  # Add batch dimension

# Call the process method
try:
    output = node.process(gemma_llm_url, gemma_mmproj_url, image, prompt, max_tokens=50)

    # Check if the output is a string
    assert isinstance(output[0], str), f"Output is not a string: {output}"
    print("GemmaMultimodal node test: SUCCESS")

except Exception as e:
    print(f"GemmaMultimodal node test: FAIL - Exception: {e}")
    raise
