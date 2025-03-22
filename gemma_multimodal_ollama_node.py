import os
import sys
import torch
import numpy as np
from PIL import Image
import io
from huggingface_hub import hf_hub_download

# Add llama.cpp to sys.path.  This might need adjustment based on your environment.
# Assuming llama-cpp-python is installed in a venv, and you want to use that venv.
# This is crucial for ComfyUI to find the llama-cpp-python module.
try:
    import llama_cpp
except ImportError:
    print("Error: llama-cpp-python is not installed.  Please install it,")
    print("       and ensure it's in your PYTHONPATH or virtual environment.")
    print("       For example: pip install llama-cpp-python")
    sys.exit(1)



class GemmaMultimodalOllama:
    """
    A custom ComfyUI node to integrate the Gemma-3-27b-it-abliterated model for multimodal tasks.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types for the node.
        """
        return {
            "required": {
                "gemma_model_url": ("STRING", {"default": "https://huggingface.co/bartowski/mlabonne_gemma-3-27b-it-abliterated-GGUF/resolve/main/mlabonne_gemma-3-27b-it-abliterated-Q5_K_L.gguf"}),
                "mmproj_url": ("STRING", {"default": "https://huggingface.co/bartowski/mlabonne_gemma-3-27b-it-abliterated-GGUF/resolve/main/mmproj-mlabonne_gemma-3-27b-it-abliterated-f32.gguf"}),
                "image": ("IMAGE",),  # ComfyUI Image object
                "prompt": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "process"
    CATEGORY = "Gemma"

    def __init__(self):
        """
        Initializes the GemmaMultimodal node.
        """
        self.model = None
        self.mmproj = None
        self.model_path = None
        self.mmproj_path = None

    def huggingface_download(self, repo_id, filename):
        """Downloads a file from Hugging Face Hub using the huggingface_hub library."""
        try:
            local_path = hf_hub_download(repo_id=repo_id, filename=filename, resume_download=True, cache_dir="huggingface_cache")
            return local_path
        except Exception as e:
            raise Exception(f"Error downloading {filename} from Hugging Face Hub: {e}")

    def load_model(self, gemma_model_url):
        """Loads the Gemma model from the given URL."""
        if self.model is None or self.model_path != gemma_model_url:
            try:
                # Extract repo_id and filename from the URL
                repo_id = "/".join(gemma_model_url.split("/")[3:5])  # e.g., bartowski/mlabonne_gemma-3-27b-it-abliterated-GGUF
                model_filename = gemma_model_url.split("/")[-1]  # e.g., mlabonne_gemma-3-27b-it-abliterated-Q5_K_L.gguf

                # Download the model file using huggingface_hub
                model_path = self.huggingface_download(repo_id, model_filename)

                self.model = llama_cpp.Llama(
                    model_path=model_path,
                    n_gpu_layers=32,  # Or however many layers you want to offload to the GPU
                    n_threads=8,  # Adjust based on your system
                    verbose=False,  # Suppress the verbose output. Useful for ComfyUI.
                )
                self.model_path = gemma_model_url
            except Exception as e:
                raise Exception(f"Error loading Gemma model: {e}")

    def load_mmproj(self, mmproj_url):
        """Loads the MMPROJ file from the given URL."""
        if self.mmproj is None or self.mmproj_path != mmproj_url:
            try:
                # Extract repo_id and filename from the URL
                repo_id = "/".join(mmproj_url.split("/")[3:5])  # e.g., bartowski/mlabonne_gemma-3-27b-it-abliterated-GGUF
                mmproj_filename = mmproj_url.split("/")[-1]  # e.g., mmproj-mlabonne_gemma-3-27b-it-abliterated-f32.gguf

                # Download the mmproj file using huggingface_hub
                mmproj_path = self.huggingface_download(repo_id, mmproj_filename)
                self.mmproj = torch.load(mmproj_path, map_location=torch.device('cpu'))
                self.mmproj_path = mmproj_url
            except Exception as e:
                raise Exception(f"Error loading MMPROJ: {e}")

    def preprocess_image(self, image):
        """
        Preprocesses the ComfyUI image into a format suitable for the model.
        This function now handles the PIL conversion and normalization.

        Args:
            image: A ComfyUI Image object (numpy array of shape (batch_size, height, width, 3)).

        Returns:
            A torch tensor of shape (batch, 3, height, width), normalized and ready for the model.
        """
        # ComfyUI images are numpy arrays of shape (batch_size, height, width, 3)).
        # We need to convert them to PIL Images first, then to torch tensors.

        images = []
        for img in image:
            # Convert numpy array to PIL Image.  Handles different modes.
            img = Image.fromarray(img.astype(np.uint8), 'RGB')
            img = convert_to_rgb(img)
            images.append(img)

        # Now, process the list of PIL images.
        processed_images = []
        for img in images:
            # Resize the image.  Use the same size as in your original notebook.
            img = img.resize((224, 224))
            # Convert to a PyTorch tensor and normalize.
            img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
            # Rearrange dimensions to (C, H, W).
            img_tensor = img_tensor.permute(2, 0, 1)
            processed_images.append(img_tensor)

        # Stack the individual image tensors to create a batch tensor.
        image_tensor = torch.stack(processed_images)
        return image_tensor

    def process(self, ollama_base_url, model_name, mmproj_file_url, image, prompt):
        """
        Processes the image and prompt using the Gemma model and MMPROJ.

        Args:
            ollama_base_url (str): Base URL of the Ollama server.
            model_name (str): Name of the Gemma model in Ollama.
            mmproj_file_url (str): URL to the MMPROJ file.
            image (torch.Tensor): ComfyUI Image object.
            prompt (str): Text prompt.

        Returns:
            str: The generated text response.
        """
        try:
            # Load the model and mmproj if they are not already loaded, or if the paths have changed.
            self.load_model(gemma_model_url)
            self.load_mmproj(mmproj_url)

            # Preprocess the image.
            image_tensor = self.preprocess_image(image)

            # Process the image with the mmproj.
            with torch.no_grad():
                image_features = self.mmproj.encode_image(image_tensor)
                image_features = image_features.float()

            # Prepare the prompt.  This is crucial to get right.  The original notebook
            # uses a specific format, and we must match it.
            prompt_text = f"<image>\n{prompt}"  # The \n was missing

            # Generate the response.  This part remains similar to the original.
            output = self.model.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": prompt_text,
                    },
                ],
                max_tokens=512,  # Or adjust as needed
                temperature=0.2,  # temperature
            )
            response = output["choices"][0]["message"]["content"]
            return (response,)

        except Exception as e:
            print(f"Error in GemmaMultimodal.process: {e}")  # Log the error
            return (f"Error: {e}",)  # Return the error message as a string.  This is important
            # so ComfyUI doesn't crash.
