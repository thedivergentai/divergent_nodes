import os
import sys
import torch
import numpy as np
from PIL import Image
import io
import requests

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



class GemmaMultimodal:
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

    def download_file(self, url, filename):
        """Downloads a file from a URL and saves it with a progress bar."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes

            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB
            progress_bar = None

            # Check if tqdm is installed, if not, don't use the progress bar
            try:
                from tqdm import tqdm
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            except ImportError:
                pass  # tqdm is not installed, so we'll just download without a progress bar

            with open(filename, 'wb') as file:
                for data in response.iter_content(block_size):
                    file.write(data)
                    if progress_bar:
                        progress_bar.update(len(data))

            if progress_bar:
                progress_bar.close()
            return filename
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error downloading {url}: {e}")
        except Exception as e:
            raise Exception(f"General error downloading {url}: {e}")

    def load_model(self, gemma_model_url):
        """Loads the Gemma model from the given URL."""
        if self.model is None or self.model_path != gemma_model_url:
            try:
                # Create a temporary file.  Important to create in a secure manner.
                # We use a deterministic filename based on the URL.
                model_filename = os.path.basename(gemma_model_url)
                # Download the model file.
                model_path = self.download_file(gemma_model_url, model_filename)

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
                # Create a temporary file. Use a deterministic filename.
                mmproj_filename = os.path.basename(mmproj_url)
                # Download the mmproj file
                mmproj_path = self.download_file(mmproj_url, mmproj_filename)
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
        # ComfyUI images are numpy arrays of shape (batch_size, height, width, 3).
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

    def process(self, gemma_model_url, mmproj_url, image, prompt):
        """
        Processes the image and prompt using the Gemma model and MMPROJ.

        Args:
            gemma_model_url (str): URL to the Gemma model file.
            mmproj_url (str): URL to the MMPROJ file.
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
