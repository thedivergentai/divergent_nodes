import os
import sys
import torch
import numpy as np
from PIL import Image
import io
import requests
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
                "gemma_model_url": ("STRING", {"default": "https://huggingface.co/bartowski/mlabonne_gemma-3-27b-it-abliterated-GGUF/resolve/main/mlabonne_gemma-3-27b-it-abliterated-Q4_K_M.gguf"}),
                "mmproj_url": ("STRING", {"default": "https://huggingface.co/bartowski/mlabonne_gemma-3-27b-it-abliterated-GGUF/resolve/main/mmproj-mlabonne_gemma-3-27b-it-abliterated-f32.gguf"}),
                "image": ("IMAGE",),  # ComfyUI Image object
                "prompt": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "process"
    CATEGORY = "Divergent Nodes 👽/VLM"  # Changed category to VLM

    def __init__(self):
        """
        Initializes the GemmaMultimodal node.
        """
        self.model = None
        self.mmproj = None
        self.model_path = None
        self.mmproj_path = None

    def download_file(self, repo_id, filename, local_filename):
        """Downloads a file from Hugging Face Hub using hf_hub_download."""
        try:
            print(f"Downloading {filename} from Hugging Face Hub...")  # Add print statement before download
            # Use hf_hub_download to download the file, which handles caching
            cached_filepath = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
            )
            print(f"Download of {filename} from Hugging Face Hub completed.")  # Add print statement after download
            # Copy the cached file to the filename expected by the node
            import shutil
            shutil.copy2(cached_filepath, local_filename) # Use copy2 to preserve metadata

            return local_filename
        except Exception as e:
            raise Exception(f"Error downloading {filename} from Hugging Face Hub: {e}")

    def load_model(self, gemma_model_url):
        """Loads the Gemma model from the given URL."""
        if self.model is None or self.model_path != gemma_model_url:
            try:
                model_filename = os.path.basename(gemma_model_url)
                # Download the model file from Hugging Face Hub and get the cached path.
                model_path = hf_hub_download(
                    repo_id="bartowski/mlabonne_gemma-3-27b-it-abliterated-GGUF",
                    filename=model_filename,
                )  # hf_hub_download returns the cached path

                print(f"Using cached model path: {model_path}")  # Log the cached model path

                try:
                    self.model = llama_cpp.Llama(
                        model_path=model_path,  # Load Llama model directly from the cached path
                        n_gpu_layers=32,
                        n_threads=8,
                        verbose=False,
                    )
                    self.model_path = gemma_model_url
                except Exception as e:
                    print(f"Detailed error loading Gemma model: {e}")  # Log detailed error message
                    raise Exception(f"Error loading Gemma model: {e}")
import os
import sys
import torch
import numpy as np
from PIL import Image
import io
import requests
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
                "gemma_model_url": ("STRING", {"default": "https://huggingface.co/bartowski/mlabonne_gemma-3-27b-it-abliterated-GGUF/resolve/main/mlabonne_gemma-3-27b-it-abliterated-Q4_K_M.gguf"}),
                "mmproj_url": ("STRING", {"default": "https://huggingface.co/bartowski/mlabonne_gemma-3-27b-it-abliterated-GGUF/resolve/main/mmproj-mlabonne_gemma-3-27b-it-abliterated-f32.gguf"}),
                "image": ("IMAGE",),  # ComfyUI Image object
                "prompt": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "process"
    CATEGORY = "Divergent Nodes 👽/VLM"  # Changed category to VLM

    def __init__(self):
        """
        Initializes the GemmaMultimodal node.
        """
        self.model = None
        self.mmproj = None
        self.model_path = None
        self.mmproj_path = None

    def download_file(self, repo_id, filename, local_filename):
        """Downloads a file from Hugging Face Hub using hf_hub_download."""
        try:
            print(f"Downloading {filename} from Hugging Face Hub...")  # Add print statement before download
            # Use hf_hub_download to download the file, which handles caching
            cached_filepath = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
            )
            print(f"Download of {filename} from Hugging Face Hub completed.")  # Add print statement after download
            # Copy the cached file to the filename expected by the node
            import shutil
            shutil.copy2(cached_filepath, local_filename) # Use copy2 to preserve metadata

            return local_filename
        except Exception as e:
            raise Exception(f"Error downloading {filename} from Hugging Face Hub: {e}")

    def load_model(self, gemma_model_url):
        """Loads the Gemma model from the given URL."""
        if self.model is None or self.model_path != gemma_model_url:
            try:
                model_filename = os.path.basename(gemma_model_url)
                # Download the model file from Hugging Face Hub and get the cached path.
                model_path = hf_hub_download(
                    repo_id="bartowski/mlabonne_gemma-3-27b-it-abliterated-GGUF",
                    filename=model_filename,
                ) # hf_hub_download returns the cached path

                print(f"Using cached model path: {model_path}") # Log the cached model path

                self.model = llama_cpp.Llama(
                    model_path=model_path,  # Load Llama model directly from the cached path
                    n_gpu_layers=32,
                    n_threads=8,
                    verbose=False,
                )
                self.model_path = gemma_model_url
            except Exception as e:
                raise Exception(f"Error loading Gemma model: {e}")

    def load_mmproj(self, mmproj_url):
        """Loads the MMPROJ file from the given URL."""
        if self.mmproj is None or self.mmproj_path != mmproj_url:
            try:
                mmproj_filename = os.path.basename(mmproj_url)
                mmproj_local_filename = mmproj_filename # Use the original filename as local filename
                # Download the mmproj file from Hugging Face Hub
                mmproj_path = self.download_file(
                    repo_id="bartowski/mlabonne_gemma-3-27b-it-abliterated-GGUF",
                    filename=mmproj_filename,
                    local_filename=mmproj_local_filename
                )
                self.mmproj = torch.load(mmproj_local_filename, map_location=torch.device('cpu')) # Use the local filename
                self.mmproj_path = mmproj_url
            except Exception as e:
                raise Exception(f"Error loading MMPROJ: {e}")

    def preprocess_image(self, image):
        """
        Preprocesses the ComfyUI image into a format suitable for the model.
        This function now handles the PIL conversion and normalization.

        Args:
            image: A ComfyUI Image object (numpy array of shape (batch, height, width, 3)).

        Returns:
            A torch tensor of shape (batch, 3, height, width), normalized and ready for the model.
        """
        # ComfyUI images are numpy arrays of shape (batch_size, height, width, 3)).
        # We need to convert them to PIL Images first, then to torch tensors.

        images = []
        for img in image:
            # Convert numpy array to PIL Image.  Handles different modes.
            img = Image.fromarray(img.astype(np.uint8), 'RGB')
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
