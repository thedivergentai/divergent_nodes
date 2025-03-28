import os
import sys
import torch
import numpy as np
from PIL import Image
import io
import requests
from huggingface_hub import hf_hub_download
from vllm import LLM, SamplingParams


class GemmaChatHandler:
    def __init__(self, mmproj_path):
        self.mmproj_path = mmproj_path
        self.mmproj = torch.load(mmproj_path, map_location=torch.device('cpu'))

    def preprocess_image(self, image):
        if image is None:
            return None

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

    def format_prompt(self, prompt, image_features=None):
        if image_features is None:
            return f"<bos><start_of_turn>user\\n\\n{prompt}<end_of_turn><start_of_turn>model\\n"
        else:
            return f"<image>\\n<bos><start_of_turn>user\\n\\n{prompt}<end_of_turn><start_of_turn>model\\n"


class GemmaMultimodal:
    """
    A custom ComfyUI node to integrate the Gemma-3-27b-it-abliterated model for multimodal tasks using vLLM.
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
                "enable_cache": ("BOOLEAN", {"default": True}),
                "max_tokens": ("INT", {"default": 512}),
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
        self.cached_model_url = None
        self.cached_mmproj_url = None
        self.chat_handler = None

    def download_file(self, repo_id, filename, local_filename):
        """Downloads a file from Hugging Face Hub using hf_hub_download."""
        try:
            print(f"Downloading {filename} from Hugging Face Hub...")
            cached_filepath = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
            )
            print(f"Download of {filename} from Hugging Face Hub completed.")
            import shutil
            shutil.copy2(cached_filepath, local_filename)
            return local_filename
        except Exception as e:
            raise Exception(f"Error downloading {filename} from Hugging Face Hub: {e}")

    def load_model(self, gemma_model_url, mmproj_url, enable_cache):
        """Loads the Gemma model using vLLM."""
        if enable_cache and self.model is not None and self.cached_model_url == gemma_model_url:
            print("Using cached Gemma model.")
            return

        try:
            model_filename = os.path.basename(gemma_model_url)
            model_path = hf_hub_download(
                repo_id="bartowski/mlabonne_gemma-3-27b-it-abliterated-GGUF",
                filename=model_filename,
            )

            print(f"Using cached model path: {model_path}")

            self.model = LLM(model=model_path)
            self.model_path = gemma_model_url
            self.cached_model_url = gemma_model_url

            self.chat_handler = GemmaChatHandler(mmproj_url)
        except Exception as e:
            raise Exception(f"Error loading Gemma model with vLLM: {e}")

    def load_mmproj(self, mmproj_url, enable_cache):
        """Loads the MMPROJ file from the given URL."""
        pass  # No longer needed

    def process(self, gemma_model_url, mmproj_url, image, prompt, enable_cache, max_tokens):
        """Processes the image and prompt using the Gemma model and MMPROJ with vLLM."""
        try:
            # Load the model and mmproj if they are not already loaded, or if the paths have changed.
            self.load_model(gemma_model_url, mmproj_url, enable_cache)

            # Preprocess the image.
            image_tensor = self.chat_handler.preprocess_image(image)

            image_features = None
            if image_tensor is not None:
                # Process the image with the mmproj.
                with torch.no_grad():
                    image_features = self.chat_handler.mmproj.encode_image(image_tensor)
                    image_features = image_features.float()

            # Prepare the prompt.  This is crucial to get right.  The original notebook
            # uses a specific format, and we must match it.
            prompt_text = self.chat_handler.format_prompt(prompt, image_features)

            # Generate the response.  This part remains similar to the original.
            sampling_params = SamplingParams(temperature=0.2, top_p=1.0, max_tokens=max_tokens)

            if image is None:
                outputs = self.model.generate(prompt_text, sampling_params)
            else:
                outputs = self.model.generate(
                    prompt=prompt_text,
                    sampling_params=sampling_params,
                    multi_modal_data={"image": image}
                )

            response = outputs[0].outputs[0].text
            return (response,)

        except Exception as e:
            print(f"Error in GemmaMultimodal.process: {e}")
            return (f"Error: {e}",)
