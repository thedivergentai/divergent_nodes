import os
import numpy as np
from PIL import Image
import json
import requests
from .utils.download_manager import download_file, clone_repository  # Import the download function

def convert_to_rgb(image):
    if image.mode in ("L", "LA"):
        return image.convert("RGB")
    return image

class GemmaMultimodalOllama:
    """
    A custom ComfyUI node to integrate the Gemma model via Ollama for multimodal tasks.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types for the node.
        """
        return {
            "required": {
                "ollama_base_url": ("STRING", {"default": "http://localhost:11434"}),  # Ollama API base URL
                "model_name": ("STRING", {"default": "hf.co/bartowski/mlabonne_gemma-3-27b-it-abliterated-GGUF:IQ2_S"}),  # Name of the model in Ollama
                "mmproj_file_url": ("STRING", {"default": "https://huggingface.co/bartowski/mlabonne_gemma-3-27b-it-abliterated-GGUF/blob/main/mmproj-mlabonne_gemma-3-27b-it-abliterated-f32.gguf"}), # URL for the mmproj file
                "image": ("IMAGE",),  # ComfyUI Image object
                "prompt": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "process"
    CATEGORY = "Divergent Nodes 👽/Gemma"

    def __init__(self):
        """
        Initializes the GemmaMultimodalOllama node.
        """
        self.ollama_base_url = ""
        self.model_name = ""

    def preprocess_image(self, image):
        """
        Preprocesses the ComfyUI image into a base64 encoded string, suitable for Ollama.

        Args:
            image: A ComfyUI Image object (numpy array of shape (batch, height, width, 3)).

        Returns:
            list: A list of base64 encoded image strings.
        """
        import base64
        images = []
        for img in image:
            # Convert numpy array to PIL Image
            img = Image.fromarray(img.astype(np.uint8), 'RGB')
            img = convert_to_rgb(img)

            # Resize the image.  Important for consistent input to the model.
            img = img.resize((224, 224))

            # Convert PIL Image to bytes
            img_bytes = img.tobytes()

            # Encode the bytes as base64
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            images.append(img_base64)
        return images

    def process(self, ollama_base_url, model_name, mmproj_file_url, image, prompt):
        """
        Processes the image and prompt using the Gemma model via Ollama.

        Args:
            ollama_base_url (str): Base URL of the Ollama server.
            model_name (str): Name of the Gemma model in Ollama.
            mmproj_file_url (str): URL for the mmproj file.
            image (torch.Tensor): ComfyUI Image object.
            prompt (str): Text prompt.

        Returns:
            str: The generated text response.
        """
        try:
            # Download mmproj file
            mmproj_filename = "mmproj-mlabonne_gemma-3-27b-it-abliterated-f32.gguf"
            mmproj_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), mmproj_filename)
            if not os.path.exists(mmproj_file_path):
                print(f"Downloading mmproj file to {mmproj_file_path}")
                download_file(mmproj_file_url, mmproj_file_path)
            else:
                print(f"MMPROJ file already exists: {mmproj_file_path}")

            # Check if the model exists in Ollama
            try:
                response = requests.post(f"{ollama_base_url}/api/pull", json={"name": model_name, "stream": False})
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                error_message = f"Model '{model_name}' not found in Ollama. Please ensure it is installed. It may need to be pulled using the command: ollama run https://huggingface.co/bartowski/mlabonne_gemma-3-27b-it-abliterated-GGUF/blob/main/mlabonne_gemma-3-27b-it-abliterated-IQ2_S.gguf. Error: {e}"
                print(error_message)
                return (error_message,)

            # Preprocess the image to base64.
            base64_images = self.preprocess_image(image)

            # Construct the Ollama API request.  Ollama expects a list of messages.
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                    "images": base64_images  # Pass the list of base64 images
                }
            ]
            data = {
                "model": model_name,
                "messages": messages,
                "stream": False, # Get the full response at once
            }

            # Send the request to the Ollama API.
            response = requests.post(f"{ollama_base_url}/api/chat", json=data)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Parse the JSON response.
            result = response.json()
            # Extract the response.  Handle the case where the 'message' field might be missing.
            if 'message' in result and 'content' in result['message']:
                response_text = result["message"]["content"]
            else:
                response_text = "Error: No response content found in Ollama output."

            return (response_text,)

        except requests.exceptions.RequestException as e:
            error_message = f"Error communicating with Ollama: {e}"
            print(error_message)
            return (error_message,)
        except json.JSONDecodeError as e:
            error_message = f"Error decoding Ollama response: {e}.  Response text: {response.text}"
            print(error_message)
            return (error_message,)
        except Exception as e:
            error_message = f"Error in GemmaMultimodalOllama.process: {e}"
            print(error_message)
            return (error_message,)
