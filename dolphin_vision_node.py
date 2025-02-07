import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import numpy as np
import os
from comfy.utils import load_torch_file
from utils.download_manager import download_model

class DolphinVision:
    def __init__(self):
        self.model_name = 'cognitivecomputations/dolphin-vision-7b'
        self.model_path = os.path.join('models', 'dolphin-vision-7b') # Assuming a 'models' directory
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.tokenizer = None

    def load_model(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path, exist_ok=True)
            # Download model components
            # Assuming the model files are in a format that can be downloaded and loaded directly
            # This is a placeholder, adjust the download URLs and file names as needed
            model_url = f"https://huggingface.co/{self.model_name}/resolve/main/config.json" # Example, replace with actual URLs
            download_model(model_url, os.path.join(self.model_path, "config.json"))
            model_url = f"https://huggingface.co/{self.model_name}/resolve/main/model.safetensors" # Example, replace with actual URLs
            download_model(model_url, os.path.join(self.model_path, "model.safetensors"))
            model_url = f"https://huggingface.co/{self.model_name}/resolve/main/tokenizer_config.json" # Example, replace with actual URLs
            download_model(model_url, os.path.join(self.model_path, "tokenizer_config.json"))
            model_url = f"https://huggingface.co/{self.model_name}/resolve/main/tokenizer.json" # Example, replace with actual URLs
            download_model(model_url, os.path.join(self.model_path, "tokenizer.json"))
            model_url = f"https://huggingface.co/{self.model_name}/resolve/main/generation_config.json" # Example, replace with actual URLs
            download_model(model_url, os.path.join(self.model_path, "generation_config.json"))
            model_url = f"https://huggingface.co/{self.model_name}/resolve/main/vocab.json" # Example, replace with actual URLs
            download_model(model_url, os.path.join(self.model_path, "vocab.json"))
            model_url = f"https://huggingface.co/{self.model_name}/resolve/main/merges.txt" # Example, replace with actual URLs
            download_model(model_url, os.path.join(self.model_path, "merges.txt"))
            model_url = f"https://huggingface.co/{self.model_name}/resolve/main/special_tokens_map.json" # Example, replace with actual URLs
            download_model(model_url, os.path.join(self.model_path, "special_tokens_map.json"))

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map='auto',
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            self.model.eval() # Set to evaluation mode
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.tokenizer = None

    def process_image(self, image):
        # Convert ComfyUI image tensor to PIL Image
        image = image.movedim(-1, -2) # CHW -> HWC
        image = image.squeeze() # Remove batch dimension
        image = image.cpu().numpy()
        image = (image * 255).astype(np.uint8) # Scale to 0-255
        image = Image.fromarray(image)
        return image

    def apply_chat_template(self, prompt):
        messages = [{"role": "user", "content": f'<image>\n{prompt}'}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text

    def __call__(self, image, prompt):
        if self.model is None or self.tokenizer is None:
            self.load_model()
            if self.model is None or self.tokenizer is None:
                return "Error: Model failed to load."

        try:
            image = self.process_image(image)
            text = self.apply_chat_template(prompt)

            text_chunks = [self.tokenizer(chunk).input_ids for chunk in text.split('<image>')]
            input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0).to(self.device)

            image_tensor = self.model.process_images([image], self.model.config).to(dtype=self.model.dtype)

            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                max_new_tokens=2048,
                use_cache=True
            )[0]

            generated_text = self.tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
            return generated_text

        except Exception as e:
            print(f"Error during generation: {e}")
            return f"Error: {e}"
