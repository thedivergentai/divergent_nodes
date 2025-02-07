import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .utils.quantization_utils import apply_quantization
import os
from PIL import Image
# The following imports are from the example and are needed
from deepseek_vl.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl.utils.io import load_pil_images


class DeepSeekVL2Node:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_quantization = None
        self.model_name = "deepseek-ai/deepseek-vl2"  # Base model name
        self.model_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.model_path = os.path.join(self.model_base_path, self.model_name.split('/')[-1])
        self.processor = None


    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Describe the image in detail."}),
                "quantization": (["bf16", "nf4"], {"default": "bf16"}),
                "model_variant": (["base", "small", "tiny"], {"default": "base"}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run_vlm"
    CATEGORY = "Divergent Nodes 👽/VLM"

    def load_model(self, quantization, model_variant):
        model_name_map = {
            "base": "deepseek-ai/deepseek-vl2",
            "small": "deepseek-ai/deepseek-vl2-small",
            "tiny": "deepseek-ai/deepseek-vl2-tiny",
        }
        selected_model_name = model_name_map.get(model_variant)
        if not selected_model_name:
            raise ValueError(f"Invalid model variant: {model_variant}")
        
        if (self.model is not None and self.tokenizer is not None and
            self.current_quantization == quantization and self.model_name == selected_model_name):
            return  # Model already loaded

        if self.model is not None or self.tokenizer is not None or self.processor is not None:
            self.model = None
            self.tokenizer = None
            self.processor = None

        try:
            self.model_name = selected_model_name
            self.processor = DeepseekVLV2Processor.from_pretrained(self.model_name, trust_remote_code=True)
            self.tokenizer = self.processor.tokenizer
            bnb_config = apply_quantization(None, quantization)

            if bnb_config is not None and quantization == "nf4":
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, quantization_config=bnb_config)
            else:  # Default to bf16
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)

            self.model.to(self.model.device) # Ensure model is on the correct device
            self.current_quantization = quantization

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def run_vlm(self, prompt, quantization, model_variant, image=None):
        try:
            self.load_model(quantization, model_variant)
        except Exception as e:
            return (f"Model loading failed: {e}",)

        if self.model is None or self.tokenizer is None or self.processor is None:
            return ("Model loading failed.",)

        if image is not None:
            # Convert the tensor image to PIL Image
            image = Image.fromarray(image.squeeze(0).mul(255).clamp(0, 255).byte().cpu().numpy())

            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image>\n{prompt}",
                    "images": [image],  # Pass the PIL Image directly
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

            try:
                # Prepare inputs using the processor
                prepare_inputs = self.processor(
                    conversations=conversation,
                    images=[image],
                    force_batchify=True,
                    system_prompt=""
                ).to(self.model.device)

                # Run image encoder
                inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

                # Generate response
                outputs = self.model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=512,
                    do_sample=False,
                    use_cache=True
                )
                answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
                return (f"{prepare_inputs['sft_format'][0]} {answer}",)

            except Exception as e:
                return (f"Error during image generation: {e}",)
        else:
            # Text-only generation (same as before, but using self.tokenizer)
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                generated_ids = self.model.generate(**inputs, max_new_tokens=100)
                generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return (generated_text,)
            except Exception as e:
                print(f"Error during generation (text-only): {e}")
                return (f"Error during generation: {e}",)
