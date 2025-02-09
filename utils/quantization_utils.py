from functools import lru_cache
from transformers import AutoModelForCausalLM
import torch
import bitsandbytes as bnb

@lru_cache(maxsize=4)  # Limit cache size to 4 models
def quantize_model(model_name, quantization_method="bitsandbytes", device="cuda", bits=8):
    """
    Quantizes a model using the specified method.  Currently only supports the "bitsandbytes" method.

    Args:
        model_name (str): Hugging Face Hub ID or path to the original model.
        quantization_method (str): The quantization method to use (currently only "bitsandbytes" is supported).
        device (str): "cpu" or "cuda".
        bits (int): Number of bits for quantization (4 or 8).

    Returns:
        transformers.PreTrainedModel: The quantized model, or None if an error occurred.
    
    Raises:
        Exception: For any unexpected errors during model loading or quantization.
    """
    cache_key = f"{model_name}_{quantization_method}_{device}_{bits}"
    print(f"Quantizing model {model_name} using {quantization_method} on {device} with {bits} bits...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        if quantization_method == "bitsandbytes":
            model = _apply_bitsandbytes_quantization(model, device, bits)
            if model is None:
                return None

        return model

    except Exception as e:
        print(f"An unexpected error occurred during quantization: {e}")
        return None


def _apply_bitsandbytes_quantization(model, device, bits):
    """
    Applies bitsandbytes quantization to the linear layers of the model.

    Args:
        model (transformers.PreTrainedModel): The model to quantize.
        device (str): "cpu" or "cuda".
        bits (int): Number of bits for quantization (4 or 8).

    Returns:
        transformers.PreTrainedModel: The quantized model, or None if an error occurred.

    Raises:
        ValueError: If the `bits` value is not 4 or 8.
        Exception: For any errors during quantization of individual layers.
    """
    try:
        if bits == 8:
            from bitsandbytes.nn import Linear8bitLt
            linear_layer_class = Linear8bitLt
        elif bits == 4:
            from bitsandbytes.nn import Linear4bit
            linear_layer_class = Linear4bit
        else:
            raise ValueError("Invalid bits value. Must be 4 or 8 for bitsandbytes.")

        model = model.to(torch.float16)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                try:
                    model.get_submodule(name).weight.requires_grad = False
                    model._modules[name] = linear_layer_class(
                        module.in_features,
                        module.out_features,
                        module.bias is not None,
                        **(
                            {"compute_dtype": torch.float16, "compress_statistics": True, "quant_type": "nf4"}
                            if bits == 4 else {"has_fp16_weights": True}
                        ),
                        device=device
                    )
                except Exception as e:
                    print(f"Error quantizing layer {name}: {e}")
                    return None  # Return None if quantization fails
        return model

    except Exception as e:
        print(f"Error applying bitsandbytes quantization: {e}")
        return None


def unload_model(model):
    """
    Unloads a model from memory and clears the CUDA cache.

    Args:
        model: The model to unload.

    Raises:
        Exception: For any errors during model deletion or cache clearing.
    """
    try:
        if model is not None:
            del model
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error unloading model: {e}")
