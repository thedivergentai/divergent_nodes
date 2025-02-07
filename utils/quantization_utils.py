import torch

def apply_quantization(quantization_type="none"):
    """Applies quantization to the model."""

    if quantization_type.lower() == "nf4":
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            return bnb_config
        except ImportError:
            print("Error: bitsandbytes library not found. Please install it (pip install bitsandbytes).")
            return None

    elif quantization_type.lower() == "bf16":
        return None

    elif quantization_type.lower() == "none":
        return None
    else:
        print(f"Error: Unsupported quantization type: {quantization_type}")
        return None
