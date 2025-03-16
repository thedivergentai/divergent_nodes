from .clip_token_counter import CLIPTokenCounter
from .utf8_encoder_node import UTF8EncoderNode

NODE_CLASS_MAPPINGS = {
    "CLIPTokenCounter": CLIPTokenCounter,
    "UTF8EncoderNode": UTF8EncoderNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTokenCounter": "CLIP Token Counter",
    "UTF8EncoderNode": "UTF8 Encoder"
}
