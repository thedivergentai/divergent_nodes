from .clip_token_counter import CLIPTokenCounter
from .utf8_encoder_node import UTF8EncoderNode
from .data_store_node import DataStoreNode
from .gemma_multimodal_node import GemmaMultimodal

NODE_CLASS_MAPPINGS = {
    "CLIPTokenCounter": CLIPTokenCounter,
    "UTF8EncoderNode": UTF8EncoderNode,
    "DataStoreNode": DataStoreNode,
    "GemmaMultimodal": GemmaMultimodal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTokenCounter": "CLIP Token Counter",
    "UTF8EncoderNode": "UTF8 Encoder",
    "DataStoreNode": "Data Store",
    "GemmaMultimodal": "Gemma Multimodal",
}
