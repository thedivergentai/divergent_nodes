from ..clip_token_counter import CLIPTokenCounter
from ..utf8_encoder_node import UTF8EncoderNode
from ..data_store_node import DataStoreNode

NODE_CLASS_MAPPINGS = {
    "CLIPTokenCounter": CLIPTokenCounter,
    "UTF8EncoderNode": UTF8EncoderNode,
    "DataStoreNode": DataStoreNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTokenCounter": "CLIP Token Counter",
    "UTF8EncoderNode": "UTF8 Encoder",
    "DataStoreNode": "Data Store"
}
