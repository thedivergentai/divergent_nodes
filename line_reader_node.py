import os
import io
import warnings
from enum import Enum

class IndexMode(Enum):
    INCREASE = "INCREASE"
    DECREASE = "DECREASE"
    FIXED = "FIXED"

class TextLineReader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_file_path": ("STRING", {"default": "prompt.txt"}),
                "index": ("INT", {"default": 0, "min": 0}),
                "index_mode": (list(IndexMode.__members__), {"default": "FIXED"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("text", "index")
    FUNCTION = "read_line"
    CATEGORY = "Divergent Nodes 👽/Text"

    def read_line(self, text_file_path, index, index_mode):
        index_int = int(index)
        index_mode_enum = IndexMode[index_mode]

        # Remove quotes from the path if they exist
        text_file_path = text_file_path.strip('"').strip("'")

        if not os.path.exists(text_file_path):
            print(f"Error: File not found at '{text_file_path}'")
            return "", index_int

        try:
            with io.open(text_file_path, mode='r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading file: {e}")
            return "", index_int

        num_lines = len(lines)

        if index_int < 0:
            warnings.warn("Warning: Index cannot be negative. Returning empty string.")
            return "", index_int

        if index_int >= num_lines:
            warnings.warn("Warning: Index out of bounds. Returning empty string.")
            return "", index_int

        text = lines[index_int].strip()

        if index_mode_enum == IndexMode.INCREASE:
            index_int += 1
        elif index_mode_enum == IndexMode.DECREASE:
            index_int -= 1

        return text, index_int

NODE_CLASS_MAPPINGS = {
    "Text Line Reader": TextLineReader
}
