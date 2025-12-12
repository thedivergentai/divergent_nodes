"""
Text utility nodes for Divergent Nodes.
"""
from .prompt_combiner_node import PromptCombiner
from .string_list_node import StringToList
from .preview_text_node import PreviewText

NODE_CLASS_MAPPINGS = {
    "PromptCombiner": PromptCombiner,
    "StringToList": StringToList,
    "PreviewText": PreviewText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptCombiner": "Prompt Combiner",
    "StringToList": "String to List",
    "PreviewText": "Preview Text",
}
