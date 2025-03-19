class UTF8EncoderNode:
    NODE_DISPLAY_NAME = "UTF8 Encoder"
    RETURN_TYPES = ("STRING",)
    CATEGORY = "Divergent Nodes 👽/Text"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"defaultInput": True}),
            }
        }

    FUNCTION = "encode_utf8"

    def encode_utf8(self, text):
        """
        Encodes the input text to UTF-8 format.

        Args:
            text (str): The text to encode.

        Returns:
            tuple: A tuple containing the UTF-8 encoded text (str).
        """
        encoded_text = text.encode('utf-8').decode('utf-8', 'replace')
        return (encoded_text,)
