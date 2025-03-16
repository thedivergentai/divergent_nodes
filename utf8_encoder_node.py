class UTF8EncoderNode:
    NODE_DISPLAY_NAME = "UTF8 Encoder"
    RETURN_TYPES = ("STRING",)
    INPUT_TYPES = {
        "text": ("STRING",),
    }
    CATEGORY = "Divergent Nodes 👽/Text"

    def encode_utf8(self, text):
        """
        Encodes the input text to UTF-8 format.

        Args:
            text (str): The text to encode.

        Returns:
            tuple: A tuple containing the UTF-8 encoded text (str).
        """
        encoded_text = text.encode('utf-8').decode('utf-8', 'ignore')
        return (encoded_text,)
