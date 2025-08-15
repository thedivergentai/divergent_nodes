import logging

logger = logging.getLogger(__name__)

def ensure_utf8_friendly(text_input: str) -> str:
    """
    Ensures the input string is UTF-8 friendly by encoding and then
    decoding with error replacement.
    Args:
        text_input: The string to process.
    Returns:
        A UTF-8 friendly version of the string.
    """
    if not isinstance(text_input, str):
        text_input = str(text_input)
    try:
        # Encode to bytes using UTF-8, replacing errors, then decode back to string
        return text_input.encode('utf-8', errors='replace').decode('utf-8')
    except Exception as e:
        logger.error(f"Error during UTF-8 conversion for input '{text_input[:100]}...': {e}", exc_info=True)
        # Fallback: return original string if conversion fails catastrophically (should be rare with 'replace')
        return text_input
