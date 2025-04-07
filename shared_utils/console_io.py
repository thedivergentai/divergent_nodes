"""
Provides utility functions for safe console input/output operations,
primarily handling potential encoding issues.
"""
import sys
from typing import Optional, TextIO
import logging

# Setup logger for this utility module
logger = logging.getLogger(__name__)
# Ensure handler is configured if root logger isn't set up
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Safe Print Function ---
def safe_print(text: str, file: Optional[TextIO] = None) -> None:
    """
    Prints text safely to the specified output stream, handling potential UnicodeEncodeErrors.

    Attempts to print directly. If a UnicodeEncodeError occurs, it tries to encode
    and decode using the stream's encoding (or UTF-8 fallback) with error replacement.
    As a final fallback, it prints the `repr()` of the text. Logs errors encountered
    during the process. Ensures output is flushed.

    Args:
        text (str): The string to print.
        file (Optional[TextIO]): The output stream (e.g., sys.stdout, sys.stderr).
                                 Defaults to sys.stdout if None.
    """
    output_stream: TextIO = file or sys.stdout
    try:
        # Attempt direct printing first
        print(text, file=output_stream, flush=True) # Added flush=True
    except UnicodeEncodeError:
        try:
            # Fallback: Encode using detected encoding or utf-8, replacing errors
            encoding: str = getattr(output_stream, 'encoding', None) or 'utf-8'
            encoded_text: str = str(text).encode(encoding, errors='replace').decode(encoding)
            print(encoded_text, file=output_stream, flush=True) # Added flush=True
            logger.debug(f"Successfully printed with encoding '{encoding}' after fallback.")
        except Exception as e:
            # Log error if fallback encoding fails
            logger.error(f"safe_print fallback failed: Could not print message due to encoding error: {e}", exc_info=True)
            # As a last resort, print the representation, which escapes non-ASCII
            try:
                print(repr(text), file=output_stream, flush=True)
            except Exception as repr_e:
                 logger.critical(f"safe_print failed completely, even repr failed: {repr_e}")
    except Exception as e:
        # Catch other potential print errors
        logger.error(f"safe_print encountered an unexpected error: {e}", exc_info=True)
