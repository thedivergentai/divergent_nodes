"""
Prompt manipulation utilities for text processing in ComfyUI nodes.
Provides helpers for combining prompts, replacing placeholders, and text manipulation.
"""
import re
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def parse_prompt_placeholders(text: str, replacements: Dict[str, str]) -> str:
    """
    Replace {key} placeholders in text with values from replacements dict.
    
    Args:
        text: Template string with {key} placeholders
        replacements: Dictionary mapping keys to replacement values
    
    Returns:
        Text with placeholders replaced
    
    Example:
        >>> parse_prompt_placeholders("A {color} {animal}", {"color": "red", "animal": "fox"})
        "A red fox"
    """
    if not text:
        return text
    
    result = text
    for key, value in replacements.items():
        placeholder = f"{{{key}}}"
        result = result.replace(placeholder, str(value))
    
    # Warn about unreplaced placeholders
    remaining = re.findall(r'\{([^}]+)\}', result)
    if remaining:
        logger.warning(f"Unreplaced placeholders found: {remaining}")
    
    return result


def combine_prompts(prompts: List[str], separator: str = ", ") -> str:
    """
    Combine multiple prompts into a single string.
    
    Args:
        prompts: List of prompt strings
        separator: String to join prompts with
    
    Returns:
        Combined prompt string
    """
    if not prompts:
        return ""
    
    # Filter out empty/None prompts
    valid_prompts = [p.strip() for p in prompts if p and p.strip()]
    
    return separator.join(valid_prompts)


def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """
    Extract likely keywords from text (basic implementation).
    Splits on common separators and filters short words.
    
    Args:
        text: Input text
        min_length: Minimum word length to consider
    
    Returns:
        List of extracted keywords
    """
    if not text:
        return []
    
    # Split on common separators
    words = re.split(r'[,;.\n\r]+', text)
    
    # Clean and filter
    keywords = []
    for word in words:
        cleaned = word.strip()
        if len(cleaned) >= min_length:
            keywords.append(cleaned)
    
    return keywords


def truncate_prompt(text: str, max_tokens: int = 77, tokenizer_fn=None) -> str:
    """
    Truncate a prompt to a maximum number of tokens.
    
    Args:
        text: Prompt text
        max_tokens: Maximum number of tokens
        tokenizer_fn: Optional tokenizer function, defaults to word count
    
    Returns:
        Truncated prompt
    """
    if not text:
        return text
    
    if tokenizer_fn:
        # Use custom tokenizer if provided
        tokens = tokenizer_fn(text)
        if len(tokens) <= max_tokens:
            return text
        return tokenizer_fn.decode(tokens[:max_tokens])
    else:
        # Fallback: use word count
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[:max_tokens]) + "..."


def clean_prompt(text: str) -> str:
    """
    Clean up a prompt by removing extra whitespace and normalizing separators.
    
    Args:
        text: Prompt text to clean
    
    Returns:
        Cleaned prompt
    """
    if not text:
        return text
    
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', text)
    
    # Normalize multiple commas/separators
    cleaned = re.sub(r',\s*,+', ',', cleaned)
    
    # Remove leading/trailing whitespace and commas
    cleaned = cleaned.strip().strip(',').strip()
    
    return cleaned
