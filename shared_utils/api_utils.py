"""
API utilities for centralized request handling with retry logic and error formatting.
Provides helpers for making robust API calls.
"""
import requests
import time
import logging
from typing import Optional, Dict, Any, Callable

logger = logging.getLogger(__name__)


def make_api_request_with_retry(
    url: str,
    method: str = "GET",
    max_retries: int = 3,
    retry_delay: float = 2.0,
    timeout: int = 30,
    **kwargs
) -> requests.Response:
    """
    Make an API request with retry logic for transient failures.
    
    Args:
        url: API endpoint URL
        method: HTTP method (GET, POST, etc.)
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (exponential backoff)
        timeout: Request timeout in seconds
        **kwargs: Additional arguments passed to requests.request()
    
    Returns:
        Response object
    
    Raises:
        requests.RequestException: If all retries fail
    """
    for attempt in range(max_retries + 1):
        try:
            response = requests.request(
                method=method,
                url=url,
                timeout=timeout,
                **kwargs
            )
            response.raise_for_status()
            return response
            
        except requests.exceptions.Timeout as e:
            if attempt < max_retries:
                delay = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries + 1}). Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"Request failed after {max_retries + 1} attempts (timeout)")
                raise
                
        except requests.exceptions.ConnectionError as e:
            if attempt < max_retries:
                delay = retry_delay * (2 ** attempt)
                logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries + 1}). Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"Request failed after {max_retries + 1} attempts (connection error)")
                raise
                
        except requests.exceptions.HTTPError as e:
            # Don't retry on 4xx client errors (except 429 rate limit)
            if e.response.status_code == 429 and attempt < max_retries:
                delay = retry_delay * (2 ** attempt)
                logger.warning(f"Rate limited (attempt {attempt + 1}/{max_retries + 1}). Retrying in {delay}s...")
                time.sleep(delay)
            elif 400 <= e.response.status_code < 500:
                logger.error(f"Client error ({e.response.status_code}): {e}")
                raise
            elif attempt < max_retries:
                delay = retry_delay * (2 ** attempt)
                logger.warning(f"Server error ({e.response.status_code}, attempt {attempt + 1}/{max_retries + 1}). Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"Request failed after {max_retries + 1} attempts (HTTP error)")
                raise
                
        except Exception as e:
            logger.error(f"Unexpected error during API request: {e}", exc_info=True)
            raise


def handle_api_error(e: Exception, service_name: str = "API") -> str:
    """
    Format an API error into a user-friendly error message.
    
    Args:
        e: Exception that occurred
        service_name: Name of the service/API for context
    
    Returns:
        Formatted error message string
    """
    if isinstance(e, requests.exceptions.Timeout):
        return f"ERROR: {service_name} request timed out. Please check your connection and try again."
    
    elif isinstance(e, requests.exceptions.ConnectionError):
        return f"ERROR: Could not connect to {service_name}. Please check your internet connection."
    
    elif isinstance(e, requests.exceptions.HTTPError):
        status = getattr(e.response, 'status_code', 'Unknown')
        if status == 401:
            return f"ERROR: {service_name} authentication failed. Please check your API key."
        elif status == 403:
            return f"ERROR: {service_name} access forbidden. You may not have permission for this resource."
        elif status == 404:
            return f"ERROR: {service_name} endpoint not found (404)."
        elif status == 429:
            return f"ERROR: {service_name} rate limit exceeded. Please wait and try again."
        elif 500 <= status < 600:
            return f"ERROR: {service_name} server error ({status}). The service may be temporarily unavailable."
        else:
            return f"ERROR: {service_name} request failed ({status}): {str(e)}"
    
    else:
        return f"ERROR: {service_name} request failed: {str(e)}"


def download_file_with_progress(
    url: str,
    destination: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    chunk_size: int = 8192
) -> bool:
    """
    Download a file with optional progress reporting.
    
    Args:
        url: URL to download from
        destination: Local file path to save to
        progress_callback: Optional callback function(current_bytes, total_bytes)
        chunk_size: Chunk size for streaming download
    
    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total_size:
                        progress_callback(downloaded, total_size)
        
        logger.info(f"Successfully downloaded file to {destination}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download file from {url}: {e}", exc_info=True)
        return False
