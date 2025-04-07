"""
Manages the lifecycle (launch, caching, termination) of KoboldCpp subprocesses
and handles API interactions for the KoboldCpp Launcher Node.

Uses a global cache to reuse running KoboldCpp instances with the same
configuration, reducing startup time. Includes graceful termination and
automatic cleanup on exit.
"""
import subprocess
import os
import sys
import time
import shlex
import json
import socket
# import threading # Lock is now imported from process_cache
# import atexit # Cleanup is now handled in process_cache
import requests
from typing import Optional, Dict, Any, Tuple, List, Union, TypeAlias
import logging

# Import cache and termination logic from the new module
from .process_cache import (
    koboldcpp_processes_cache,
    cache_lock,
    terminate_process,
    CacheKeyT, # Import type aliases if needed here
    CacheEntryT,
    PopenObject # Import type aliases if needed here
)

# Setup logger for this module
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Type Aliases ---
# PopenObject, CacheKeyT, CacheEntryT are now imported from process_cache

# --- Process Cache ---
# Global cache and lock are now imported from process_cache

# --- Helper Functions for Process Management ---

def find_free_port() -> int:
    """
    Finds an available network port by binding to port 0 and letting the OS choose.

    Returns:
        int: An available port number.

    Raises:
        OSError: If binding to find a free port fails.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0)) # Bind to port 0 to let the OS pick a free port
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port: int = s.getsockname()[1] # Get the assigned port
            logger.debug(f"Found free port: {port}")
            return port
    except OSError as e:
        logger.error(f"Could not find a free port: {e}", exc_info=True)
        raise OSError("Failed to find a free network port for KoboldCpp.") from e

def check_api_ready(port: int, timeout: int = 60) -> bool:
    """
    Checks if the KoboldCpp API server is responding on the given port.

    Polls the `/api/extra/version` endpoint until it responds successfully
    or the timeout is reached.

    Args:
        port (int): The port number where the KoboldCpp API is expected.
        timeout (int): Maximum time in seconds to wait for the API (default: 60).

    Returns:
        bool: True if the API responds successfully within the timeout, False otherwise.
    """
    start_time: float = time.time()
    # Use a lightweight endpoint like /api/extra/version for the check
    # This endpoint exists in recent KoboldCpp versions and is less resource-intensive
    # than /api/v1/model, which might trigger model loading.
    url: str = f"http://127.0.0.1:{port}/api/extra/version"
    logger.debug(f"Checking API readiness at {url} (timeout={timeout}s)")

    while time.time() - start_time < timeout:
        try:
            # Use a short timeout for individual requests to avoid hanging here
            response = requests.get(url, timeout=2)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            # Check if response content seems valid (optional, but good practice)
            # content = response.json() # Assuming it returns JSON
            # if content.get("result"): # Check for expected key
            logger.info(f"API on port {port} is ready (responded to {url}).")
            return True
        except requests.exceptions.ConnectionError:
            # Expected error if the server isn't up yet, wait and retry
            logger.debug(f"API on port {port} not ready yet (Connection Error). Retrying...")
        except requests.exceptions.Timeout:
            # Log timeouts during check, might indicate slow startup or network issue
            logger.debug(f"API check timed out on port {port}. Retrying...")
        except requests.exceptions.RequestException as e:
            # Log other request-related errors (like HTTPError from raise_for_status)
            logger.warning(f"Error checking API readiness on port {port}: {e}. Retrying...")
        except Exception as e:
            # Log other unexpected errors during check
            logger.error(f"Unexpected error checking API readiness on port {port}: {e}", exc_info=True)
            # Depending on the error, might want to break or continue retrying

        time.sleep(1.0) # Wait a bit longer before retrying

    logger.error(f"API on port {port} did not become ready within the {timeout} second timeout.")
    return False

# terminate_process function is now defined in process_cache.py
# cleanup_koboldcpp_processes function is now defined in process_cache.py
# atexit registration is now handled in process_cache.py

# --- API Call Helper ---

def _call_kobold_api(port: int, payload: Dict[str, Any], cache_key: CacheKeyT) -> str:
    """
    Sends a generation request to the running KoboldCpp API endpoint.

    Handles common request errors (timeouts, connection errors, bad status codes)
    and attempts to parse the generated text from the response JSON.
    If a connection error occurs for a cached process that has died, it attempts
    to remove the dead process from the cache.

    Args:
        port (int): The port number the KoboldCpp API is running on.
        payload (Dict[str, Any]): The JSON payload for the `/api/v1/generate` endpoint.
        cache_key (CacheKeyT): The cache key associated with the process, used for
                               cleanup if a connection error occurs and the process died.

    Returns:
        str: The generated text string on success, or an error message prefixed
             with "ERROR:" on failure.
    """
    api_url: str = f"http://127.0.0.1:{port}/api/v1/generate"
    logger.info(f"Sending API request to {api_url}")
    logger.debug(f"API Payload: {json.dumps(payload, indent=2)}") # Log payload for debugging

    error_prefix = "ERROR:"
    default_error_msg = f"{error_prefix} API call failed to {api_url}."

    try:
        # Set a reasonable timeout for the API call (e.g., 5 minutes)
        response = requests.post(api_url, json=payload, timeout=300)
        response.raise_for_status() # Check for HTTP errors (4xx, 5xx)

        response_json = response.json()
        logger.debug(f"API Response JSON: {json.dumps(response_json, indent=2)}")

        # Extract text from the expected response structure
        results = response_json.get("results")
        if results and isinstance(results, list) and len(results) > 0:
            generated_text = results[0].get("text")
            if generated_text is not None:
                logger.info("API call successful, received generated text.")
                return str(generated_text) # Ensure it's a string
            else:
                logger.warning(f"API response missing 'text' field in results[0]: {response_json}")
                return f"{error_prefix} 'text' field missing in API response results."
        else:
            logger.warning(f"API response missing 'results' list or results are empty: {response_json}")
            return f"{error_prefix} Invalid 'results' structure in API response."

    except requests.exceptions.Timeout:
        logger.error(f"API request timed out after 300 seconds to {api_url}.")
        return f"{error_prefix} API request timed out ({api_url})."
    except requests.exceptions.ConnectionError as e:
        logger.error(f"API connection failed for {api_url}: {e}", exc_info=True)
        # Check if the corresponding cached process died and remove it
        with cache_lock: # Use imported lock
            cached_entry = koboldcpp_processes_cache.get(cache_key) # Use imported cache
            # Check if entry exists, has a process, and the process has terminated
            if cached_entry and isinstance(cached_entry.get("process"), subprocess.Popen) and cached_entry["process"].poll() is not None:
                pid_str = f"(PID: {cached_entry['process'].pid})" if hasattr(cached_entry['process'], 'pid') and cached_entry['process'].pid is not None else "(PID unknown)"
                logger.warning(f"API connection failed for port {port}. Associated process {pid_str} has terminated. Removing from cache.")
                koboldcpp_processes_cache.pop(cache_key, None) # Use imported cache
                # No need to call terminate_process here as it's already dead
            elif cached_entry:
                 logger.warning(f"API connection failed for port {port}, but cached process seems alive or invalid. Check KoboldCpp instance.")
            else:
                 logger.warning(f"API connection failed for port {port}, but no corresponding cache entry found.")
        return f"{error_prefix} API connection failed: {e}"
    except requests.exceptions.RequestException as e:
        # Handles other request errors like HTTPError from raise_for_status
        logger.error(f"API request failed for {api_url}: {e}", exc_info=True)
        return f"{error_prefix} API request failed: {e}"
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response from {api_url}: {e}", exc_info=True)
        logger.debug(f"Raw API response content: {response.text}")
        return f"{error_prefix} Failed to parse JSON response from API."
    except Exception as e:
        # Catch any other unexpected errors during the API call or parsing
        logger.error(f"Unexpected error during API call to {api_url}: {e}", exc_info=True)
        return f"{error_prefix} Unexpected API error: {e}"

# --- Process Launch and Cache Management Helper ---

def _build_kobold_command(setup_params: Dict[str, Any], port: int) -> List[str]:
    """
    Builds the command list (arguments) for launching the KoboldCpp executable.

    Constructs the command based on validated setup parameters provided by the node.

    Args:
        setup_params (Dict[str, Any]): Dictionary containing validated setup parameters
                                       (paths, settings like context size, GPU layers, etc.).
        port (int): The dynamically assigned free port number for this instance.

    Returns:
        List[str]: A list of strings representing the command and its arguments,
                   suitable for `subprocess.Popen`.

    Raises:
        ValueError: If `extra_cli_args` contains invalid syntax that `shlex.split` cannot parse.
        KeyError: If required keys like `koboldcpp_path` or `model_path` are missing
                  (should be caught by upstream validation, but good practice).
    """
    try:
        # --- Base Command ---
        command: List[str] = [
            str(setup_params["koboldcpp_path"]), # Ensure path is string
            "--model", str(setup_params["model_path"]),
            "--port", str(port),
            "--contextsize", str(setup_params["context_size"]),
            "--gpulayers", str(setup_params["n_gpu_layers"]),
            "--quiet" # Keep KoboldCpp's own console output minimal by default
        ]
        logger.debug(f"Base KoboldCpp command: {' '.join(command)}")

        # --- Optional Arguments ---
        # Multimodal Projector
        if mmproj := setup_params.get("mmproj_path"):
            command.extend(["--mmproj", str(mmproj)])
            logger.debug("Added --mmproj argument.")

        # GPU Acceleration
        gpu_accel = setup_params.get("gpu_acceleration", "None")
        if gpu_accel == "CuBLAS":
            command.append("--usecublas")
            logger.debug("Added --usecublas argument.")
        elif gpu_accel == "CLBlast":
            # Assuming default devices 0, 0 - might need configuration later
            command.extend(["--useclblast", "0", "0"])
            logger.debug("Added --useclblast argument.")
        elif gpu_accel == "Vulkan":
            command.append("--usevulkan")
            logger.debug("Added --usevulkan argument.")

        # Memory/Performance Flags
        if setup_params.get("use_mmap"):
            command.append("--usemmap")
            logger.debug("Added --usemmap argument.")
        if setup_params.get("use_mlock"):
            command.append("--usemlock")
            logger.debug("Added --usemlock argument.")
        if setup_params.get("flash_attention"):
            command.append("--flashattention")
            logger.debug("Added --flashattention argument.")

        # Quantization
        if (quant_kv := setup_params.get("quant_kv", 0)) > 0:
            command.extend(["--quantkv", str(quant_kv)])
            logger.debug(f"Added --quantkv {quant_kv} argument.")

        # Threads
        if (threads := setup_params.get("threads")) is not None and threads > 0:
            command.extend(["--threads", str(threads)])
            logger.debug(f"Added --threads {threads} argument.")

        # Extra CLI Arguments (handle carefully)
        if extra_args := setup_params.get("extra_cli_args"):
            try:
                # Use shlex.split for safe parsing of command-line strings
                parsed_extra_args = shlex.split(str(extra_args))
                command.extend(parsed_extra_args)
                logger.debug(f"Added extra CLI arguments: {parsed_extra_args}")
            except Exception as e:
                logger.error(f"Failed to parse extra_cli_args '{extra_args}': {e}", exc_info=True)
                raise ValueError(f"Could not parse Extra CLI Arguments: '{extra_args}'. Error: {e}") from e

        logger.debug(f"Final KoboldCpp command list: {command}")
        return command

    except KeyError as e:
        logger.error(f"Missing required setup parameter '{e}' for building KoboldCpp command.")
        raise KeyError(f"Missing required setup parameter '{e}'") from e
    except Exception as e:
        logger.error(f"Unexpected error building KoboldCpp command: {e}", exc_info=True)
        raise RuntimeError("Failed to build KoboldCpp launch command.") from e


def _launch_kobold_process(command: List[str], port: int) -> PopenObject:
    """
    Launches the KoboldCpp subprocess using the constructed command list.

    Redirects stdout and stderr to pipes for potential debugging.
    Uses CREATE_NO_WINDOW flag on Windows to prevent console popup.

    Args:
        command (List[str]): The fully constructed command list including executable and arguments.
        port (int): The port number assigned, used primarily for logging context.

    Returns:
        PopenObject: The Popen object representing the newly launched subprocess.

    Raises:
        FileNotFoundError: If the KoboldCpp executable specified in `command[0]` is not found.
        OSError: If there are OS-level errors during process creation (e.g., permissions).
        RuntimeError: For other unexpected errors during process launch.
    """
    logger.info(f"Launching KoboldCpp process on port {port}...")
    # Log the command safely for debugging, handling potential spaces in paths
    # Note: This might not be perfectly reconstructible as a shell command if args have complex quoting.
    logger.debug(f"Launch command: {' '.join(shlex.quote(arg) for arg in command)}")
    try:
        # Use CREATE_NO_WINDOW on Windows to prevent the console window from appearing.
        # This flag is ignored on non-Windows platforms.
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,    # Capture stdout
            stderr=subprocess.PIPE,    # Capture stderr
            text=True,                 # Decode stdout/stderr as text
            encoding='utf-8',          # Specify encoding
            errors='replace',          # Handle potential decoding errors
            creationflags=creationflags, # Platform-specific flags
            shell=False                # IMPORTANT: Never use shell=True with untrusted input
        )
        logger.info(f"KoboldCpp process launched successfully (PID: {process.pid}) on port {port}.")
        return process
    except FileNotFoundError as e:
        # Specific error if the executable path is wrong
        logger.error(f"KoboldCpp executable not found at '{command[0]}'. Please check the path.")
        # Re-raise with a more user-friendly message if needed, or just the original
        raise FileNotFoundError(f"KoboldCpp executable not found at '{command[0]}'. Check path.") from e
    except OSError as e:
        # Catch other OS errors like permission denied
        logger.error(f"OS error launching KoboldCpp process: {e}", exc_info=True)
        raise OSError(f"OS error launching KoboldCpp: {e}") from e
    except Exception as e:
        # Catch any other unexpected errors during Popen
        logger.error(f"Unexpected error launching KoboldCpp process: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error launching KoboldCpp process: {e}") from e


def _validate_cached_process(cached_entry: CacheEntryT, cache_key: CacheKeyT) -> Tuple[Optional[PopenObject], Optional[int]]:
    """Checks if a cached process entry is valid and the process is running and responsive."""
    process = cached_entry.get("process")
    port = cached_entry.get("port")

    if not isinstance(process, subprocess.Popen) or not isinstance(port, int):
        logger.warning(f"Invalid cache entry structure for key {cache_key}. Removing.")
        return None, None # Invalid entry

    if process.poll() is not None:
        logger.warning(f"Cached process (PID: {process.pid}, Port: {port}) for setup {cache_key} has terminated. Removing.")
        return None, None # Process terminated

    # Check API readiness with a short timeout for responsiveness
    if not check_api_ready(port, timeout=5):
        logger.warning(f"Cached process (PID: {process.pid}, Port: {port}) running but API not responding. Terminating and removing.")
        terminate_process(process) # Terminate unresponsive process (use imported function)
        return None, None # Unresponsive

    # If all checks pass
    logger.info(f"Reusing running and responsive KoboldCpp instance (PID: {process.pid}) on port {port}.")
    return process, port


def _launch_new_kobold_instance(setup_params: Dict[str, Any], cache_key: CacheKeyT) -> Tuple[Optional[PopenObject], Optional[int], Optional[str]]:
    """Handles the logic for launching a new KoboldCpp instance."""
    logger.info("Attempting to launch a new KoboldCpp instance.")
    port_to_use: Optional[int] = None
    process_to_use: Optional[PopenObject] = None
    error_message: Optional[str] = None

    try:
        # 1. Find a free port
        port_to_use = find_free_port()

        # 2. Build the launch command
        command = _build_kobold_command(setup_params, port_to_use)

        # 3. Launch the process
        process_to_use = _launch_kobold_process(command, port_to_use)

        # 4. Check if API becomes ready
        if not check_api_ready(port_to_use, timeout=120): # Increased timeout for initial launch
            error_message = f"ERROR: Launched KoboldCpp on port {port_to_use} but API did not become ready within 120s."
            # Attempt to get process output for debugging before terminating
            try:
                stdout, stderr = process_to_use.communicate(timeout=2)
                if stderr: error_message += f"\nStderr:\n{stderr.strip()}"
                if stdout: error_message += f"\nStdout:\n{stdout.strip()}"
            except Exception as comm_e:
                error_message += f"\n(Error reading process output: {comm_e})"
            finally:
                 terminate_process(process_to_use) # Ensure termination even if communicate fails (use imported function)
            logger.error(error_message)
            return None, None, error_message # Return error

        # 5. Success: Add to cache and return
        logger.info(f"New KoboldCpp instance ready on port {port_to_use} (PID: {process_to_use.pid}). Caching.")
        koboldcpp_processes_cache[cache_key] = {"process": process_to_use, "port": port_to_use} # Use imported cache
        return process_to_use, port_to_use, None

    except (OSError, ValueError, FileNotFoundError, RuntimeError, KeyError) as e:
        # Catch errors from find_free_port, _build_kobold_command, _launch_kobold_process
        logger.error(f"Failed to launch new KoboldCpp instance: {e}", exc_info=True)
        if process_to_use: # Terminate if launch started but failed later (e.g., readiness check)
            terminate_process(process_to_use) # Use imported function
        return None, None, f"ERROR: Failed to launch KoboldCpp: {e}"
    except Exception as e:
         # Catch any truly unexpected errors
         logger.critical(f"Critical unexpected error during new instance launch: {e}", exc_info=True)
         if process_to_use: terminate_process(process_to_use) # Use imported function
         return None, None, f"ERROR: Unexpected critical error launching KoboldCpp: {e}"


def _get_cached_or_launch_process(cache_key: CacheKeyT, setup_params: Dict[str, Any]
                                  ) -> Tuple[Optional[PopenObject], Optional[int], Optional[str]]:
    """
    Manages the KoboldCpp process cache.

    Checks the cache for a valid, running, and responsive process matching the
    `cache_key`. If found, returns it. If not found, or if the cached process
    is invalid/unresponsive, it terminates any other running cached instances
    and attempts to launch a new instance using `_launch_new_kobold_instance`.

    Args:
        cache_key (CacheKeyT): The tuple representing the unique configuration.
        setup_params (Dict[str, Any]): Dictionary containing parameters needed for launch.

    Returns:
        Tuple[Optional[PopenObject], Optional[int], Optional[str]]:
        - Popen object for the process (or None on failure).
        - Port number (or None on failure).
        - Error message string (or None on success).
    """
    with cache_lock: # Ensure thread-safe access to the cache (use imported lock)
        # --- 1. Check Cache ---
        cached_entry = koboldcpp_processes_cache.get(cache_key) # Use imported cache
        if cached_entry:
            process, port = _validate_cached_process(cached_entry, cache_key)
            if process and port is not None:
                # _validate_cached_process now uses terminate_process imported from process_cache
                return process, port, None # Valid cache hit
            else:
                # Invalid/dead entry found, remove it
                koboldcpp_processes_cache.pop(cache_key, None) # Use imported cache
                # Continue to launch logic...
        else:
             logger.info(f"No cached process found for setup key: {cache_key}")

        # --- 2. Cache Miss or Invalid Entry: Terminate Others & Launch New ---
        logger.info("Attempting to launch new KoboldCpp instance (terminating others first if necessary).")
        # Terminate any *other* cached instances before launching a new one
        # This ensures only one launcher-managed instance runs at a time.
        current_keys = list(koboldcpp_processes_cache.keys()) # Use imported cache
        terminated_others = False
        for key in current_keys:
             # No need to check key != cache_key because we already popped the invalid entry if it existed
             entry_to_terminate = koboldcpp_processes_cache.pop(key, None) # Use imported cache
             if entry_to_terminate and isinstance(entry_to_terminate.get("process"), subprocess.Popen):
                  proc = entry_to_terminate["process"]
                  logger.info(f"Terminating other cached instance (PID: {proc.pid}, Setup: {key})")
                  terminate_process(proc) # Use imported function
                  terminated_others = True
        if terminated_others:
             logger.debug("Finished terminating other cached instances.")

        # Launch the new instance
        process, port, error_message = _launch_new_kobold_instance(setup_params, cache_key)
        return process, port, error_message


# --- Main Execution Logic ---

def launch_and_call_api(
    # --- Setup Args (CLI) ---
    koboldcpp_path: str,
    model_path: str,
    mmproj_path: Optional[str] = None,
    gpu_acceleration: str = "None",
    n_gpu_layers: int = 0,
    context_size: int = 4096,
    use_mmap: bool = False,
    use_mlock: bool = False,
    flash_attention: bool = False,
    quant_kv: int = 0,
    threads: Optional[int] = None,
    extra_cli_args: str = "",
    # --- Generation Args (API JSON) ---
    prompt_text: str = "",
    base64_image: Optional[str] = None,
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.92,
    top_k: int = 0,
    rep_pen: float = 1.1,
    stop_sequence: Optional[List[str]] = None
) -> str:
    """
    Main entry point called by the KoboldCppLauncherNode.

    Orchestrates the process of getting a running KoboldCpp instance (either
    from cache or by launching a new one) and then sending a generation
    request to its API.

    Args:
        koboldcpp_path (str): Path to the KoboldCpp executable.
        model_path (str): Path to the GGUF model file.
        mmproj_path (Optional[str]): Optional path to the multimodal projector file.
        gpu_acceleration (str): GPU acceleration mode ("None", "CuBLAS", etc.).
        n_gpu_layers (int): Number of layers to offload to GPU.
        context_size (int): Model context size.
        use_mmap (bool): Whether to use memory mapping.
        use_mlock (bool): Whether to lock model in memory.
        flash_attention (bool): Whether to enable flash attention.
        quant_kv (int): KV cache quantization level (0, 1, 2).
        threads (Optional[int]): Number of CPU threads to use (None for auto).
        extra_cli_args (str): Additional arguments for the KoboldCpp CLI.
        prompt_text (str): The text prompt for generation.
        base64_image (Optional[str]): Optional Base64 encoded image string for multimodal input.
        max_length (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling p value.
        top_k (int): Sampling top-k value.
        rep_pen (float): Repetition penalty.
        stop_sequence (Optional[List[str]]): Optional list of strings to stop generation at.

    Returns:
        str: The generated text string on success, or an error message prefixed with "ERROR:"
             if any step (path validation, process launch, API call) fails.
    """
    logger.info("KoboldCpp Launcher: Received request.")
    start_time = time.time()

    # --- 1. Validate Essential Paths ---
    # Perform basic existence checks early to fail fast.
    logger.debug("Validating required file paths...")
    if not os.path.isfile(koboldcpp_path): # Check if it's a file specifically
        error_msg = f"ERROR: KoboldCpp executable not found or is not a file at: {koboldcpp_path}"
        logger.error(error_msg)
        return error_msg
    if not os.path.isfile(model_path):
        error_msg = f"ERROR: Model file not found or is not a file at: {model_path}"
        logger.error(error_msg)
        return error_msg
    if mmproj_path and not os.path.isfile(mmproj_path):
        error_msg = f"ERROR: MMProj file not found or is not a file at: {mmproj_path}"
        logger.error(error_msg)
        return error_msg
    logger.debug("Required paths validated successfully.")

    # --- 2. Create Cache Key from Setup Parameters ---
    # Consolidate all parameters that define a unique KoboldCpp instance configuration.
    setup_params: Dict[str, Any] = {
        "koboldcpp_path": koboldcpp_path,
        "model_path": model_path,
        "mmproj_path": mmproj_path,
        "gpu_acceleration": gpu_acceleration,
        "n_gpu_layers": n_gpu_layers,
        "context_size": context_size,
        "use_mmap": use_mmap,
        "use_mlock": use_mlock,
        "flash_attention": flash_attention,
        "quant_kv": quant_kv,
        "threads": threads,
        "extra_cli_args": extra_cli_args,
    }
    # Create a hashable key from the sorted dictionary items
    cache_key: CacheKeyT = tuple(sorted(setup_params.items()))
    logger.debug(f"Generated cache key: {cache_key}")

    # --- 3. Get Process/Port (Handles Cache and Launching) ---
    # This function encapsulates the core logic of checking the cache,
    # validating cached processes, terminating old ones, and launching/validating new ones.
    logger.info("Acquiring KoboldCpp process (checking cache or launching)...")
    process_to_use, port_to_use, error_message = _get_cached_or_launch_process(cache_key, setup_params)

    if error_message:
        # Error message already logged within the helper function
        return error_message # Return error from launch/cache check
    if port_to_use is None or process_to_use is None:
        # This case should ideally be covered by error_message, but acts as a safeguard
        critical_error_msg = "ERROR: Failed to get a valid process or port after launch attempt."
        logger.critical(critical_error_msg) # Log as critical because state is unexpected
        return critical_error_msg
    logger.info(f"Successfully acquired KoboldCpp process (PID: {process_to_use.pid}) on port {port_to_use}.")

    # --- 4. Prepare API Payload ---
    # Construct the prompt, potentially adding multimodal prefix
    # Note: The exact prompt format might need adjustment based on the model.
    # This format is common for instructing models with image context.
    final_prompt = f"\n(Attached Image)\n\n### Instruction:\n{prompt_text}\n### Response:\n" if base64_image else prompt_text
    logger.debug(f"Final prompt for API (first 100 chars): '{final_prompt[:100]}...'")

    payload: Dict[str, Any] = {
        "prompt": final_prompt,
        "max_length": max_length,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "rep_pen": rep_pen,
        "n": 1, # Generate one response
        "use_default_badwordsids": False, # Typically false for API usage
        # Add other relevant generation parameters here if needed
    }
    if base64_image:
        payload["images"] = [base64_image] # KoboldCpp expects a list of base64 strings
        logger.debug("Added base64 image to API payload.")
    if stop_sequence:
        payload["stop_sequence"] = stop_sequence
        logger.debug(f"Added stop sequences: {stop_sequence}")

    # --- 5. Call API ---
    logger.info(f"Calling KoboldCpp API on port {port_to_use}...")
    generated_text = _call_kobold_api(port_to_use, payload, cache_key)

    end_time = time.time()
    duration = end_time - start_time
    if generated_text.startswith("ERROR:"):
        logger.error(f"KoboldCpp Launcher request failed. Duration: {duration:.2f}s. Result: {generated_text}")
    else:
        logger.info(f"KoboldCpp Launcher request successful. Duration: {duration:.2f}s.")
        logger.debug(f"Generated text (first 100 chars): '{generated_text[:100]}...'")

    return generated_text
