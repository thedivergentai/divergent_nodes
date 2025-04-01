import subprocess
import os
import sys
import time
import shlex
import json
import base64
import io
import socket
import threading
import atexit
import re
import requests # Added

# --- Default Configuration ---
DEFAULT_KOBOLDCPP_PATH = r"C:\Users\djtri\Documents\KoboldCpp\koboldcpp_cu12.exe" # Default, user can override

# --- Process Cache ---
# Cache format: { cache_key: {"process": process_object, "port": port_number} }
koboldcpp_processes_cache = {}
cache_lock = threading.Lock() # To prevent race conditions when accessing the cache

# --- Helper Functions ---
import torch
import numpy as np
from PIL import Image

def tensor_to_pil(tensor):
    """Converts a torch tensor (B, H, W, C) to a single PIL Image (first image in batch)."""
    if tensor is None:
        return None
    images = []
    # Ensure tensor is on CPU and detach from graph
    tensor = tensor.detach().cpu()
    for i in range(tensor.shape[0]):
        img_np = tensor[i].numpy()
        # Handle different dtypes and ranges
        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
            # Assuming range [0, 1] for float, scale to [0, 255]
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        elif img_np.dtype != np.uint8:
            print(f"[KoboldCppNode] Warning: Unexpected tensor dtype {img_np.dtype}, attempting conversion.")
            # Attempt conversion, assuming data range is appropriate or clipping handles it
            try:
                img_np = img_np.astype(np.uint8)
            except ValueError as e:
                 print(f"[KoboldCppNode] Error converting tensor dtype {img_np.dtype} to uint8: {e}", file=sys.stderr)
                 continue # Skip this image if conversion fails

        # Ensure it's 3D (H, W, C) or 2D (H, W)
        if img_np.ndim == 3 and img_np.shape[2] == 1: # Grayscale image with channel dim
            img_np = img_np.squeeze(axis=2) # Convert to 2D grayscale
        elif img_np.ndim != 2 and (img_np.ndim != 3 or img_np.shape[2] not in [3, 4]):
             print(f"[KoboldCppNode] Warning: Unexpected tensor shape {img_np.shape}, skipping image.", file=sys.stderr)
             continue

        try:
            pil_image = Image.fromarray(img_np)
            images.append(pil_image)
        except Exception as e:
            print(f"[KoboldCppNode] Error converting tensor slice to PIL Image: {e}", file=sys.stderr)
            return None # Return None if any conversion fails
    # Return the first image in the batch
    return images[0] if images else None


def pil_to_base64(pil_image, format="jpeg"):
    """Converts a PIL Image to a Base64 encoded string."""
    if not pil_image:
        return None
    try:
        buffer = io.BytesIO()
        img_format = format.upper()
        # Convert RGBA/P to RGB if necessary for JPEG
        if pil_image.mode in ['RGBA', 'P'] and img_format == 'JPEG':
            pil_image = pil_image.convert('RGB')

        pil_image.save(buffer, format=img_format)
        img_bytes = buffer.getvalue()
        base64_string = base64.b64encode(img_bytes).decode('utf-8')
        return base64_string
    except Exception as e:
        print(f"[KoboldCppNode] Error converting PIL image to Base64: {e}", file=sys.stderr)
        return None

def find_free_port():
    """Finds an available network port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0)) # Bind to port 0 to let the OS choose an available port
        return s.getsockname()[1] # Return the chosen port

def check_api_ready(port, timeout=60):
    """Checks if the KoboldCpp API is responding."""
    start_time = time.time()
    url = f"http://localhost:{port}/api/extra/version" # Use version endpoint for check
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=1) # Short timeout for individual check
            if response.status_code == 200:
                print(f"[KoboldCppNode] API on port {port} is ready.")
                return True
        except requests.exceptions.ConnectionError:
            pass # Ignore connection errors while waiting
        except requests.exceptions.Timeout:
            print(f"[KoboldCppNode] API check timed out on port {port}.") # Log timeouts during check
        except Exception as e:
             print(f"[KoboldCppNode] Error checking API readiness on port {port}: {e}", file=sys.stderr)
             # Don't immediately fail on other errors, maybe transient
        time.sleep(0.5) # Wait before retrying
    print(f"[KoboldCppNode] API on port {port} did not become ready within {timeout} seconds.", file=sys.stderr)
    return False

def terminate_process(process):
    """Attempts to terminate a subprocess gracefully, then kills it."""
    if process and process.poll() is None: # Check if process exists and is running
        print(f"[KoboldCppNode] Terminating KoboldCpp process (PID: {process.pid})...")
        try:
            # Try graceful termination first (SIGTERM on Unix, equivalent on Windows)
            process.terminate()
            process.wait(timeout=5) # Wait a bit for graceful exit
            print(f"[KoboldCppNode] Process {process.pid} terminated gracefully.")
        except (subprocess.TimeoutExpired, PermissionError, OSError) as e:
            print(f"[KoboldCppNode] Graceful termination failed for PID {process.pid} ({e}), killing...", file=sys.stderr)
            try:
                process.kill()
                process.wait(timeout=2) # Wait briefly for kill
                print(f"[KoboldCppNode] Process {process.pid} killed.")
            except Exception as kill_e:
                print(f"[KoboldCppNode] Error killing process {process.pid}: {kill_e}", file=sys.stderr)
        except Exception as term_e:
             print(f"[KoboldCppNode] Error during termination of process {process.pid}: {term_e}", file=sys.stderr)
             # Attempt kill as fallback
             try:
                  if process.poll() is None:
                       process.kill()
                       print(f"[KoboldCppNode] Process {process.pid} killed as fallback.")
             except Exception as kill_e_fb:
                  print(f"[KoboldCppNode] Error killing process {process.pid} as fallback: {kill_e_fb}", file=sys.stderr)


def cleanup_koboldcpp_processes():
    """Terminates all cached KoboldCpp processes."""
    print("[KoboldCppNode] Cleaning up cached KoboldCpp processes...")
    with cache_lock:
        keys_to_remove = list(koboldcpp_processes_cache.keys()) # Avoid modifying dict while iterating
        for key in keys_to_remove:
            cache_entry = koboldcpp_processes_cache.pop(key, None)
            if cache_entry and "process" in cache_entry:
                terminate_process(cache_entry["process"])
    print("[KoboldCppNode] Cleanup finished.")

# Register the cleanup function to run on exit
atexit.register(cleanup_koboldcpp_processes)

# --- Main Execution Logic ---

def launch_and_call_api(
    # --- Setup Args (CLI) ---
    koboldcpp_path,
    model_path,
    mmproj_path=None,
    gpu_acceleration="None",
    n_gpu_layers=0,
    context_size=4096,
    use_mmap=False,
    use_mlock=False,
    flash_attention=False,
    quant_kv=0,
    threads=None,
    extra_cli_args="",
    # --- Generation Args (API JSON) ---
    prompt_text="",
    base64_image=None,
    max_length=512,
    temperature=0.7,
    top_p=0.92,
    top_k=0,
    rep_pen=1.1,
    stop_sequence=None,
    # --- Control ---
    force_new_instance=False # Added for potential future use? Or just rely on cache logic.
):
    """
    Manages launching/caching KoboldCpp instances and calling the API.
    Handles the hybrid launch + API approach with caching.
    """
    # --- Validate Paths ---
    if not os.path.exists(koboldcpp_path):
        return f"ERROR: KoboldCpp executable not found at: {koboldcpp_path}"
    if not os.path.exists(model_path):
        return f"ERROR: Model file not found at: {model_path}"
    if mmproj_path and not os.path.exists(mmproj_path):
        return f"ERROR: MMProj file not found at: {mmproj_path}"

    # --- Create Cache Key from Setup Parameters ---
    # Use a tuple of sorted items for consistent hashing
    setup_params = {
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
    cache_key = tuple(sorted(setup_params.items()))

    process_to_use = None
    port_to_use = None
    error_message = None

    with cache_lock:
        cached_entry = koboldcpp_processes_cache.get(cache_key)

        # --- Cache Hit Logic ---
        if cached_entry:
            print(f"[KoboldCppNode] Found cached process for setup: {cache_key}")
            process = cached_entry["process"]
            port = cached_entry["port"]
            # Check if process is still alive and API is responsive
            if process.poll() is None:
                if check_api_ready(port, timeout=5): # Quick check for running instance
                    print(f"[KoboldCppNode] Reusing running KoboldCpp instance on port {port}.")
                    process_to_use = process
                    port_to_use = port
                else:
                    print(f"[KoboldCppNode] Cached process on port {port} found but API not responding. Terminating.", file=sys.stderr)
                    terminate_process(process)
                    koboldcpp_processes_cache.pop(cache_key, None) # Remove dead entry
            else:
                print(f"[KoboldCppNode] Cached process for setup {cache_key} has terminated. Removing from cache.", file=sys.stderr)
                koboldcpp_processes_cache.pop(cache_key, None) # Remove dead entry

        # --- Cache Miss Logic (or if cached process was dead) ---
        if not process_to_use:
            print(f"[KoboldCppNode] No active cached process found for setup: {cache_key}. Launching new instance.")
            # Terminate any OTHER existing cached process before launching a new one
            # This assumes we only want one instance running via this node at a time
            current_keys = list(koboldcpp_processes_cache.keys())
            for key in current_keys:
                 if key != cache_key: # Don't terminate self if we just removed it
                      entry_to_terminate = koboldcpp_processes_cache.pop(key, None)
                      if entry_to_terminate:
                           print(f"[KoboldCppNode] Terminating other cached instance (Setup: {key})")
                           terminate_process(entry_to_terminate["process"])

            # Find a free port
            try:
                port_to_use = find_free_port()
                print(f"[KoboldCppNode] Found free port: {port_to_use}")
            except Exception as e:
                return f"ERROR: Could not find a free port: {e}"

            # Construct CLI command for launching the server
            command = [
                koboldcpp_path,
                "--model", model_path,
                "--port", str(port_to_use),
                "--contextsize", str(context_size),
                "--gpulayers", str(n_gpu_layers),
                "--quiet" # Keep it quiet
            ]
            # Add optional setup flags
            if mmproj_path: command.extend(["--mmproj", mmproj_path])
            if gpu_acceleration == "CuBLAS": command.append("--usecublas")
            elif gpu_acceleration == "CLBlast": command.extend(["--useclblast", "0", "0"]) # Still default, user needs extra_cli_args
            elif gpu_acceleration == "Vulkan": command.append("--usevulkan")
            if use_mmap: command.append("--usemmap")
            if use_mlock: command.append("--usemlock")
            if flash_attention: command.append("--flashattention")
            if quant_kv > 0: command.extend(["--quantkv", str(quant_kv)])
            if threads is not None and threads > 0: command.extend(["--threads", str(threads)])
            if extra_cli_args:
                try:
                    command.extend(shlex.split(extra_cli_args))
                except Exception as e:
                    return f"ERROR: Could not parse Extra CLI Arguments: {e}"

            # Launch the process
            print(f"[KoboldCppNode] Launching KoboldCpp: {' '.join(command)}")
            try:
                creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                # Use Popen to run in background
                process_to_use = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE, # Capture stdout/stderr for debugging if needed
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    creationflags=creationflags,
                    shell=False
                )
                print(f"[KoboldCppNode] KoboldCpp process launched (PID: {process_to_use.pid}) on port {port_to_use}.")
                # Store in cache immediately
                koboldcpp_processes_cache[cache_key] = {"process": process_to_use, "port": port_to_use}
            except Exception as e:
                return f"ERROR: Failed to launch KoboldCpp process: {e}"

            # Wait for the API to become ready
            if not check_api_ready(port_to_use, timeout=120): # Increased timeout for model loading
                error_message = f"ERROR: Launched KoboldCpp on port {port_to_use} but API did not become ready."
                # Read stderr from the failed process for more info
                try:
                     stdout_launch, stderr_launch = process_to_use.communicate(timeout=1)
                     if stderr_launch:
                          error_message += f"\nProcess Stderr:\n{stderr_launch.strip()}"
                     if stdout_launch:
                          error_message += f"\nProcess Stdout:\n{stdout_launch.strip()}"
                except Exception as comm_e:
                     error_message += f"\n(Could not get process output: {comm_e})"

                terminate_process(process_to_use)
                # Remove from cache if launch failed
                koboldcpp_processes_cache.pop(cache_key, None)
                return error_message

    # --- Prepare API Request Payload ---
    final_prompt = ""
    if base64_image:
        # Format adapted from user log / common practice
        final_prompt = f"\n(Attached Image)\n\n### Instruction:\n{prompt_text}\n### Response:\n"
    else:
        final_prompt = prompt_text # Assuming direct prompt is fine for text-only

    payload = {
        "prompt": final_prompt,
        "max_length": max_length,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "rep_pen": rep_pen,
        # Map other relevant params from Kobold API docs / user log if needed
        # Example params from user log (some might not be needed/settable per-request):
        "n": 1,
        # "max_context_length": context_size, # Usually set on launch
        "top_a": 0, # Add if input needed
        "typical": 1, # Add if input needed
        "tfs": 1, # Add if input needed
        # "rep_pen_range": 360, # Add if input needed
        # "rep_pen_slope": 0.7, # Add if input needed
        # "sampler_order": [6, 0, 1, 3, 4, 2, 5], # Add if input needed
        "use_default_badwordsids": False,
        # "quiet": True, # Already passed on launch
    }
    if base64_image:
        payload["images"] = [base64_image]
    if stop_sequence:
        payload["stop_sequence"] = stop_sequence

    # --- Call KoboldCpp API ---
    api_url = f"http://localhost:{port_to_use}/api/v1/generate" # Standard endpoint
    print(f"[KoboldCppNode] Sending API request to {api_url}")
    generated_text = f"ERROR: API call failed to {api_url}."
    try:
        response = requests.post(api_url, json=payload, timeout=300) # 5 min timeout for generation
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        response_json = response.json()
        # Extract text - structure based on Kobold API standard
        results = response_json.get("results")
        if results and isinstance(results, list) and len(results) > 0:
            generated_text = results[0].get("text", "ERROR: 'text' field not found in API response results.")
        else:
            generated_text = f"ERROR: 'results' field not found or invalid in API response: {response_json}"

    except requests.exceptions.Timeout:
        generated_text = f"ERROR: API request timed out after 300 seconds to {api_url}."
        # Consider terminating the process if the API call times out? Maybe it hung.
        # terminate_process(process_to_use)
        # koboldcpp_processes_cache.pop(cache_key, None)
    except requests.exceptions.RequestException as e:
        generated_text = f"ERROR: API request failed: {e}"
        # If connection failed, the process might be dead. Check and remove from cache.
        if isinstance(e, requests.exceptions.ConnectionError):
             with cache_lock:
                  cached_entry = koboldcpp_processes_cache.get(cache_key)
                  if cached_entry and cached_entry["process"].poll() is not None:
                       print("[KoboldCppNode] API connection failed, removing dead process from cache.", file=sys.stderr)
                       koboldcpp_processes_cache.pop(cache_key, None)
                       terminate_process(cached_entry["process"]) # Ensure cleanup
    except Exception as e:
         generated_text = f"ERROR: An unexpected error occurred during API call or response parsing: {e}"

    return generated_text


# --- ComfyUI Node Definition ---

class KoboldCppLauncherNode: # Renamed from KoboldCppNode
    """
    ComfyUI node to LAUNCH and run models using KoboldCpp (koboldcpp_cu12.exe).
    Launches KoboldCpp instance with specified setup parameters (cached),
    and sends generation requests via its API. Supports image input.
    """
    GPU_ACCELERATION_MODES = ["None", "CuBLAS", "CLBlast", "Vulkan"]
    QUANT_KV_OPTIONS = ["0: f16", "1: q8", "2: q4"] # Display names
    QUANT_KV_MAP = {"0: f16": 0, "1: q8": 1, "2: q4": 2} # Map back to int

    def __init__(self):
        pass # No instance state needed, cache is global

    @classmethod
    def INPUT_TYPES(s):
        default_path_exists = os.path.exists(DEFAULT_KOBOLDCPP_PATH)
        kobold_path_default = DEFAULT_KOBOLDCPP_PATH if default_path_exists else ""
        if not default_path_exists:
             print(f"[KoboldCppNode] Warning: Default KoboldCpp path not found: {DEFAULT_KOBOLDCPP_PATH}. Please provide the correct path.")

        return {
            "required": {
                # --- Setup Args (Used for Launching/Caching) ---
                "koboldcpp_path": ("STRING", {"multiline": False, "default": kobold_path_default}),
                "model_path": ("STRING", {"multiline": False, "default": "path/to/your/model.gguf"}),
                "gpu_acceleration": (s.GPU_ACCELERATION_MODES, {"default": "CuBLAS"}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 1000, "step": 1}),
                "context_size": ("INT", {"default": 4096, "min": 256, "max": 131072, "step": 256}),
                # --- Generation Args (Passed via API JSON) ---
                "prompt": ("STRING", {"multiline": True, "default": "Describe the image."}),
                "max_length": ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.92, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "rep_pen": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 3.0, "step": 0.01}),
            },
            "optional": {
                 # --- Optional Setup Args (Used for Launching/Caching) ---
                "mmproj_path": ("STRING", {"multiline": False, "default": ""}),
                "threads": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "use_mmap": ("BOOLEAN", {"default": True}),
                "use_mlock": ("BOOLEAN", {"default": False}),
                "flash_attention": ("BOOLEAN", {"default": False}),
                "quant_kv": (s.QUANT_KV_OPTIONS, {"default": "0: f16"}),
                "extra_cli_args": ("STRING", {"multiline": False, "default": ""}),
                 # --- Optional Generation Args (Passed via API JSON) ---
                "image_optional": ("IMAGE",),
                "stop_sequence": ("STRING", {"multiline": True, "default": ""}),
                # Add other API params as optional inputs if desired (e.g., top_a, typical)
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = "Divergent Nodes 👽/KoboldCpp" # Keep category, maybe add subcategory later if needed

    def execute(self, koboldcpp_path, model_path, prompt, gpu_acceleration, n_gpu_layers,
                context_size, max_length, temperature, top_p, top_k, rep_pen, # Added rep_pen
                mmproj_path="", image_optional=None, threads=0, use_mmap=True, use_mlock=False,
                flash_attention=False, quant_kv="0: f16", extra_cli_args="", stop_sequence=""):

        base64_image_string = None
        generated_text = "ERROR: Node execution failed."

        # --- Handle Image Input (Convert to Base64) ---
        if image_optional is not None:
             pil_image = tensor_to_pil(image_optional)
             if pil_image:
                 base64_image_string = pil_to_base64(pil_image, format="jpeg")
                 if not base64_image_string:
                      return ("ERROR: Failed to convert input image to Base64.",)
             else:
                 print("[KoboldCppNode] Warning: Could not convert input tensor to PIL image.", file=sys.stderr)

        # --- Map quant_kv display name back to integer ---
        quant_kv_int = self.QUANT_KV_MAP.get(quant_kv, 0)

        # --- Prepare Stop Sequences ---
        stop_sequence_list = None
        if stop_sequence and stop_sequence.strip():
             stop_sequence_list = [seq.strip() for seq in re.split(r'[,\n]', stop_sequence) if seq.strip()]
             if not stop_sequence_list: stop_sequence_list = None

        # --- Strip quotes from paths ---
        koboldcpp_path_cleaned = koboldcpp_path.strip().strip('"')
        model_path_cleaned = model_path.strip().strip('"')
        mmproj_path_cleaned = mmproj_path.strip().strip('"') if mmproj_path else None

        # --- Call the main logic function ---
        generated_text = launch_and_call_api(
            # Setup Args
            koboldcpp_path=koboldcpp_path_cleaned,
            model_path=model_path_cleaned,
            mmproj_path=mmproj_path_cleaned,
            gpu_acceleration=gpu_acceleration,
            n_gpu_layers=n_gpu_layers,
            context_size=context_size,
            use_mmap=use_mmap,
            use_mlock=use_mlock,
            flash_attention=flash_attention,
            quant_kv=quant_kv_int,
            threads=threads if threads > 0 else None,
            extra_cli_args=extra_cli_args,
            # Generation Args
            prompt_text=prompt,
            base64_image=base64_image_string,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            rep_pen=rep_pen,
            stop_sequence=stop_sequence_list
        )

        return (generated_text,)

# NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS are defined in __init__.py


# --- Basic API Connector Node ---

class KoboldCppApiNode:
    """
    ComfyUI node to connect to an ALREADY RUNNING KoboldCpp instance via its API.
    Does NOT launch KoboldCpp itself. Supports image input via API.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_url": ("STRING", {"multiline": False, "default": "http://localhost:5001"}),
                # --- Generation Args (Passed via API JSON) ---
                "prompt": ("STRING", {"multiline": True, "default": "Describe the image."}),
                "max_length": ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.92, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "rep_pen": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 3.0, "step": 0.01}),
            },
            "optional": {
                 # --- Optional Generation Args (Passed via API JSON) ---
                "image_optional": ("IMAGE",),
                "stop_sequence": ("STRING", {"multiline": True, "default": ""}),
                # Add other relevant API params as optional inputs if desired
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = "Divergent Nodes 👽/KoboldCpp"

    def execute(self, api_url, prompt, max_length, temperature, top_p, top_k, rep_pen,
                image_optional=None, stop_sequence=""):

        base64_image_string = None
        generated_text = "ERROR: Node execution failed."
        api_url_cleaned = api_url.strip().rstrip('/') # Clean up URL

        # --- Check API Readiness ---
        version_url = f"{api_url_cleaned}/api/extra/version"
        try:
            response = requests.get(version_url, timeout=3) # Quick timeout for check
            response.raise_for_status()
            print(f"[KoboldCppApiNode] Successfully connected to KoboldCpp API at {api_url_cleaned}.")
        except requests.exceptions.RequestException as e:
            return (f"ERROR: Failed to connect to KoboldCpp API at {api_url_cleaned}. Is it running? Details: {e}",)
        except Exception as e:
             return (f"ERROR: Unexpected error checking API status at {api_url_cleaned}: {e}",)


        # --- Handle Image Input (Convert to Base64) ---
        if image_optional is not None:
             pil_image = tensor_to_pil(image_optional)
             if pil_image:
                 base64_image_string = pil_to_base64(pil_image, format="jpeg")
                 if not base64_image_string:
                      return ("ERROR: Failed to convert input image to Base64.",)
             else:
                 print("[KoboldCppApiNode] Warning: Could not convert input tensor to PIL image.", file=sys.stderr)

        # --- Prepare Stop Sequences ---
        stop_sequence_list = None
        if stop_sequence and stop_sequence.strip():
             stop_sequence_list = [seq.strip() for seq in re.split(r'[,\n]', stop_sequence) if seq.strip()]
             if not stop_sequence_list: stop_sequence_list = None

        # --- Prepare API Request Payload ---
        final_prompt = ""
        if base64_image_string:
            final_prompt = f"\n(Attached Image)\n\n### Instruction:\n{prompt}\n### Response:\n"
        else:
            final_prompt = prompt

        payload = {
            "prompt": final_prompt,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "rep_pen": rep_pen,
            "use_default_badwordsids": False,
            # Add other params if needed
        }
        if base64_image_string:
            payload["images"] = [base64_image_string]
        if stop_sequence_list:
            payload["stop_sequence"] = stop_sequence_list

        # --- Call KoboldCpp API ---
        generate_url = f"{api_url_cleaned}/api/v1/generate"
        print(f"[KoboldCppApiNode] Sending API request to {generate_url}")
        try:
            response = requests.post(generate_url, json=payload, timeout=300) # 5 min timeout
            response.raise_for_status()

            response_json = response.json()
            results = response_json.get("results")
            if results and isinstance(results, list) and len(results) > 0:
                generated_text = results[0].get("text", "ERROR: 'text' field not found in API response results.")
            else:
                generated_text = f"ERROR: 'results' field not found or invalid in API response: {response_json}"

        except requests.exceptions.Timeout:
            generated_text = f"ERROR: API request timed out after 300 seconds to {generate_url}."
        except requests.exceptions.RequestException as e:
            generated_text = f"ERROR: API request failed: {e}"
        except Exception as e:
             generated_text = f"ERROR: An unexpected error occurred during API call or response parsing: {e}"

        return (generated_text,)
