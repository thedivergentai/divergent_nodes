import subprocess
import os
import sys
import threading
import queue
import time
import re
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError # Correct import path based on docs

# --- Default Configuration ---
DEFAULT_LLAMA_CLI_PATH = r"C:\Users\djtri\Documents\llama_cpp_build\llama.cpp\build\bin\Release\llama-gemma3-cli.exe" # Assuming Release build now

# Model Definitions
MODEL_CONFIGS = {
    "12B": {
        "repo_id": "bartowski/mlabonne_gemma-3-12b-it-abliterated-GGUF",
        "text_filename": "mlabonne_gemma-3-12b-it-abliterated-Q4_K_M.gguf",
        "mmproj_filename": "mmproj-mlabonne_gemma-3-12b-it-abliterated-f32.gguf",
    },
    "27B": {
        "repo_id": "bartowski/mlabonne_gemma-3-27b-it-abliterated-GGUF",
        "text_filename": "mlabonne_gemma-3-27b-it-abliterated-Q4_K_M.gguf",
        "mmproj_filename": "mmproj-mlabonne_gemma-3-27b-it-abliterated-f32.gguf",
    }
}
DEFAULT_MODEL_SIZE = "12B"
# --- End Configuration ---

# --- Helper Functions ---
import torch
import numpy as np
from PIL import Image

def tensor_to_pil(tensor):
    """Converts a torch tensor (B, H, W, C) to a list of PIL Images."""
    if tensor is None:
        return None
    images = []
    for i in range(tensor.shape[0]):
        img_np = tensor[i].cpu().numpy()
        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
             img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        elif img_np.dtype != np.uint8:
             print(f"[Gemma3VisionNode] Warning: Unexpected tensor dtype {img_np.dtype}, attempting conversion.")
             img_np = img_np.astype(np.uint8)
        try:
            pil_image = Image.fromarray(img_np)
            images.append(pil_image)
        except Exception as e:
            print(f"[Gemma3VisionNode] Error converting tensor slice to PIL Image: {e}")
            return None
    return images[0] if images else None

# --- Model Download Logic ---
_model_paths_cache = {} # Cache paths per model size

def download_model_files_cached(model_size):
    """Downloads models for a specific size if not already cached, returns paths."""
    global _model_paths_cache
    config = MODEL_CONFIGS.get(model_size)
    if not config:
        raise ValueError(f"Invalid model size specified: {model_size}")

    cache_key_text = f"{model_size}_text"
    cache_key_mmproj = f"{model_size}_mmproj"

    if cache_key_text in _model_paths_cache and cache_key_mmproj in _model_paths_cache:
        return _model_paths_cache[cache_key_text], _model_paths_cache[cache_key_mmproj]

    print(f"[Gemma3VisionNode] Checking/downloading models for size {model_size}...")
    text_model_path, mmproj_model_path = None, None
    try:
        print(f"Downloading text model: {config['repo_id']} / {config['text_filename']}")
        text_model_path = hf_hub_download(
            repo_id=config['repo_id'],
            filename=config['text_filename'],
            resume_download=True,
        )
        print(f"[Gemma3VisionNode] Text model ({model_size}) ready: {text_model_path}")

        print(f"Downloading mmproj model: {config['repo_id']} / {config['mmproj_filename']}")
        mmproj_model_path = hf_hub_download(
            repo_id=config['repo_id'],
            filename=config['mmproj_filename'],
            resume_download=True,
        )
        print(f"[Gemma3VisionNode] Multimodal projector ({model_size}) ready: {mmproj_model_path}")

        _model_paths_cache[cache_key_text] = text_model_path
        _model_paths_cache[cache_key_mmproj] = mmproj_model_path
        return text_model_path, mmproj_model_path

    except HfHubHTTPError as e:
        print(f"[Gemma3VisionNode] Error downloading models ({model_size}): {e}", file=sys.stderr)
        print("[Gemma3VisionNode] Please ensure internet connectivity and correct repo/filenames.", file=sys.stderr)
        return None, None
    except Exception as e:
        print(f"[Gemma3VisionNode] An unexpected error occurred during download ({model_size}): {e}", file=sys.stderr)
        return None, None

# --- Subprocess Interaction Logic ---

def read_output(pipe, output_queue, stop_event):
    """Reads output from a pipe and puts it into a queue."""
    try:
        while not stop_event.is_set():
            line = pipe.readline()
            if line:
                output_queue.put(line) # Keep newline for context
            else:
                time.sleep(0.05) # Small sleep if pipe empty
                if pipe.closed or not pipe.readable():
                    break
    except Exception as e:
        output_queue.put(f"[THREAD_ERROR] Error reading pipe: {e}\n")
    finally:
        output_queue.put(None) # Sentinel value

def run_gemma_cli(cli_path, text_model_path, mmproj_model_path, prompt, n_gpu_layers=99, image_path=None, temp=None, top_k=None, top_p=None):
    """Starts llama-gemma3-cli, sends commands, and captures output."""
    if not os.path.exists(cli_path):
        return f"ERROR: llama-gemma3-cli not found at: {cli_path}"

    command = [
        cli_path,
        "-m", text_model_path,
        "--mmproj", mmproj_model_path,
        "-ngl", str(n_gpu_layers) # Add GPU layers argument
    ]
    if temp is not None:
        command.extend(["--temp", str(temp)])
    if top_k is not None:
         command.extend(["--top-k", str(top_k)])
    if top_p is not None:
         command.extend(["--top-p", str(top_p)])

    print(f"[Gemma3VisionNode] Starting process: {' '.join(command)}")
    process = None
    stdout_queue = queue.Queue()
    stderr_queue = queue.Queue()
    stop_event = threading.Event()
    stdout_thread = None
    stderr_thread = None
    full_output = ""
    error_output = ""
    model_response = ""

    try:
        # Try CREATE_NO_WINDOW first, fallback if needed
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            creationflags=creationflags,
            shell=False
        )

        stdout_thread = threading.Thread(target=read_output, args=(process.stdout, stdout_queue, stop_event), daemon=True)
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr, stderr_queue, stop_event), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        # --- Wait for initial prompt ---
        print("[Gemma3VisionNode] Waiting for model load...")
        initial_prompt_seen = False
        start_time = time.time()
        timeout = 180 # Increased timeout

        while time.time() - start_time < timeout:
            try:
                line = stdout_queue.get(timeout=0.1)
                if line is None: break
                full_output += line
                if line.strip().endswith('>'):
                    initial_prompt_seen = True
                    print("[Gemma3VisionNode] Model loaded (prompt detected).")
                    break
            except queue.Empty:
                if process.poll() is not None:
                    raise RuntimeError("Process terminated unexpectedly during startup.")
                time.sleep(0.1)

        if not initial_prompt_seen:
             raise TimeoutError("Did not detect initial prompt '>' within timeout.")

        # --- Send Image Command (if applicable) ---
        if image_path:
            abs_image_path = os.path.abspath(image_path)
            if not os.path.exists(abs_image_path):
                raise FileNotFoundError(f"Image file not found: {abs_image_path}")

            print(f"[Gemma3VisionNode] Loading image: {abs_image_path}")
            image_command = f"/image {abs_image_path}\n"
            process.stdin.write(image_command)
            process.stdin.flush()
            full_output += f"[SENT] {image_command}"

            image_load_output = ""
            image_prompt_seen = False
            start_time = time.time()
            image_timeout = 15
            while time.time() - start_time < image_timeout:
                 try:
                      line = stdout_queue.get(timeout=0.1)
                      if line is None: break
                      full_output += line
                      image_load_output += line
                      if line.strip().endswith('>'):
                           image_prompt_seen = True
                           print("[Gemma3VisionNode] Image load command processed.")
                           break
                 except queue.Empty:
                      if process.poll() is not None:
                           raise RuntimeError("Process terminated unexpectedly after image command.")
                      time.sleep(0.1)
            if not image_prompt_seen:
                 print("[Gemma3VisionNode] Warning: Did not detect prompt after image load command.")

        # --- Send Prompt (with formatting) ---
        print(f"[Gemma3VisionNode] Sending formatted prompt...")
        formatted_prompt = f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        process.stdin.write(formatted_prompt)
        process.stdin.flush()
        full_output += f"[SENT] {formatted_prompt}"

        # --- Read Response ---
        print("[Gemma3VisionNode] Reading response...")
        response_lines = []
        response_prompt_seen = False
        start_time = time.time()
        response_timeout = 120

        prompt_echo_pattern = re.compile(r'^\s*>\s*$')
        end_prompt_pattern = re.compile(r'\n?>\s*$')
        in_response_section = False

        while time.time() - start_time < response_timeout:
            try:
                line = stdout_queue.get(timeout=0.1)
                if line is None: break
                full_output += line

                if not in_response_section:
                    if prompt_echo_pattern.match(line):
                        in_response_section = True
                    continue

                if end_prompt_pattern.search(line):
                     response_part = end_prompt_pattern.sub('', line)
                     if response_part:
                          response_lines.append(response_part)
                     response_prompt_seen = True
                     print("[Gemma3VisionNode] End prompt detected.")
                     break
                else:
                     response_lines.append(line)

            except queue.Empty:
                if process.poll() is not None:
                    print("[Gemma3VisionNode] Warning: Process terminated while waiting for response.")
                    break
                time.sleep(0.1)

        if not response_prompt_seen:
            print("[Gemma3VisionNode] Warning: Did not detect end prompt '>' after response timeout.")

        model_response = "".join(response_lines).strip()
        print(f"[Gemma3VisionNode] Raw response length: {len(model_response)}")

    except (RuntimeError, TimeoutError, FileNotFoundError, Exception) as e:
        error_output += f"ERROR during execution: {type(e).__name__}: {e}\n"
        print(f"[Gemma3VisionNode] {error_output.strip()}", file=sys.stderr)
    finally:
        # --- Cleanup ---
        if process:
            stop_event.set()
            if process.poll() is None:
                try:
                    print("[Gemma3VisionNode] Sending /quit command.")
                    process.stdin.write("/quit\n")
                    process.stdin.flush()
                    process.stdin.close()
                    process.wait(timeout=5)
                    print("[Gemma3VisionNode] Process terminated gracefully.")
                except (OSError, ValueError, subprocess.TimeoutExpired) as e:
                    print(f"[Gemma3VisionNode] Process did not exit gracefully ({e}), killing.", file=sys.stderr)
                    process.kill()
                except Exception as e_term:
                     print(f"[Gemma3VisionNode] Error during termination: {e_term}", file=sys.stderr)
                     if process.poll() is None: process.kill()

            if stdout_thread and stdout_thread.is_alive(): stdout_thread.join(timeout=1)
            if stderr_thread and stderr_thread.is_alive(): stderr_thread.join(timeout=1)

            while True:
                try: line = stdout_queue.get_nowait(); full_output += line if line else ""
                except queue.Empty: break
            while True:
                try: line = stderr_queue.get_nowait(); error_output += line if line else ""
                except queue.Empty: break

    if error_output:
        print(f"[Gemma3VisionNode] Stderr output:\n{error_output.strip()}", file=sys.stderr)
        if model_response and "ERROR during execution" in error_output:
             model_response = f"ERROR: Check Logs.\n---\n{model_response}"
        elif "ERROR during execution" in error_output:
             model_response = f"ERROR: Execution failed. Check console logs.\nDetails:\n{error_output.strip()}"
        elif not model_response:
             model_response = f"WARNING: Process finished with stderr output but no main response. Check logs.\nDetails:\n{error_output.strip()}"

    return model_response if model_response else "ERROR: No response generated."


# --- ComfyUI Node Definition ---

class Gemma3VisionNode:
    """
    ComfyUI node to run Gemma 3 vision models using llama-gemma3-cli.
    Requires llama-gemma3-cli to be compiled separately (ideally with CUDA).
    """
    MODEL_SIZES = list(MODEL_CONFIGS.keys())

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_size": (s.MODEL_SIZES, {"default": DEFAULT_MODEL_SIZE}),
                "prompt": ("STRING", {"multiline": True, "default": "Describe the image."}),
                "n_gpu_layers": ("INT", {"default": 99, "min": 0, "max": 1000, "step": 1}), # GPU layers
            },
            "optional": {
                "image_optional": ("IMAGE",),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 1, "max": 1000, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cli_path_override": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = "Divergent Nodes 👽/Gemma"

    def execute(self, model_size, prompt, n_gpu_layers, image_optional=None, temperature=0.8, top_k=40, top_p=0.95, cli_path_override=""):

        cli_path = cli_path_override.strip() if cli_path_override.strip() else DEFAULT_LLAMA_CLI_PATH
        image_file_path = None

        # --- Handle Image Input ---
        if image_optional is not None:
             pil_image = tensor_to_pil(image_optional)
             if pil_image:
                 try:
                     temp_dir = os.path.join(os.path.dirname(__file__), "temp_images")
                     os.makedirs(temp_dir, exist_ok=True)
                     temp_filename = f"temp_image_{int(time.time()*1000)}.png"
                     image_file_path = os.path.join(temp_dir, temp_filename)
                     pil_image.save(image_file_path, "PNG")
                     print(f"[Gemma3VisionNode] Saved input image tensor to temporary file: {image_file_path}")
                 except Exception as e_save:
                     print(f"[Gemma3VisionNode] Error saving temporary image: {e_save}", file=sys.stderr)
                     image_file_path = None
             else:
                 print("[Gemma3VisionNode] Warning: Could not convert input tensor to PIL image.", file=sys.stderr)

        # --- Download Models ---
        text_model_path, mmproj_model_path = download_model_files_cached(model_size)
        if not text_model_path or not mmproj_model_path:
            return ("ERROR: Failed to download necessary model files. Check logs.",)

        # --- Run CLI ---
        generated_text = "ERROR: Execution did not complete."
        try:
            generated_text = run_gemma_cli(
                cli_path=cli_path,
                text_model_path=text_model_path,
                mmproj_model_path=mmproj_model_path,
                prompt=prompt,
                n_gpu_layers=n_gpu_layers, # Pass GPU layers
                image_path=image_file_path,
                temp=temperature,
                top_k=top_k,
                top_p=top_p
            )
        finally:
            # --- Cleanup Temp File ---
            if image_file_path and os.path.exists(image_file_path):
                 try:
                      os.remove(image_file_path)
                      print(f"[Gemma3VisionNode] Removed temporary image file: {image_file_path}")
                 except Exception as e_rem:
                      print(f"[Gemma3VisionNode] Warning: Failed to remove temporary image file {image_file_path}: {e_rem}", file=sys.stderr)

        return (generated_text,)

# NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS are defined in __init__.py
