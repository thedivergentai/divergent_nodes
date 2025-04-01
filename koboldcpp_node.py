import subprocess
import os
import sys
import tempfile
import time
import shlex # For parsing extra args safely
import json # Added
import base64 # Added
import io # Added

# --- Default Configuration ---
# Attempt to find KoboldCpp in common locations or user's path
# User should ideally provide this path if it's not standard.
DEFAULT_KOBOLDCPP_PATH = r"C:\Users\djtri\Documents\KoboldCpp\koboldcpp_cu12.exe" # Default, user can override

# --- Helper Functions ---
import torch
import numpy as np
from PIL import Image

def tensor_to_pil(tensor):
    """Converts a torch tensor (B, H, W, C) to a single PIL Image (first image in batch)."""
    if tensor is None:
        return None
    images = []
    for i in range(tensor.shape[0]):
        img_np = tensor[i].cpu().numpy()
        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
             img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        elif img_np.dtype != np.uint8:
             print(f"[KoboldCppNode] Warning: Unexpected tensor dtype {img_np.dtype}, attempting conversion.")
             img_np = img_np.astype(np.uint8)
        try:
            pil_image = Image.fromarray(img_np)
            images.append(pil_image)
        except Exception as e:
            print(f"[KoboldCppNode] Error converting tensor slice to PIL Image: {e}")
            return None
    # Return the first image in the batch
    return images[0] if images else None

def pil_to_base64(pil_image, format="jpeg"):
    """Converts a PIL Image to a Base64 encoded string."""
    if not pil_image:
        return None
    try:
        buffer = io.BytesIO()
        # Convert RGBA to RGB if necessary for JPEG
        if pil_image.mode == 'RGBA' and format.lower() == 'jpeg':
            pil_image = pil_image.convert('RGB')
        elif pil_image.mode == 'P' and format.lower() == 'jpeg': # Handle palette images for JPEG
             pil_image = pil_image.convert('RGB')

        pil_image.save(buffer, format=format.upper())
        img_bytes = buffer.getvalue()
        base64_string = base64.b64encode(img_bytes).decode('utf-8')
        return base64_string
    except Exception as e:
        print(f"[KoboldCppNode] Error converting PIL image to Base64: {e}", file=sys.stderr)
        return None

# --- Subprocess Interaction Logic ---

def run_koboldcpp_cli(
    koboldcpp_path,
    model_path,
    # --- Generation Params (passed via JSON) ---
    prompt_text,
    base64_image=None,
    max_length=512,
    temperature=0.7,
    top_p=0.92,
    top_k=0,
    rep_pen=1.1,
    stop_sequence=None, # Optional list of stop sequences
    # --- Setup Params (passed via CLI) ---
    mmproj_path=None,
    gpu_acceleration="None", # Options: "None", "CuBLAS", "CLBlast", "Vulkan"
    n_gpu_layers=0,
    context_size=4096,
    use_mmap=False,
    use_mlock=False,
    flash_attention=False,
    quant_kv=0, # 0=f16, 1=q8, 2=q4
    threads=None, # None = auto
    extra_cli_args=""
):
    """
    Runs koboldcpp_cu12.exe, passing setup args via CLI and generation args via JSON on stdin.
    Captures output from stdout.
    """
    # --- Validate Paths ---
    if not os.path.exists(koboldcpp_path):
        return f"ERROR: KoboldCpp executable not found at: {koboldcpp_path}"
    if not os.path.exists(model_path):
        return f"ERROR: Model file not found at: {model_path}"
    if mmproj_path and not os.path.exists(mmproj_path):
        return f"ERROR: MMProj file not found at: {mmproj_path}"

    # --- Construct CLI Command (Setup Arguments Only) ---
    command = [
        koboldcpp_path,
        "--model", model_path,
        "--contextsize", str(context_size),
        "--gpulayers", str(n_gpu_layers),
        "--quiet" # Add quiet to minimize stdout noise from KoboldCpp itself
        # Removed --prompt, --promptlimit, --temp, --top-p, --top-k as they go in JSON
    ]

    # Add optional flags based on inputs
    # Add optional setup flags based on inputs
    if mmproj_path:
        command.extend(["--mmproj", mmproj_path])

     # GPU Acceleration
    # (GPU Acceleration, mmap, mlock, flash_attention, quant_kv, threads logic remains the same)
    if gpu_acceleration == "CuBLAS":
        command.append("--usecublas")
    elif gpu_acceleration == "CLBlast":
        command.extend(["--useclblast", "0", "0"])
        print("[KoboldCppNode] Warning: Using default CLBlast platform/device 0 0. Use 'extra_cli_args' if you need different IDs (e.g., '--useclblast 1 0').")
    elif gpu_acceleration == "Vulkan":
        command.append("--usevulkan")

    if use_mmap:
        command.append("--usemmap")
    if use_mlock:
        command.append("--usemlock")
    if flash_attention:
        if gpu_acceleration != "CuBLAS":
             print("[KoboldCppNode] Warning: Flash Attention typically requires CuBLAS.")
        command.append("--flashattention")
    if quant_kv > 0:
        if not flash_attention:
             print("[KoboldCppNode] Warning: Quantized KV Cache (--quantkv) usually requires Flash Attention (--flashattention) for full effect.")
        command.extend(["--quantkv", str(quant_kv)])

    if threads is not None and threads > 0:
        command.extend(["--threads", str(threads)])

     # Add extra setup arguments, parsed safely
    # (Parsing extra_cli_args remains the same)
    if extra_cli_args:
        try:
            extra_args_list = shlex.split(extra_cli_args)
            command.extend(extra_args_list)
        except Exception as e:
            return f"ERROR: Could not parse Extra CLI Arguments: {e}\nArguments provided: {extra_cli_args}"

    # --- Construct JSON Payload (Generation Arguments) ---
    # Format prompt based on image presence
    final_prompt = ""
    if base64_image:
        # Using a format similar to the user's log example for multimodal prompts
        # Note: The exact placeholder like "(Attached Image)" might vary or be unnecessary
        # if KoboldCpp implicitly knows an image is attached via the "images" key.
        # Let's try a common format first.
        final_prompt = f"\n(Attached Image)\n\n### Instruction:\n{prompt_text}\n### Response:\n"
    else:
        final_prompt = prompt_text # Or adjust if a specific format is needed for text-only

    payload = {
        "prompt": final_prompt,
        "max_length": max_length,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "rep_pen": rep_pen,
        "quiet": True, # Ensure quiet mode in payload too
        # Add other relevant payload keys based on Kobold API / user log
        "use_default_badwordsids": False, # From user log
        "bypass_eos": False, # Assuming default, might need input
        # "sampler_order": [6, 0, 1, 3, 4, 2, 5], # Can be added if needed
    }

    if base64_image:
        payload["images"] = [base64_image]

    if stop_sequence:
        payload["stop_sequence"] = stop_sequence

    try:
        json_payload = json.dumps(payload)
    except Exception as e:
        return f"ERROR: Failed to serialize JSON payload: {e}"

    # --- Execute Process ---
    print(f"[KoboldCppNode] Running command: {' '.join(command)}")
    print(f"[KoboldCppNode] Sending JSON payload via stdin: {json_payload[:200]}...") # Log truncated payload

    stdout_data = ""
    stderr_data = ""
    process = None
    try:
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            creationflags=creationflags,
            shell=False
        )

        # Send JSON payload to stdin and capture output
        stdout_data, stderr_data = process.communicate(input=json_payload, timeout=300) # 5 min timeout

        if process.returncode != 0:
            print(f"[KoboldCppNode] KoboldCpp process exited with code {process.returncode}", file=sys.stderr)
            error_message = f"ERROR: KoboldCpp process failed (code {process.returncode}).\n"
            if stderr_data:
                error_message += f"Stderr:\n{stderr_data.strip()}\n"
            if stdout_data:
                 error_message += f"Stdout (may contain partial output or errors):\n{stdout_data.strip()}"
            return error_message.strip()

        # --- Parse Output ---
        # The actual response might be mixed with other logs in stdout, even with --quiet.
        # Based on the user log, the response appears after the timing info.
        # Let's try finding the response after the last timing line or similar marker.
        # A simple approach first: assume the last non-empty lines are the response.
        lines = stdout_data.strip().splitlines()
        response_content = ""
        # Find the line with timing info like "[HH:MM:SS] CtxLimit:..."
        timing_line_index = -1
        for i in reversed(range(len(lines))):
             if re.search(r"\[\d{2}:\d{2}:\d{2}\]\s+CtxLimit:", lines[i]):
                  timing_line_index = i
                  break

        if timing_line_index != -1 and timing_line_index + 1 < len(lines):
             # Assume response starts on the next line after timing info
             response_content = "\n".join(lines[timing_line_index + 1:]).strip()
        elif lines:
             # Fallback: If timing line not found, maybe return the last line? Or all non-empty lines?
             # Let's try returning the last non-empty line as a guess.
             for line in reversed(lines):
                  if line.strip():
                       response_content = line.strip()
                       print("[KoboldCppNode] Warning: Could not find timing marker in output, returning last non-empty line as response.")
                       break
             if not response_content: # If all lines were empty after stripping
                  response_content = stdout_data.strip() # Return everything as fallback
                  print("[KoboldCppNode] Warning: Could not parse response effectively, returning full stdout.")
        else:
             # If stdout is empty or only whitespace
             response_content = ""

        if not response_content and stderr_data:
             print("[KoboldCppNode] No response content found in stdout, checking stderr.", file=sys.stderr)
             return f"ERROR: No response generated. Stderr:\n{stderr_data.strip()}"
        elif not response_content:
             return "ERROR: No response generated (stdout was empty)."

        return response_content

    except FileNotFoundError:
        return f"ERROR: KoboldCpp executable not found at the specified path: {koboldcpp_path}"
    except subprocess.TimeoutExpired:
        if process: process.kill()
        stdout_data, stderr_data = process.communicate() # Get any remaining output
        error_message = "ERROR: KoboldCpp process timed out after 300 seconds.\n"
        if stderr_data: error_message += f"Stderr:\n{stderr_data.strip()}\n"
        if stdout_data: error_message += f"Stdout:\n{stdout_data.strip()}"
        return error_message.strip()
    except Exception as e:
        if process: process.kill() # Ensure process is killed on other errors
        error_message = f"ERROR during execution: {type(e).__name__}: {e}\n"
        # Attempt to get stderr if available
        try:
             if stderr_data: error_message += f"Stderr:\n{stderr_data.strip()}\n"
             if stdout_data: error_message += f"Stdout:\n{stdout_data.strip()}"
        except: pass # Ignore errors during error reporting itself
        print(f"[KoboldCppNode] {error_message.strip()}", file=sys.stderr)
        return error_message.strip()


# --- ComfyUI Node Definition ---

class KoboldCppNode:
    """
    ComfyUI node to run models using KoboldCpp (koboldcpp_cu12.exe).
    Passes setup args via CLI and generation args via JSON on stdin.
    Supports text and image input (via Base64 in JSON).
    """
    # (Keep GPU_ACCELERATION_MODES, QUANT_KV_OPTIONS, QUANT_KV_MAP)
    GPU_ACCELERATION_MODES = ["None", "CuBLAS", "CLBlast", "Vulkan"]
    QUANT_KV_OPTIONS = ["0: f16", "1: q8", "2: q4"] # Display names
    QUANT_KV_MAP = {"0: f16": 0, "1: q8": 1, "2: q4": 2} # Map back to int

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # (Keep default path checking logic)
        default_path_exists = os.path.exists(DEFAULT_KOBOLDCPP_PATH)
        kobold_path_default = DEFAULT_KOBOLDCPP_PATH if default_path_exists else ""
        if not default_path_exists:
             print(f"[KoboldCppNode] Warning: Default KoboldCpp path not found: {DEFAULT_KOBOLDCPP_PATH}. Please provide the correct path in the node input.")

        return {
            "required": {
                # --- Setup Args (CLI) ---
                "koboldcpp_path": ("STRING", {"multiline": False, "default": kobold_path_default}),
                "model_path": ("STRING", {"multiline": False, "default": "path/to/your/model.gguf"}),
                "gpu_acceleration": (s.GPU_ACCELERATION_MODES, {"default": "CuBLAS"}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 1000, "step": 1}), # -1 for auto
                "context_size": ("INT", {"default": 4096, "min": 256, "max": 131072, "step": 256}),
                # --- Generation Args (JSON via stdin) ---
                "prompt": ("STRING", {"multiline": True, "default": "Describe the image."}), # Renamed from prompt_text for clarity
                "max_length": ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}), # Renamed from max_output_tokens
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.92, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}), # 0 = disabled
                "rep_pen": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 3.0, "step": 0.01}), # Added rep_pen input
            },
            "optional": {
                 # --- Optional Setup Args (CLI) ---
                "mmproj_path": ("STRING", {"multiline": False, "default": ""}),
                "threads": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}), # 0 = auto
                "use_mmap": ("BOOLEAN", {"default": True}),
                "use_mlock": ("BOOLEAN", {"default": False}),
                "flash_attention": ("BOOLEAN", {"default": False}),
                "quant_kv": (s.QUANT_KV_OPTIONS, {"default": "0: f16"}),
                "extra_cli_args": ("STRING", {"multiline": False, "default": ""}),
                 # --- Optional Generation Args (JSON via stdin) ---
                "image_optional": ("IMAGE",), # Now functional
                "stop_sequence": ("STRING", {"multiline": True, "default": ""}), # Input for stop sequences (comma/newline separated)
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = "Divergent Nodes 👽/KoboldCpp"

    def execute(self, koboldcpp_path, model_path, prompt, gpu_acceleration, n_gpu_layers,
                context_size, max_length, temperature, top_p, top_k, rep_pen, # Added rep_pen
                mmproj_path="", image_optional=None, threads=0, use_mmap=True, use_mlock=False,
                flash_attention=False, quant_kv="0: f16", extra_cli_args="", stop_sequence=""):

        base64_image_string = None
        generated_text = "ERROR: Node execution did not complete."

        # --- Handle Image Input (Convert to Base64) ---
        if image_optional is not None:
             pil_image = tensor_to_pil(image_optional)
             if pil_image:
                 base64_image_string = pil_to_base64(pil_image, format="jpeg") # Use JPEG for potentially smaller size
                 if not base64_image_string:
                      return ("ERROR: Failed to convert input image to Base64.",)
             else:
                 print("[KoboldCppNode] Warning: Could not convert input tensor to PIL image.", file=sys.stderr)
                 # Proceed without image if conversion failed but tensor was provided

        # --- Map quant_kv display name back to integer ---
        quant_kv_int = self.QUANT_KV_MAP.get(quant_kv, 0)

        # --- Prepare Stop Sequences ---
        stop_sequence_list = None
        if stop_sequence and stop_sequence.strip():
             # Split by newline or comma, trim whitespace
             stop_sequence_list = [seq.strip() for seq in re.split(r'[,\n]', stop_sequence) if seq.strip()]
             if not stop_sequence_list: # Handle case where input is just whitespace/commas
                  stop_sequence_list = None

        # --- Strip quotes from paths ---
        koboldcpp_path_cleaned = koboldcpp_path.strip().strip('"')
        model_path_cleaned = model_path.strip().strip('"')
        mmproj_path_cleaned = mmproj_path.strip().strip('"') if mmproj_path else None

        # --- Run CLI with JSON payload ---
        # No temp file needed for image path anymore
        generated_text = run_koboldcpp_cli(
            # Setup Args (CLI)
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
            # Generation Args (JSON via stdin)
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
