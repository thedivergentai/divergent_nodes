import subprocess
import os
import sys
import tempfile
import time
import shlex # For parsing extra args safely

# --- Default Configuration ---
# Attempt to find KoboldCpp in common locations or user's path
# User should ideally provide this path if it's not standard.
DEFAULT_KOBOLDCPP_PATH = r"C:\Users\djtri\Documents\KoboldCpp\koboldcpp_cu12.exe" # Default, user can override

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
             print(f"[KoboldCppNode] Warning: Unexpected tensor dtype {img_np.dtype}, attempting conversion.")
             img_np = img_np.astype(np.uint8)
        try:
            pil_image = Image.fromarray(img_np)
            images.append(pil_image)
        except Exception as e:
            print(f"[KoboldCppNode] Error converting tensor slice to PIL Image: {e}")
            return None
    return images[0] if images else None

# --- Subprocess Interaction Logic ---

def run_koboldcpp_cli(
    koboldcpp_path,
    model_path,
    prompt,
    mmproj_path=None,
    gpu_acceleration="None", # Options: "None", "CuBLAS", "CLBlast", "Vulkan"
    n_gpu_layers=0,
    context_size=4096,
    max_output_tokens=512,
    temperature=0.7,
    top_p=0.92,
    top_k=0,
    rep_pen=1.1,
    use_mmap=False,
    use_mlock=False,
    flash_attention=False,
    quant_kv=0, # 0=f16, 1=q8, 2=q4
    threads=None, # None = auto
    extra_cli_args="",
    image_path=None # Path to temporary image file
):
    """Runs koboldcpp_cu12.exe with the --prompt flag and captures output."""
    if not os.path.exists(koboldcpp_path):
        return f"ERROR: KoboldCpp executable not found at: {koboldcpp_path}"
    if not os.path.exists(model_path):
        return f"ERROR: Model file not found at: {model_path}"
    if mmproj_path and not os.path.exists(mmproj_path):
        return f"ERROR: MMProj file not found at: {mmproj_path}"
    if image_path and not os.path.exists(image_path):
         # This shouldn't happen if temp file saving works, but check anyway
         return f"ERROR: Temporary image file not found: {image_path}"

    # --- Image Input Handling ---
    # KoboldCpp's --prompt mode doesn't seem to have a direct way to specify an input image via CLI.
    # The /image command is for interactive mode. We'll need the API or a different approach for images.
    if image_path:
        return "ERROR: Image input is not currently supported with the --prompt execution mode used by this node. Use text prompt only for now."
        # If KoboldCpp adds a CLI flag like --image-input <path> for --prompt mode, we can add it here.

    command = [
        koboldcpp_path,
        "--model", model_path,
        "--prompt", prompt, # Use the non-interactive prompt mode
        "--contextsize", str(context_size),
        "--promptlimit", str(max_output_tokens), # Map max_output_tokens to --promptlimit
        "--gpulayers", str(n_gpu_layers),
        "--temp", str(temperature),
        "--top-p", str(top_p),
        # KoboldCpp doesn't have a direct --rep-pen flag for --prompt mode?
        # It might apply default settings or use settings from a config if loaded.
        # We'll omit it for now unless the API is used later.
    ]

    # Add optional flags based on inputs
    if top_k > 0:
        command.extend(["--top-k", str(top_k)]) # Only add if > 0

    if mmproj_path:
        command.extend(["--mmproj", mmproj_path])

    # GPU Acceleration
    if gpu_acceleration == "CuBLAS":
        command.append("--usecublas") # Add options like lowvram, mmq later if needed as inputs
    elif gpu_acceleration == "CLBlast":
        # CLBlast needs platform/device IDs, which are hard to get automatically.
        # Defaulting to 0 0, user might need extra_cli_args for specific hardware.
        command.extend(["--useclblast", "0", "0"])
        print("[KoboldCppNode] Warning: Using default CLBlast platform/device 0 0. Use 'extra_cli_args' if you need different IDs (e.g., '--useclblast 1 0').")
    elif gpu_acceleration == "Vulkan":
        command.append("--usevulkan") # Add device ID later if needed

    if use_mmap:
        command.append("--usemmap")
    if use_mlock:
        command.append("--usemlock")
    if flash_attention:
        # Flash attention might require specific GPU setup (CUDA)
        if gpu_acceleration != "CuBLAS":
             print("[KoboldCppNode] Warning: Flash Attention typically requires CuBLAS.")
        command.append("--flashattention")
    if quant_kv > 0:
        if not flash_attention:
             print("[KoboldCppNode] Warning: Quantized KV Cache (--quantkv) usually requires Flash Attention (--flashattention) for full effect.")
        command.extend(["--quantkv", str(quant_kv)])

    if threads is not None and threads > 0:
        command.extend(["--threads", str(threads)])

    # Add extra arguments, parsed safely
    if extra_cli_args:
        try:
            # Use shlex to handle quoted arguments properly
            extra_args_list = shlex.split(extra_cli_args)
            command.extend(extra_args_list)
        except Exception as e:
            return f"ERROR: Could not parse Extra CLI Arguments: {e}\nArguments provided: {extra_cli_args}"

    print(f"[KoboldCppNode] Running command: {' '.join(command)}") # Log the command being run

    full_output = ""
    error_output = ""
    try:
        # Use subprocess.run for simpler non-interactive execution
        # Capture stdout and stderr, set a reasonable timeout
        # Try CREATE_NO_WINDOW first, fallback if needed
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=300, # 5 minute timeout, adjust as needed
            check=False, # Don't raise exception on non-zero exit code, check manually
            creationflags=creationflags,
            shell=False # Important for security and correct arg handling
        )

        full_output = process.stdout
        error_output = process.stderr

        if process.returncode != 0:
            print(f"[KoboldCppNode] KoboldCpp process exited with code {process.returncode}", file=sys.stderr)
            error_message = f"ERROR: KoboldCpp process failed (code {process.returncode}).\n"
            if error_output:
                error_message += f"Stderr:\n{error_output.strip()}\n"
            if full_output:
                 error_message += f"Stdout:\n{full_output.strip()}" # Include stdout in error if stderr is empty
            return error_message.strip()

        # If successful, stdout should contain the generated text.
        # KoboldCpp with --prompt might include some loading messages before the actual output.
        # We might need to refine this to extract only the AI's response if necessary.
        # For now, return the full stdout. Users might see loading text.
        return full_output.strip() if full_output else "ERROR: No output generated by KoboldCpp."

    except FileNotFoundError:
        return f"ERROR: KoboldCpp executable not found at the specified path: {koboldcpp_path}"
    except subprocess.TimeoutExpired:
        return f"ERROR: KoboldCpp process timed out after 300 seconds."
    except Exception as e:
        error_output += f"ERROR during execution: {type(e).__name__}: {e}\n"
        print(f"[KoboldCppNode] {error_output.strip()}", file=sys.stderr)
        # Try to return stderr if available
        if error_output:
             return f"ERROR: An exception occurred. Check console logs.\nDetails:\n{error_output.strip()}"
        else:
             return f"ERROR: An unexpected exception occurred: {e}. Check console logs."


# --- ComfyUI Node Definition ---

class KoboldCppNode:
    """
    ComfyUI node to run models using KoboldCpp (koboldcpp_cu12.exe).
    Uses the --prompt flag for non-interactive text generation.
    Image input is NOT supported in this mode.
    """
    GPU_ACCELERATION_MODES = ["None", "CuBLAS", "CLBlast", "Vulkan"]
    QUANT_KV_OPTIONS = ["0: f16", "1: q8", "2: q4"] # Display names
    QUANT_KV_MAP = {"0: f16": 0, "1: q8": 1, "2: q4": 2} # Map back to int

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # Check if the default path exists, provide a warning if not
        default_path_exists = os.path.exists(DEFAULT_KOBOLDCPP_PATH)
        kobold_path_default = DEFAULT_KOBOLDCPP_PATH if default_path_exists else ""
        if not default_path_exists:
             print(f"[KoboldCppNode] Warning: Default KoboldCpp path not found: {DEFAULT_KOBOLDCPP_PATH}. Please provide the correct path in the node input.")

        return {
            "required": {
                "koboldcpp_path": ("STRING", {"multiline": False, "default": kobold_path_default}),
                "model_path": ("STRING", {"multiline": False, "default": "path/to/your/model.gguf"}),
                "prompt": ("STRING", {"multiline": True, "default": "Write a short story about a brave knight."}),
                "gpu_acceleration": (s.GPU_ACCELERATION_MODES, {"default": "CuBLAS"}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 1000, "step": 1}), # -1 for auto
                "context_size": ("INT", {"default": 4096, "min": 256, "max": 131072, "step": 256}),
                "max_output_tokens": ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.92, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}), # 0 = disabled
                # Rep Pen omitted for --prompt mode for now
            },
            "optional": {
                "mmproj_path": ("STRING", {"multiline": False, "default": ""}),
                "image_optional": ("IMAGE",), # Keep for future, but will error out for now
                "threads": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}), # 0 = auto
                "use_mmap": ("BOOLEAN", {"default": True}), # Often good default
                "use_mlock": ("BOOLEAN", {"default": False}),
                "flash_attention": ("BOOLEAN", {"default": False}),
                "quant_kv": (s.QUANT_KV_OPTIONS, {"default": "0: f16"}),
                "extra_cli_args": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = "Divergent Nodes 👽/KoboldCpp"

    def execute(self, koboldcpp_path, model_path, prompt, gpu_acceleration, n_gpu_layers,
                context_size, max_output_tokens, temperature, top_p, top_k,
                mmproj_path="", image_optional=None, threads=0, use_mmap=True, use_mlock=False,
                flash_attention=False, quant_kv="0: f16", extra_cli_args=""):

        image_file_path = None
        generated_text = "ERROR: Node execution did not complete."

        # --- Handle Image Input (Save temporarily, but expect error from run_koboldcpp_cli) ---
        if image_optional is not None:
             pil_image = tensor_to_pil(image_optional)
             if pil_image:
                 try:
                     # Use tempfile for safer temporary file creation
                     with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="kobold_img_") as temp_file:
                         pil_image.save(temp_file.name, "PNG")
                         image_file_path = temp_file.name
                         print(f"[KoboldCppNode] Saved input image tensor to temporary file: {image_file_path}")
                 except Exception as e_save:
                     print(f"[KoboldCppNode] Error saving temporary image: {e_save}", file=sys.stderr)
                     image_file_path = None # Ensure it's None if saving failed
             else:
                 print("[KoboldCppNode] Warning: Could not convert input tensor to PIL image.", file=sys.stderr)

        # --- Map quant_kv display name back to integer ---
        quant_kv_int = self.QUANT_KV_MAP.get(quant_kv, 0)

        # --- Strip quotes from paths ---
        koboldcpp_path_cleaned = koboldcpp_path.strip().strip('"')
        model_path_cleaned = model_path.strip().strip('"')
        mmproj_path_cleaned = mmproj_path.strip().strip('"') if mmproj_path else None

        # --- Run CLI ---
        try:
            generated_text = run_koboldcpp_cli(
                koboldcpp_path=koboldcpp_path_cleaned,
                model_path=model_path_cleaned,
                prompt=prompt,
                mmproj_path=mmproj_path_cleaned,
                gpu_acceleration=gpu_acceleration,
                n_gpu_layers=n_gpu_layers,
                context_size=context_size,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                # rep_pen=rep_pen, # Omitted
                use_mmap=use_mmap,
                use_mlock=use_mlock,
                flash_attention=flash_attention,
                quant_kv=quant_kv_int,
                threads=threads if threads > 0 else None, # Pass None for auto
                extra_cli_args=extra_cli_args,
                image_path=image_file_path # Pass image path, run_koboldcpp_cli will handle the error
            )
        finally:
            # --- Cleanup Temp File ---
            if image_file_path and os.path.exists(image_file_path):
                 try:
                      os.remove(image_file_path)
                      print(f"[KoboldCppNode] Removed temporary image file: {image_file_path}")
                 except Exception as e_rem:
                      print(f"[KoboldCppNode] Warning: Failed to remove temporary image file {image_file_path}: {e_rem}", file=sys.stderr)

        return (generated_text,)

# NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS are defined in __init__.py
