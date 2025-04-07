import os
import sys # Keep sys import if needed elsewhere, otherwise remove
import re
import torch
from typing import Optional, Dict, Any, Tuple, List, TypeAlias # Added TypeAlias
import logging # Import standard logging

# Use relative import from the parent directory's utils
try:
    # Assumes shared_utils is one level up from koboldcpp
    from ..shared_utils import tensor_to_pil, pil_to_base64
except ImportError:
    # Fallback for direct execution or different structure
    # Note: safe_print is not used here anymore
    logging.warning("Could not perform relative import for shared_utils, attempting direct import.")
    # Ensure logger is configured before first use if fallback occurs
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    from shared_utils import tensor_to_pil, pil_to_base64

# Import the core logic from process_manager
try:
    from .process_manager import launch_and_call_api
    # Attempt to import DEFAULT_KOBOLDCPP_PATH if it exists, otherwise handle absence
    try:
        # This path might not be defined if process_manager itself has issues or changes
        # DEFAULT_KOBOLDCPP_PATH = "path/to/default/koboldcpp.exe" # Example placeholder if needed
        # For now, assume it might be imported or handle its absence gracefully
        pass # If it's not needed, just pass
    except ImportError:
        # DEFAULT_KOBOLDCPP_PATH = "" # Define as empty if not found
        logging.info("DEFAULT_KOBOLDCPP_PATH not found in process_manager, defaulting to empty.")
        DEFAULT_KOBOLDCPP_PATH = "" # Define locally if import fails

except ImportError:
    # Fallback if run directly
    logging.warning("Could not perform relative import for process_manager, attempting direct import.")
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    from process_manager import launch_and_call_api
    try:
        # from process_manager import DEFAULT_KOBOLDCPP_PATH
        pass # Assume not needed or handle absence
    except ImportError:
        # DEFAULT_KOBOLDCPP_PATH = ""
        logging.info("DEFAULT_KOBOLDCPP_PATH not found in process_manager (fallback), defaulting to empty.")
        DEFAULT_KOBOLDCPP_PATH = "" # Define locally if import fails


# Setup logger for this module
logger = logging.getLogger(__name__)
# Ensure handler is configured if root logger isn't set up (e.g., running standalone)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- ComfyUI Node Definition ---

class KoboldCppLauncherNode:
    """
    ComfyUI node to LAUNCH and run models using a local KoboldCpp executable.

    This node acts as an interface to the KoboldCpp process management and API
    calling logic defined in `process_manager.py`. It gathers parameters from
    the UI, prepares them, and invokes the backend logic to get a running
    KoboldCpp instance (using caching) and perform text generation.
    Supports optional multimodal input via an image tensor.
    """
    # Define types using ComfyUI conventions
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = "Divergent Nodes 👽/KoboldCpp" # Keep category consistent

    # Class variables for dropdown options
    GPU_ACCELERATION_MODES: List[str] = ["None", "CuBLAS", "CLBlast", "Vulkan"]
    QUANT_KV_OPTIONS: List[str] = ["0: f16", "1: q8", "2: q4"] # Display names
    QUANT_KV_MAP: Dict[str, int] = {"0: f16": 0, "1: q8": 1, "2: q4": 2} # Map back to int value

    def __init__(self):
        """Initializes the Launcher node instance."""
        # No instance-specific state needed; process state managed globally in process_manager
        logger.debug("KoboldCppLauncherNode instance created.")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """
        Defines the input types and options for the ComfyUI node interface.

        Includes required parameters for launching KoboldCpp and performing generation,
        as well as optional parameters for advanced configuration and multimodal input.
        Tooltips are added for better user guidance.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary defining "required" and "optional" inputs.
        """
        # Check if the default path exists to provide a better default
        # Use the imported or defaulted DEFAULT_KOBOLDCPP_PATH
        kobold_path_default = "" # Default to empty initially
        try:
             # Check if DEFAULT_KOBOLDCPP_PATH was successfully defined/imported
             if 'DEFAULT_KOBOLDCPP_PATH' in globals() and DEFAULT_KOBOLDCPP_PATH:
                  default_path_exists = os.path.exists(DEFAULT_KOBOLDCPP_PATH)
                  kobold_path_default = DEFAULT_KOBOLDCPP_PATH if default_path_exists else ""
                  if not default_path_exists:
                       print(f"[WARN] KoboldLauncherNode: Default KoboldCpp path not found: {DEFAULT_KOBOLDCPP_PATH}. Please provide the correct path.")
                  else:
                       print(f"[INFO] KoboldLauncherNode: Default KoboldCpp path found: {DEFAULT_KOBOLDCPP_PATH}")
             else:
                  print("[WARN] KoboldLauncherNode: DEFAULT_KOBOLDCPP_PATH not defined or empty. Please provide KoboldCpp path.")
        except NameError:
             print("[WARN] KoboldLauncherNode: DEFAULT_KOBOLDCPP_PATH check failed. Please provide KoboldCpp path.")
             DEFAULT_KOBOLDCPP_PATH = "" # Ensure it's defined even if check fails


        # Define input types using ComfyUI tuple format: (TYPE_STRING, Optional[Dict])
        # Added tooltips for clarity
        return {
            "required": {
                # --- Setup Args (Used for Launching/Caching) ---
                "koboldcpp_path": ("STRING", {"multiline": False, "default": kobold_path_default, "tooltip": "Path to the KoboldCpp executable (e.g., koboldcpp.exe or koboldcpp-linux-x64)."}),
                "model_path": ("STRING", {"multiline": False, "default": "path/to/your/model.gguf", "tooltip": "Path to the GGUF model file."}),
                "gpu_acceleration": (cls.GPU_ACCELERATION_MODES, {"default": "CuBLAS", "tooltip": "Select GPU acceleration mode (match your KoboldCpp build)."}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 1000, "step": 1, "tooltip": "Number of model layers to offload to GPU (-1 for auto/max)."}),
                "context_size": ("INT", {"default": 4096, "min": 256, "max": 131072, "step": 256, "tooltip": "Model context size (max tokens)."}),
                # --- Generation Args (Passed via API JSON) ---
                "prompt": ("STRING", {"multiline": True, "default": "Describe the image.", "tooltip": "The text prompt for generation."}),
                "max_length": ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1, "tooltip": "Maximum number of tokens to generate."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Sampling temperature (higher = more creative, lower = more deterministic)."}),
                "top_p": ("FLOAT", {"default": 0.92, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Nucleus sampling probability threshold."}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "Top-K sampling threshold (0 disables)."}),
                "rep_pen": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 3.0, "step": 0.01, "tooltip": "Repetition penalty (higher = less repetition)."}),
            },
            "optional": {
                 # --- Optional Setup Args (Used for Launching/Caching) ---
                "mmproj_path": ("STRING", {"multiline": False, "default": "", "tooltip": "Optional path to multimodal projector file (for LLaVA models)."}),
                "threads": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1, "tooltip": "Number of CPU threads to use (0 for auto)."}),
                "use_mmap": ("BOOLEAN", {"default": True, "tooltip": "Enable memory mapping (usually recommended)."}),
                "use_mlock": ("BOOLEAN", {"default": False, "tooltip": "Lock model in RAM (requires sufficient RAM)."}),
                "flash_attention": ("BOOLEAN", {"default": False, "tooltip": "Enable Flash Attention (requires compatible build/hardware)."}),
                "quant_kv": (cls.QUANT_KV_OPTIONS, {"default": "0: f16", "tooltip": "Quantization level for KV cache."}),
                "extra_cli_args": ("STRING", {"multiline": False, "default": "", "tooltip": "Additional command-line arguments for KoboldCpp."}),
                 # --- Optional Generation Args (Passed via API JSON) ---
                "image_optional": ("IMAGE", {"tooltip": "Optional image input for multimodal models."}), # ComfyUI's IMAGE type
                "stop_sequence": ("STRING", {
                    "multiline": True,
                    "default": "", # Comma or newline separated
                    "tooltip": "Comma or newline-separated strings to stop generation at."
                }),
            }
        }

    def execute(self,
                # Required setup args
                koboldcpp_path: str,
                model_path: str,
                gpu_acceleration: str,
                n_gpu_layers: int,
                context_size: int,
                # Required generation args
                prompt: str,
                max_length: int,
                temperature: float,
                top_p: float,
                top_k: int,
                rep_pen: float,
                # Optional setup args
                mmproj_path: str = "",
                threads: int = 0,
                use_mmap: bool = True,
                use_mlock: bool = False,
                flash_attention: bool = False,
                quant_kv: str = "0: f16", # Comes as string from dropdown
                extra_cli_args: str = "",
                # Optional generation args
                image_optional: Optional[torch.Tensor] = None,
                stop_sequence: str = ""
                ) -> Tuple[str]:
        """
        Executes the KoboldCpp Launcher node logic.

        This method is called by ComfyUI when the node needs to run. It handles:
        1. Processing optional image input (converting to Base64).
        2. Mapping UI dropdown selections (like quant_kv) to internal values.
        3. Parsing multi-line string inputs (like stop_sequence).
        4. Cleaning input paths.
        5. Calling the `launch_and_call_api` function from `process_manager.py`
           which manages the KoboldCpp process lifecycle and API interaction.
        6. Returning the generated text or an error message.

        Args:
            koboldcpp_path (str): Path to the KoboldCpp executable.
            model_path (str): Path to the GGUF model file.
            prompt (str): The text prompt for generation.
            gpu_acceleration (str): GPU acceleration mode selected ("None", "CuBLAS", etc.).
            n_gpu_layers (int): Number of layers to offload to GPU (-1 for auto/max).
            context_size (int): Model context size.
            max_length (int): Maximum number of tokens for generation.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling p value.
            top_k (int): Sampling top-k value (0 disables).
            rep_pen (float): Repetition penalty.
            mmproj_path (str): Optional path to the multimodal projector file. Defaults to "".
            image_optional (Optional[torch.Tensor]): Optional ComfyUI IMAGE tensor. Defaults to None.
            threads (int): Number of CPU threads to use (0 for auto). Defaults to 0.
            use_mmap (bool): Whether to use memory mapping. Defaults to True.
            use_mlock (bool): Whether to lock the model in memory. Defaults to False.
            flash_attention (bool): Whether to enable flash attention. Defaults to False.
            quant_kv (str): String representation of KV cache quantization ("0: f16", etc.). Defaults to "0: f16".
            extra_cli_args (str): Additional command-line arguments for KoboldCpp. Defaults to "".
            stop_sequence (str): Optional comma or newline-separated stop strings. Defaults to "".

        Returns:
            Tuple[str]: A tuple containing a single string: the generated text or an error message
                        prefixed with "ERROR:".
        """
        logger.info("KoboldCpp Launcher Node: Starting execution.")
        base64_image_string: Optional[str] = None
        generated_text: str = "ERROR: Node execution failed." # Default error message

        # --- 1. Handle Image Input (Convert to Base64) ---
        if image_optional is not None:
             logger.debug("Processing optional image input.")
             try:
                 pil_image = tensor_to_pil(image_optional) # Use shared util
                 if pil_image:
                     base64_image_string = pil_to_base64(pil_image, format="jpeg") # Use shared util
                     if not base64_image_string:
                          logger.error("Failed to convert input image tensor to Base64 string.")
                          # Return error immediately if conversion fails
                          return ("ERROR: Failed to convert input image to Base64.",)
                     logger.debug("Successfully converted image to Base64.")
                 else:
                     # tensor_to_pil returning None might indicate an issue with the tensor format
                     logger.warning("Could not convert input tensor to PIL image. Check tensor format/shape.")
                     # Depending on requirements, might return error or proceed without image
                     # For now, proceed without image but log warning.
             except Exception as e:
                  logger.error(f"Error during image processing: {e}", exc_info=True)
                  return (f"ERROR: Failed during image processing: {e}",)

        # --- 2. Map quant_kv display name back to integer value ---
        quant_kv_int: int = self.QUANT_KV_MAP.get(quant_kv, 0) # Default to 0 if mapping fails
        if quant_kv not in self.QUANT_KV_MAP:
             logger.warning(f"Invalid quant_kv value '{quant_kv}' received from input. Defaulting to 0 (f16).")

        # --- 3. Prepare Stop Sequences (Split string into list) ---
        stop_sequence_list: Optional[List[str]] = None
        if stop_sequence and stop_sequence.strip():
             # Split by comma or newline, strip whitespace from each item, filter empty strings
             stop_sequence_list = [seq.strip() for seq in re.split(r'[,\n]', stop_sequence) if seq.strip()]
             if not stop_sequence_list:
                 stop_sequence_list = None # Ensure it's None if list becomes empty after stripping
             logger.debug(f"Parsed stop sequences: {stop_sequence_list}")
        else:
             logger.debug("No stop sequences provided.")

        # --- 4. Clean paths (remove potential surrounding quotes/spaces) ---
        # It's crucial that the paths passed to process_manager are clean and validated there.
        try:
            koboldcpp_path_cleaned: str = koboldcpp_path.strip().strip('"\' ')
            model_path_cleaned: str = model_path.strip().strip('"\' ')
            # Handle empty optional path correctly -> None
            mmproj_path_cleaned: Optional[str] = mmproj_path.strip().strip('"\' ') if mmproj_path and mmproj_path.strip() else None
            logger.debug(f"Cleaned KoboldCpp path: '{koboldcpp_path_cleaned}'")
            logger.debug(f"Cleaned Model path: '{model_path_cleaned}'")
            if mmproj_path_cleaned: logger.debug(f"Cleaned MMProj path: '{mmproj_path_cleaned}'")
        except Exception as e:
             logger.error(f"Error cleaning input paths: {e}", exc_info=True)
             # Path cleaning should not fail, but catch just in case
             return (f"ERROR: Invalid input path format: {e}",)

        # --- 5. Call the main logic function from process_manager ---
        logger.info("Calling process manager to launch/cache process and call API.")
        try:
            # Map threads=0 (UI default for auto) to None for process_manager logic
            threads_arg: Optional[int] = threads if threads > 0 else None

            generated_text = launch_and_call_api(
                # Setup Args for launching/caching
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
                threads=threads_arg,
                extra_cli_args=extra_cli_args,
                # Generation Args for API call
                prompt_text=prompt,
                base64_image=base64_image_string,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                rep_pen=rep_pen,
                stop_sequence=stop_sequence_list
            )
        except Exception as e:
             # Catch unexpected errors from the launch_and_call_api function itself
             # These should ideally be handled within launch_and_call_api and return an ERROR: string,
             # but this acts as a final safeguard.
             logger.critical(f"Critical error calling launch_and_call_api: {e}", exc_info=True)
             generated_text = f"ERROR: Critical failure in process manager: {e}"

        # --- 6. Return the result ---
        # Return the result (generated text or error message) in the expected ComfyUI tuple format
        logger.info("KoboldCpp Launcher Node: Execution finished.")
        return (generated_text,)

# Note: NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS are defined
# in the koboldcpp/__init__.py file.
