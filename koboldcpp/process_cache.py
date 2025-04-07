"""
Manages the global cache and termination logic for KoboldCpp subprocesses.

This module isolates the caching mechanism and process termination logic
used by the process_manager.
"""
import subprocess
import threading
import atexit
import logging
from typing import Optional, Dict, Any, Tuple, List, Union, TypeAlias

# --- Type Aliases (Copied from process_manager for consistency) ---
PopenObject: TypeAlias = subprocess.Popen
CacheKeyT: TypeAlias = Tuple[Tuple[str, Any], ...]
CacheEntryT: TypeAlias = Dict[str, Union[PopenObject, int]]

# Setup logger for this module
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Process Cache ---
# Global cache to store running KoboldCpp processes managed by the launcher node.
koboldcpp_processes_cache: Dict[CacheKeyT, CacheEntryT] = {}
cache_lock = threading.Lock() # Lock for thread-safe access to the cache.

# --- Termination Logic ---

def terminate_process(process: Optional[PopenObject]) -> None:
    """
    Attempts to terminate a subprocess gracefully, then forcefully if necessary.

    Handles cases where the process is None or has already terminated.

    Args:
        process (Optional[PopenObject]): The subprocess.Popen object to terminate.
                                         Handles None safely.
    """
    if not process:
        logger.debug("Terminate process called with None process.")
        return
    # Use context manager for PID if available (Python 3.10+) for safety, else access directly
    pid_str = f"(PID: {process.pid})" if hasattr(process, 'pid') and process.pid is not None else "(PID unknown)"

    if process.poll() is not None:
        logger.debug(f"Process {pid_str} already terminated.")
        return # Process already finished

    logger.info(f"Attempting to terminate KoboldCpp process {pid_str}...")
    try:
        # 1. Try graceful termination (SIGTERM/TerminateProcess)
        process.terminate()
        try:
            # Wait for a short period for the process to exit
            process.wait(timeout=5)
            logger.info(f"Process {pid_str} terminated gracefully.")
            return # Success
        except subprocess.TimeoutExpired:
            # 2. If graceful termination times out, force kill (SIGKILL/TerminateProcess)
            logger.warning(f"Graceful termination timed out for process {pid_str}. Attempting to kill...")
            process.kill()
            try:
                process.wait(timeout=2) # Wait briefly for kill confirmation
                logger.info(f"Process {pid_str} killed forcefully.")
            except subprocess.TimeoutExpired:
                 logger.error(f"Process {pid_str} did not terminate even after kill signal.")
            except Exception as kill_wait_e:
                 logger.error(f"Error waiting for process {pid_str} after kill: {kill_wait_e}", exc_info=True)
            return # Finished kill attempt
        except Exception as wait_e:
             logger.error(f"Error waiting for process {pid_str} after terminate signal: {wait_e}", exc_info=True)
             # Fall through to kill attempt if wait fails unexpectedly

    except Exception as term_e:
         # Catch other potential errors during the initial terminate call
         logger.error(f"Error during initial termination attempt for process {pid_str}: {term_e}", exc_info=True)
         # Attempt kill as fallback if terminate failed unexpectedly
         try:
              if process.poll() is None: # Check again if it's still running
                   logger.warning(f"Initial termination failed for process {pid_str}. Attempting kill as fallback...")
                   process.kill()
                   process.wait(timeout=2) # Brief wait
                   logger.info(f"Process {pid_str} killed as fallback.")
              else:
                   logger.info(f"Process {pid_str} terminated between terminate error and kill fallback.")
         except Exception as kill_e_fb:
              logger.error(f"Error killing process {pid_str} as fallback: {kill_e_fb}", exc_info=True)

# --- Cleanup Logic ---

def cleanup_koboldcpp_processes() -> None:
    """
    Terminates all cached KoboldCpp processes managed by this module.

    This function is registered with `atexit` to ensure cleanup happens
    when the Python interpreter exits, preventing orphaned processes.
    It iterates through the process cache safely using a lock.
    """
    logger.info("Cleaning up any remaining cached KoboldCpp processes on application exit...")
    terminated_count = 0
    # Use the lock defined in this module
    with cache_lock:
        # Create a list of keys to iterate over, allowing safe removal from the dict
        keys_to_remove = list(koboldcpp_processes_cache.keys())
        logger.debug(f"Found {len(keys_to_remove)} cache entries to potentially clean up.")
        for key in keys_to_remove:
            cache_entry = koboldcpp_processes_cache.pop(key, None) # Remove entry safely
            # Ensure entry and process object exist before attempting termination
            if cache_entry and isinstance(cache_entry.get("process"), subprocess.Popen):
                process_to_terminate: PopenObject = cache_entry["process"]
                pid_str = f"(PID: {process_to_terminate.pid})" if hasattr(process_to_terminate, 'pid') and process_to_terminate.pid is not None else "(PID unknown)"
                logger.debug(f"Cleaning up process {pid_str} for cache key: {key}")
                terminate_process(process_to_terminate) # Use terminate_process from this module
                terminated_count += 1
            elif cache_entry:
                 logger.warning(f"Cache entry for key {key} was invalid or missing process object during cleanup.")

    if terminated_count > 0:
        logger.info(f"Cleanup finished. Terminated {terminated_count} cached processes.")
    else:
        logger.info("Cleanup finished. No active cached processes found to terminate.")

# Register the cleanup function to run automatically on exit
# Ensure this registration happens only once when the module is imported.
if __name__ != "__main__": # Prevent registration if script is run directly
    atexit.register(cleanup_koboldcpp_processes)
    logger.debug("Registered KoboldCpp process cleanup function with atexit.")
