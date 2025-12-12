import logging
import sys
from colorama import Fore, Style, init

# Initialize colorama for cross-platform ANSI escape code support
init(autoreset=True)

# Define custom log level for special success messages
SUCCESS_HIGHLIGHT = 25
logging.addLevelName(SUCCESS_HIGHLIGHT, "SUCCESS")

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter for Divergent Nodes with branded, colored output.
    
    Output format:
    [Divergent Nodes 👽] [NodeName] EMOJI LEVEL: message
    
    Colors:
    - Brand prefix: Green (always)
    - Node name: Cyan (always)
    - Level + message: Color based on log level
    """
    
    # Brand identity
    BRAND_NAME = "Divergent Nodes 👽"
    BRAND_COLOR = Fore.GREEN + Style.BRIGHT
    NODE_COLOR = Fore.CYAN + Style.BRIGHT
    
    # Level-specific colors
    LEVEL_COLORS = {
        "DEBUG": Fore.LIGHTBLACK_EX,  # Dim for debug
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
        "SUCCESS": Fore.MAGENTA + Style.BRIGHT,
    }
    
    # Level-specific emojis (tasteful, informative)
    LEVEL_EMOJIS = {
        "DEBUG": "🔍",      # Magnifying glass for debugging
        "INFO": "✅",       # Checkmark for success/info
        "WARNING": "⚠️",   # Warning triangle
        "ERROR": "❌",      # Red X for errors
        "CRITICAL": "🚨",  # Siren for critical
        "SUCCESS": "🎉",   # Celebration for special success
    }
    
    # Map module names to user-friendly node names
    NODE_NAME_MAP = {
        "gemini_api_node": "GeminiNode",
        "gemini_utils": "GeminiUtils",
        "gemini_batch_node": "GeminiBatch",
        "save_image_enhanced_node": "SaveImageEnhanced",
        "token_counter_node": "CLIPTokenCounter",
        "musiq_node": "MusiQScorer",
        "musiq_scorer": "MusiQScorer",
        "musiq_batch_node": "MusiQBatch",
        "lora_strength_plot_node": "LoRAStrengthXYPlot",
        "grid_assembly": "XYPlotGrid",
        "plot_utils": "XYPlotUtils",
        "sampling_helpers": "XYPlotSampling",
        "image_helpers": "XYPlotImages",
        "image_conversion": "ImageUtils",
        "config_manager": "ConfigManager",
        "logging_utils": "Logger",
        "console_io": "ConsoleIO",
        "text_encoding_utils": "TextEncoding",
        "prompt_utils": "PromptUtils",
        "batch_utils": "BatchUtils",
        "api_utils": "APIUtils",
    }
    
    def _get_node_name(self, record: logging.LogRecord) -> str:
        """Extract a user-friendly node name from the logger name."""
        if record.name == 'source':
            return "Core"
        
        # Get the last part of the logger name (module name)
        module_name = record.name.split('.')[-1]
        
        # Look up in our mapping, or format the module name nicely
        if module_name in self.NODE_NAME_MAP:
            return self.NODE_NAME_MAP[module_name]
        else:
            # Convert snake_case to PascalCase as fallback
            return ''.join(word.capitalize() for word in module_name.split('_'))
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with Divergent Nodes branding."""
        # Get components
        node_name = self._get_node_name(record)
        level_name = record.levelname
        level_color = self.LEVEL_COLORS.get(level_name, Style.RESET_ALL)
        emoji = self.LEVEL_EMOJIS.get(level_name, "")
        message = record.getMessage()
        
        # Build the formatted string
        formatted = (
            f"{self.BRAND_COLOR}[{self.BRAND_NAME}]{Style.RESET_ALL} "
            f"{self.NODE_COLOR}[{node_name}]{Style.RESET_ALL} "
            f"{level_color}{emoji} {level_name}: {message}{Style.RESET_ALL}"
        )
        
        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        
        # Add stack info if present
        if record.stack_info:
            formatted += f"\n{record.stack_info}"
        
        return formatted


def setup_node_logging(level: int = logging.INFO) -> None:
    """
    Configure logging for all Divergent Nodes.
    
    Call this ONCE in the main __init__.py.
    
    Args:
        level: Minimum log level to display (default: INFO).
               Set to logging.DEBUG for verbose output.
    """
    # Get the root logger for our package
    logger = logging.getLogger('source')
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return  # Already configured
    
    # Prevent propagation to root logger (avoids duplicate/unformatted output)
    logger.propagate = False
    
    # Create and configure console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(ColoredFormatter())
    
    logger.addHandler(handler)


class ProgressLogger:
    """
    Single-line progress logger that updates in place using carriage return.
    
    Features:
    - Updates the same line instead of flooding console
    - Optional progress bar visualization
    - ETA calculation
    - Branded output consistent with ColoredFormatter
    
    Usage:
        progress = ProgressLogger("Downloading model", total=100, node_name="GeminiNode")
        for i in range(100):
            progress.update(i + 1, status="file.safetensors")
        progress.finish("Complete!")
        
    Or as context manager:
        with ProgressLogger("Processing", total=50) as progress:
            for i in range(50):
                progress.update(i + 1)
    """
    
    BRAND_NAME = "Divergent Nodes 👽"
    BRAND_COLOR = Fore.GREEN + Style.BRIGHT
    NODE_COLOR = Fore.CYAN + Style.BRIGHT
    PROGRESS_COLOR = Fore.YELLOW
    SUCCESS_COLOR = Fore.GREEN + Style.BRIGHT
    
    def __init__(
        self,
        task_name: str,
        total: int = 100,
        node_name: str = "Core",
        bar_width: int = 20,
        show_bar: bool = True,
        show_eta: bool = True,
        stream = None
    ):
        """
        Initialize progress logger.
        
        Args:
            task_name: Description of the task (e.g., "Downloading model")
            total: Total number of steps/items
            node_name: Name of the node for branding
            bar_width: Width of progress bar in characters
            show_bar: Whether to show visual progress bar
            show_eta: Whether to calculate and show ETA
            stream: Output stream (default: sys.stdout)
        """
        self.task_name = task_name
        self.total = total
        self.node_name = node_name
        self.bar_width = bar_width
        self.show_bar = show_bar
        self.show_eta = show_eta
        self.stream = stream or sys.stdout
        
        self.current = 0
        self.start_time = None
        self._last_line_length = 0
        
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is None:
            self.finish()
        else:
            self.finish(f"Failed: {exc_val}", success=False)
        return False
    
    def start(self) -> None:
        """Start the progress tracking (records start time)."""
        import time
        self.start_time = time.time()
        self.current = 0
        self._write_progress()
    
    def update(self, current: int = None, increment: int = 1, status: str = "") -> None:
        """
        Update progress.
        
        Args:
            current: Set absolute progress value (overrides increment)
            increment: Increment progress by this amount
            status: Optional status text (e.g., current filename)
        """
        if current is not None:
            self.current = current
        else:
            self.current += increment
        
        self._write_progress(status)
    
    def finish(self, message: str = "Complete", success: bool = True) -> None:
        """
        Finish progress and print final message on new line.
        
        Args:
            message: Completion message
            success: Whether task succeeded (affects emoji/color)
        """
        # Clear the progress line
        self._clear_line()
        
        # Print final branded message
        emoji = "🎉" if success else "❌"
        color = self.SUCCESS_COLOR if success else Fore.RED
        
        final = (
            f"{self.BRAND_COLOR}[{self.BRAND_NAME}]{Style.RESET_ALL} "
            f"{self.NODE_COLOR}[{self.node_name}]{Style.RESET_ALL} "
            f"{color}{emoji} {self.task_name}: {message}{Style.RESET_ALL}"
        )
        print(final, file=self.stream)
        self.stream.flush()
    
    def _write_progress(self, status: str = "") -> None:
        """Write progress line with carriage return (updates in place)."""
        import time
        
        # Calculate percentage
        percent = (self.current / self.total * 100) if self.total > 0 else 0
        
        # Build progress bar
        bar_str = ""
        if self.show_bar:
            filled = int(self.bar_width * self.current / self.total) if self.total > 0 else 0
            bar_str = f"[{'█' * filled}{'░' * (self.bar_width - filled)}] "
        
        # Calculate ETA
        eta_str = ""
        if self.show_eta and self.start_time and self.current > 0:
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed
            remaining = (self.total - self.current) / rate if rate > 0 else 0
            
            if remaining < 60:
                eta_str = f" ETA: {remaining:.0f}s"
            elif remaining < 3600:
                eta_str = f" ETA: {remaining/60:.1f}m"
            else:
                eta_str = f" ETA: {remaining/3600:.1f}h"
        
        # Truncate status if too long
        max_status_len = 30
        if len(status) > max_status_len:
            status = status[:max_status_len-3] + "..."
        status_str = f" ({status})" if status else ""
        
        # Build the full line
        line = (
            f"\r{self.BRAND_COLOR}[{self.BRAND_NAME}]{Style.RESET_ALL} "
            f"{self.NODE_COLOR}[{self.node_name}]{Style.RESET_ALL} "
            f"{self.PROGRESS_COLOR}⏳ {self.task_name}: "
            f"{bar_str}{self.current}/{self.total} ({percent:.1f}%){status_str}{eta_str}"
            f"{Style.RESET_ALL}"
        )
        
        # Pad with spaces to overwrite previous longer lines
        padding = max(0, self._last_line_length - len(line))
        line += " " * padding
        
        self._last_line_length = len(line)
        
        # Write without newline, flush immediately
        print(line, end="", file=self.stream)
        self.stream.flush()
    
    def _clear_line(self) -> None:
        """Clear the current line."""
        # Move to start of line and overwrite with spaces
        clear = "\r" + " " * self._last_line_length + "\r"
        print(clear, end="", file=self.stream)
        self.stream.flush()


def create_progress(
    task_name: str,
    total: int,
    node_name: str = "Core",
    **kwargs
) -> ProgressLogger:
    """
    Factory function to create a ProgressLogger.
    
    Example:
        progress = create_progress("Downloading", total=100, node_name="GeminiNode")
        for i in range(100):
            progress.update(i + 1)
        progress.finish()
    """
    return ProgressLogger(task_name, total, node_name, **kwargs)


def download_with_progress(
    url: str,
    destination: str,
    node_name: str = "Core",
    chunk_size: int = 8192
) -> bool:
    """
    Download a file with branded progress bar.
    
    Args:
        url: URL to download from
        destination: Local file path to save to
        node_name: Node name for branding
        chunk_size: Download chunk size in bytes
    
    Returns:
        True if successful, False otherwise
        
    Example:
        success = download_with_progress(
            "https://example.com/model.safetensors",
            "/path/to/model.safetensors",
            node_name="GeminiNode"
        )
    """
    import requests
    import os
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get total size from headers
        total_size = int(response.headers.get('content-length', 0))
        
        # Extract filename for display
        filename = os.path.basename(destination)
        if len(filename) > 25:
            filename = filename[:22] + "..."
        
        progress = ProgressLogger(
            f"Downloading {filename}",
            total=total_size,
            node_name=node_name,
            show_bar=True,
            show_eta=True
        )
        progress.start()
        
        downloaded = 0
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Update progress with human-readable size
                    if total_size > 0:
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        progress.update(
                            current=downloaded,
                            status=f"{mb_done:.1f}/{mb_total:.1f} MB"
                        )
        
        progress.finish(f"Saved to {filename}")
        return True
        
    except Exception as e:
        logger = logging.getLogger('source')
        logger.error(f"Download failed: {e}")
        return False


class SpinnerLogger:
    """
    Animated spinner for tasks with unknown duration.
    
    Usage:
        spinner = SpinnerLogger("Connecting to API", node_name="GeminiNode")
        spinner.start()
        # ... do work ...
        spinner.finish("Connected!")
    """
    
    BRAND_NAME = "Divergent Nodes 👽"
    BRAND_COLOR = Fore.GREEN + Style.BRIGHT
    NODE_COLOR = Fore.CYAN + Style.BRIGHT
    SPINNER_COLOR = Fore.YELLOW
    SUCCESS_COLOR = Fore.GREEN + Style.BRIGHT
    
    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    def __init__(self, task_name: str, node_name: str = "Core", stream=None):
        self.task_name = task_name
        self.node_name = node_name
        self.stream = stream or sys.stdout
        self._frame_index = 0
        self._last_line_length = 0
        self._running = False
        self._thread = None
    
    def start(self) -> None:
        """Start the spinner animation in a background thread."""
        import threading
        import time
        
        self._running = True
        
        def spin():
            while self._running:
                self._write_frame()
                time.sleep(0.1)
                self._frame_index = (self._frame_index + 1) % len(self.SPINNER_FRAMES)
        
        self._thread = threading.Thread(target=spin, daemon=True)
        self._thread.start()
    
    def _write_frame(self) -> None:
        """Write current spinner frame."""
        frame = self.SPINNER_FRAMES[self._frame_index]
        
        line = (
            f"\r{self.BRAND_COLOR}[{self.BRAND_NAME}]{Style.RESET_ALL} "
            f"{self.NODE_COLOR}[{self.node_name}]{Style.RESET_ALL} "
            f"{self.SPINNER_COLOR}{frame} {self.task_name}...{Style.RESET_ALL}"
        )
        
        padding = max(0, self._last_line_length - len(line))
        line += " " * padding
        self._last_line_length = len(line)
        
        print(line, end="", file=self.stream)
        self.stream.flush()
    
    def finish(self, message: str = "Done", success: bool = True) -> None:
        """Stop spinner and print final message."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.2)
        
        # Clear line
        clear = "\r" + " " * self._last_line_length + "\r"
        print(clear, end="", file=self.stream)
        
        # Print final message
        emoji = "✅" if success else "❌"
        color = self.SUCCESS_COLOR if success else Fore.RED
        
        final = (
            f"{self.BRAND_COLOR}[{self.BRAND_NAME}]{Style.RESET_ALL} "
            f"{self.NODE_COLOR}[{self.node_name}]{Style.RESET_ALL} "
            f"{color}{emoji} {self.task_name}: {message}{Style.RESET_ALL}"
        )
        print(final, file=self.stream)
        self.stream.flush()

