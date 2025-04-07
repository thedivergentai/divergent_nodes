# Project Architecture

This document describes the directory structure and module organization for this ComfyUI custom node pack.

## Directory Structure

The project follows a modular structure, grouping related nodes and utilities into packages:

```
.
├── __init__.py               # Main entry point, aggregates mappings
├── clip_utils/               # Nodes related to CLIP model utilities
│   ├── __init__.py
│   └── token_counter_node.py
├── google_ai/                # Nodes interacting with Google AI APIs
│   ├── __init__.py
│   └── gemini_api_node.py
├── kobold_cpp/               # Nodes interacting with KoboldCpp
│   ├── __init__.py
│   ├── api_connector_node.py # Node to connect to existing KoboldCpp API
│   ├── launcher_node.py      # Node to launch and manage KoboldCpp process
│   └── process_manager.py    # Logic for launching, caching, terminating KoboldCpp
├── shared_utils/             # Common utility functions used across nodes
│   ├── __init__.py
│   ├── console_io.py         # Safe printing function
│   └── image_conversion.py   # Tensor <-> PIL <-> Base64 conversions
├── xy_plotting/              # Nodes related to generating XY plots
│   ├── __init__.py
│   ├── grid_assembly.py      # Logic for creating image grids and labels
│   └── lora_strength_plot_node.py # Specific XY plot node implementation
├── docs/                     # Project documentation
│   ├── ARCHITECTURE.md       # This file
│   ├── CODING_STANDARDS.md   # Coding standards guide
│   └── TASK.md               # Refactoring task tracker
├── examples/                 # (Optional) Example ComfyUI workflow JSON files
│   └── ...
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview, setup, usage
└── .gitignore                # Git ignore rules
```

## Module Descriptions

*   **`__init__.py` (Root):** The main entry point recognized by ComfyUI. It imports `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` from each node package's `__init__.py` and aggregates them into the final mappings exposed to ComfyUI.
*   **`shared_utils/`:** Contains common, reusable functions that are not specific to a single node's core logic.
    *   `console_io.py`: Provides `safe_print` for handling potential console encoding issues.
    *   `image_conversion.py`: Handles conversions between `torch.Tensor`, `PIL.Image`, and Base64 strings, commonly needed for image inputs/outputs and API interactions.
*   **`kobold_cpp/`:** Encapsulates all logic related to KoboldCpp interaction.
    *   `process_manager.py`: Manages the lifecycle (launch, check readiness, terminate, cleanup) and caching of KoboldCpp subprocesses. Contains the core `launch_and_call_api` function used by the launcher node. This separation isolates the complex process handling logic.
    *   `launcher_node.py`: Defines the `KoboldCppLauncherNode` class. Its primary role is to gather inputs from the ComfyUI interface, call the appropriate functions in `process_manager.py`, and return the result. It does *not* contain the process management logic itself.
    *   `api_connector_node.py`: Defines the `KoboldCppApiNode` class for connecting to an *already running* KoboldCpp instance. It handles direct API communication using `requests`.
*   **`google_ai/`:** Contains nodes interacting with Google AI services (currently Gemini).
    *   `gemini_api_node.py`: Defines the `GeminiNode` class, handling API key management, request building, API calls to the Gemini API, and response parsing.
*   **`clip_utils/`:** Contains nodes providing utility functions related to CLIP models.
    *   `token_counter_node.py`: Defines the `CLIPTokenCounter` node for counting tokens using a specified CLIP tokenizer.
*   **`xy_plotting/`:** Contains nodes for generating XY plot grids.
    *   `lora_strength_plot_node.py`: Defines the `LoraStrengthXYPlot` node. It orchestrates the plot generation by iterating through axes, calling the sampler, and using grid assembly functions.
    *   `grid_assembly.py`: (To be created/populated) Will contain the logic for taking a list of generated images and assembling them into a final grid image, including drawing labels.
*   **`docs/`:** Contains project documentation.
*   **`examples/`:** (Optional) Should contain example ComfyUI workflow `.json` files demonstrating how to use the nodes in this pack.

This structure aims to improve modularity, reduce file sizes, and make the codebase easier to navigate and maintain.
