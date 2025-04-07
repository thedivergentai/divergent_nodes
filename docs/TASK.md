# Refactoring Task Progress

This document tracks the progress of refactoring the ComfyUI custom node pack.

**Objective:** Refactor the codebase for improved structure, clarity, documentation, testability, and consistency, making it easier for AI agents and human developers to interact with. Enforce a maximum file length of 500 lines.

**Plan Phases:**

**Phase 1: Establish Structure & Utilities (Focus: Organization)**
*   [X] Create Directory Structure (`kobold_cpp/`, `google_ai/`, `clip_utils/`, `xy_plotting/`, `shared_utils/`, `docs/`)
*   [X] Create Shared Utilities (`shared_utils/`):
    *   [X] `console_io.py` (with `safe_print`)
    *   [X] `image_conversion.py` (with `tensor_to_pil`, `pil_to_base64`)
    *   [X] `__init__.py`

**Phase 2: Split & Reorganize KoboldCpp (Focus: Modularity & Line Limits)**
*   [X] Create `kobold_cpp/api_connector_node.py` (with `KoboldCppApiNode`)
*   [X] Delete original `koboldcpp_node.py`
*   [X] Create `kobold_cpp/__init__.py`
*   [X] Update main `__init__.py` for `koboldcpp` package
*   [X] Create `kobold_cpp/process_manager.py` (with process/cache logic, `launch_and_call_api`)
*   [X] Refactor `kobold_cpp/launcher_node.py` (imports from `process_manager`)

**Phase 3: Reorganize Remaining Nodes (Focus: Organization)**
*   [ ] **XY Plotting (`xy_plotting/`):**
    *   [ ] Move `xy_plot_nodes.py` -> `xy_plotting/lora_strength_plot_node.py`.
    *   [ ] Create `xy_plotting/__init__.py`.
    *   [ ] Create `xy_plotting/grid_assembly.py` (logic moved later).
    *   [ ] Refactor `lora_strength_plot_node.py` imports & ensure line limit.
*   [ ] **CLIP Utils (`clip_utils/`):**
    *   [ ] Move `clip_token_counter.py` -> `clip_utils/token_counter_node.py`.
    *   [ ] Create `clip_utils/__init__.py`.
*   [ ] **Google AI (`google_ai/`):**
    *   [ ] Move `gemini_node.py` -> `google_ai/gemini_api_node.py`.
    *   [ ] Create `google_ai/__init__.py`.
    *   [ ] Update `gemini_api_node.py` imports.

**Phase 4: Finalize Structure & Documentation (Focus: Integration & Standards)**
*   [ ] Update Main `__init__.py` for all moved nodes.
*   [X] Create `docs/TASK.md`.
*   [ ] Create `docs/CODING_STANDARDS.md`.
*   [ ] Create `docs/ARCHITECTURE.md`.

**Phase 5: Internal Refactoring (Focus: Code Quality)**
*   [ ] **KoboldCpp:**
    *   [ ] Refactor `kobold_cpp/process_manager.py` (decompose `launch_and_call_api`, add type hints, logging, error handling).
    *   [ ] Refactor `kobold_cpp/launcher_node.py` (add type hints, docstrings, logging).
    *   [ ] Refactor `kobold_cpp/api_connector_node.py` (add type hints, docstrings, logging, error handling).
    *   [ ] Add Unit Tests.
*   [ ] **XY Plotting:**
    *   [ ] Move grid/label logic to `grid_assembly.py`.
    *   [ ] Refactor `lora_strength_plot_node.py` (decompose `execute`, add type hints, docstrings, logging, error handling).
    *   [ ] Refactor `grid_assembly.py` (add type hints, docstrings, logging, error handling).
    *   [ ] Add Unit/Integration Tests.
*   [ ] **Google AI:**
    *   [ ] Refactor `gemini_api_node.py` (decompose `generate`, add type hints, docstrings, logging, error handling).
    *   [ ] Add Unit Tests (mocking `genai`).
*   [ ] **CLIP Utils:**
    *   [ ] Refactor `token_counter_node.py` (add type hints, docstrings, logging, structure).
    *   [ ] Add Unit Tests.
*   [ ] **Shared Utils:**
    *   [ ] Add type hints, docstrings, logging.
    *   [ ] Add Unit Tests.
*   [ ] **Project Files:**
    *   [ ] Update `README.md`.
    *   [ ] Add linter/formatter configs (`pyproject.toml`?).
    *   [ ] Add example workflows (`examples/`).
