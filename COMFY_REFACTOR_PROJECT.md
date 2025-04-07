# ComfyUI Custom Nodes Refactoring Project

This document tracks the autonomous refactoring process for the ComfyUI custom nodes located in this repository. The goal is to optimize the codebase for AI agent interaction and improve human developer productivity.

## Project Goal

Autonomously refactor the target ComfyUI custom node codebase(s) to optimize it for interaction with other AI agents and improve human developer productivity within AI-integrated IDEs. Systematically enhance structure, clarity, documentation, testability, and consistency, **ensuring no code file exceeds 500 lines (as per user confirmation, overriding the 400-line limit mentioned in `docs/CODING_STANDARDS.md`)**, so that AI assistants can more effectively understand, navigate, modify, generate, test, and debug the Python backend and any associated JavaScript frontend code. Execute this plan, ensuring all significant changes are presented for human review and *tested within ComfyUI* before final integration, and that all progress and decisions are logged in this document.

**Context:** This refactoring effort builds upon a previous plan documented in `docs/TASK.md` and existing standards in `docs/CODING_STANDARDS.md` and `docs/ARCHITECTURE.md`. The initial phases focused on establishing the modular directory structure and organizing node files. This current task corresponds primarily to **Phase 5: Internal Refactoring** from `docs/TASK.md`, focusing on code quality improvements (decomposition, type hints, docstrings, logging, error handling, testing, examples, README) within the established structure.

### Confirmed High-Priority AI Objective

*   **Creating/Updating example workflow JSON files:** Refactoring efforts will prioritize changes that make it easier for AI agents to understand node inputs, outputs, and typical usage patterns to generate accurate and useful example workflows. This includes clear naming, comprehensive type hinting, and well-documented node purposes.

## Refactoring Plan Outline (Based on Task Prompt)

1.  **Strategic Planning & Preparation Phase (Analyze & Propose)**
    *   Confirm AI Objectives & Metrics
    *   Establish Baselines (Tests, Linting, File Lengths, Tech Debt)
    *   Propose Refactoring Scope & Prioritization (Focus on file length initially)
    *   Define Incremental Strategy (Node-by-node, file-by-file, splitting long files)
    *   Adhere to Git Hygiene
    *   Internalize Team Guidance
    *   Verify Tooling
2.  **Core Code Refactoring Tasks (Execute Iteratively with Review Gates)**
    *   Enhance Modularity & SRP (incl. **500-line file limit enforcement**)
    *   Maximize Clarity and Readability (Naming, Logic Simplification, Dead Code Removal)
    *   Prioritize Explicitness (Type Hinting, ComfyUI Types, Dependencies, Error Handling)
    *   Develop Documentation *as* Code & Context (Docstrings, READMEs, Examples)
    *   Build Testability & Tests (Unit, Integration)
    *   Enforce Consistency (Formatting, Linting, Conventions, File Length)
    *   Improve Navigability (File Structure, `__init__.py`)
    *   Manage Configuration Explicitly
3.  **Supporting Infrastructure & Process Interaction (Verify & Utilize)**
    *   CI/CD Pipeline Interaction
    *   Utilize AI-Assisted Workflow Documentation
    *   Facilitate Human Code Review & Testing (Crucial step!)
    *   Manage Context
4.  **Post-Refactoring Evaluation & Maintenance Phase (Report & Assist)**
    *   Report Outcomes vs. Baselines
    *   Update Project Documentation (Final State)
    *   Support Continuous Improvement

## Proposed Refactoring Scope & Prioritized Task List (Phase 5 Internal Refactoring)

*(Based on Phase I analysis and `docs/TASK.md`. Awaiting approval.)*

**Scope:** All Python files within `xy_plotting/`, `koboldcpp/`, `google_ai/`, `shared_utils/`, `clip_utils/`, plus root `__init__.py` and `README.md`.

**Prioritization (Module by Module):** Focus on decomposition, clarity, type hinting, docstrings, logging, error handling. Unit test addition will be considered for core logic.

1.  **XY Plotting (`xy_plotting/`)**
    *   Files: `lora_strength_plot_node.py`, `grid_assembly.py`, `__init__.py`.
    *   Tasks: Decompose `lora_strength_plot_node.py`, move grid logic to `grid_assembly.py`, add type hints, docstrings, logging, error handling.
2.  **KoboldCpp (`koboldcpp/`)**
    *   Files: `process_manager.py`, `launcher_node.py`, `api_connector_node.py`, `__init__.py`.
    *   Tasks: Decompose `process_manager.py`, add type hints, docstrings, logging, error handling to all main files.
3.  **Google AI (`google_ai/`)**
    *   Files: `gemini_api_node.py`, `__init__.py`.
    *   Tasks: Decompose `gemini_api_node.py`, add type hints, docstrings, logging, error handling.
4.  **Shared Utils (`shared_utils/`)**
    *   Files: `image_conversion.py`, `console_io.py`, `__init__.py`.
    *   Tasks: Add/improve type hints, docstrings, logging.
5.  **CLIP Utils (`clip_utils/`)**
    *   Files: `token_counter_node.py`, `__init__.py`.
    *   Tasks: Add/improve type hints, docstrings, logging.
6.  **Project Files & Finalization**
    *   Files: Root `__init__.py`, `README.md`.
    *   Tasks: Update root `__init__.py` if needed, enhance `README.md`, create example workflows (`examples/`), add linter/formatter configs (`pyproject.toml`?).

## Incremental Refactoring Strategy

*(Based on Phase I analysis. Awaiting approval.)*

1.  **Module-by-Module Processing:** Refactor modules sequentially according to the prioritization list above (XY Plotting -> KoboldCpp -> Google AI -> Shared Utils -> CLIP Utils -> Project Files).
2.  **File-Focused Iteration:** Within each module, refactor the primary Python files one by one.
3.  **Core Task Application:** Apply core refactoring tasks iteratively (Decomposition, Type Hints, Docstrings, Logging, Error Handling, Line Limit).
4.  **Review & Testing Cycle:** After completing a module, present changes for human review and **await confirmation of successful functional testing in ComfyUI** before proceeding. Log outcomes.
5.  **Version Control:** Use atomic commits with clear messages.

## Coding Standards & Conventions

*(To be refined, incorporating `docs/CODING_STANDARDS.md` where applicable)*

*   **Maximum File Length:** No `.py` or `.js` file shall exceed **500 lines** (User confirmed). Files exceeding this limit must be refactored and split.
*   Python: PEP 8 (using Black/Ruff where possible, as per `docs/CODING_STANDARDS.md`).
*   JavaScript: (Specify if JS exists and standards are known, e.g., Prettier/ESLint)
*   Naming Conventions: Clear, descriptive names for variables, functions, classes, node mappings, categories (`folder/NodeName` where applicable).
*   Type Hinting: Comprehensive Python type hints are mandatory (as per `docs/CODING_STANDARDS.md`).
*   Docstrings: Detailed docstrings for classes and methods (Python), JSDoc for JS (as per `docs/CODING_STANDARDS.md`).
*   Error Handling: Consistent and informative error handling. Use `logging` module. (Pattern TBD, guided by `docs/CODING_STANDARDS.md`).
*   Modularity: Adhere to the Single Responsibility Principle (SRP). Extract reusable logic.
*   Git Commits: Atomic commits with conventional messages.

## Testing Strategy

*(To be populated/refined during Phase I/II)*

*   **Baseline:** No existing automated tests (unit or integration) were found in the project structure (Checked on [Date]). Test coverage is effectively 0%. Improving testability and adding tests (especially unit tests for core logic) will be a goal during refactoring (Phase II).
*   **Unit Tests:** Plan: Write/improve unit tests (`unittest`/`pytest`) for core Python logic (node `FUNCTION`, helpers, utils). Mock inputs.
*   **Integration Tests:** Plan: Identify/create simple test workflows (`.json`). Explore automation possibilities (ComfyUI API/headless).
*   **Manual Testing:** **Crucial:** All changes must be tested functionally within the ComfyUI interface by a human reviewer before final integration.

## Key Decisions Log

*(To be populated as decisions are made)*

*   [Date] - Initialized project documentation file `COMFY_REFACTOR_PROJECT.md`.
*   [Date] - Confirmed high-priority AI objective: Creating/Updating example workflow JSON files.
*   [Date] - Read existing documentation (`docs/ARCHITECTURE.md`, `docs/CODING_STANDARDS.md`, `docs/TASK.md`). Confirmed current task aligns with Phase 5 of `docs/TASK.md`.
*   [Date] - Confirmed adherence to **500-line limit** per user instruction, overriding the 400-line limit in `docs/CODING_STANDARDS.md`.
*   [Date] - Verified necessary tooling: Python (3.12.8) and Git (2.47.1.windows.1) are available.

## Known Issues & Technical Debt Log

*(Populated during Phase I baseline analysis)*

*   **Lack of Automated Tests:** No unit or integration tests found. This increases the risk of regressions during refactoring. (Identified on [Date], aligns with `docs/TASK.md` Phase 5 goals).
*   **Linters/Formatters Not Installed:** `requirements.txt` does not include `ruff`, `flake8`, or `black`, although they are recommended in `docs/CODING_STANDARDS.md`. Suggest adding `ruff` and `black` during Phase II/V to enforce standards. (Identified on [Date]).
*   **File Lengths:** All `.py` files are currently under the 500-line limit. The longest are `xy_plotting/lora_strength_plot_node.py` (487 lines) and `koboldcpp/process_manager.py` (433 lines). These are candidates for decomposition to improve modularity, even though they don't strictly violate the limit. (Checked on [Date]).
*   **JavaScript:** No `.js` files found in the project. (Checked on [Date]).

## Resource Links

*   ComfyUI Custom Nodes Overview: [https://docs.comfy.org/custom-nodes/overview](https://docs.comfy.org/custom-nodes/overview)
*   Getting Started Walkthrough: [https://docs.comfy.org/custom-nodes/walkthrough](https://docs.comfy.org/custom-nodes/walkthrough)
*   Backend - Server Overview/Properties: [https://docs.comfy.org/custom-nodes/backend/server_overview](https://docs.comfy.org/custom-nodes/backend/server_overview)
*   Backend - Lifecycle: [https://docs.comfy.org/custom-nodes/backend/lifecycle](https://docs.comfy.org/custom-nodes/backend/lifecycle)
*   Backend - Datatypes: [https://docs.comfy.org/custom-nodes/backend/datatypes](https://docs.comfy.org/custom-nodes/backend/datatypes)
*   Backend - Images and Masks: [https://docs.comfy.org/custom-nodes/backend/images_and_masks](https://docs.comfy.org/custom-nodes/backend/images_and_masks)
*   Backend - More on Inputs: [https://docs.comfy.org/custom-nodes/backend/more_on_inputs](https://docs.comfy.org/custom-nodes/backend/more_on_inputs)
*   Backend - Lazy Evaluation: [https://docs.comfy.org/custom-nodes/backend/lazy_evaluation](https://docs.comfy.org/custom-nodes/backend/lazy_evaluation)
*   Backend - Expansion: [https://docs.comfy.org/custom-nodes/backend/expansion](https://docs.comfy.org/custom-nodes/backend/expansion)
*   Backend - Lists: [https://docs.comfy.org/custom-nodes/backend/lists](https://docs.comfy.org/custom-nodes/backend/lists)
*   Backend - Snippets: [https://docs.comfy.org/custom-nodes/backend/snippets](https://docs.comfy.org/custom-nodes/backend/snippets)
*   Backend - Tensors: [https://docs.comfy.org/custom-nodes/backend/tensors](https://docs.comfy.org/custom-nodes/backend/tensors)
*   JS - Overview: [https://docs.comfy.org/custom-nodes/js/javascript_overview](https://docs.comfy.org/custom-nodes/js/javascript_overview)
*   JS - Hooks: [https://docs.comfy.org/custom-nodes/js/javascript_hooks](https://docs.comfy.org/custom-nodes/js/javascript_hooks)
*   JS - Objects & Hijacking: [https://docs.comfy.org/custom-nodes/js/javascript_objects_and_hijacking](https://docs.comfy.org/custom-nodes/js/javascript_objects_and_hijacking)
*   JS - Settings: [https://docs.comfy.org/custom-nodes/js/javascript_settings](https://docs.comfy.org/custom-nodes/js/javascript_settings)
*   JS - Examples: [https://docs.comfy.org/custom-nodes/js/javascript_examples](https://docs.comfy.org/custom-nodes/js/javascript_examples)
*   Workflow Templates: [https://docs.comfy.org/custom-nodes/workflow_templates](https://docs.comfy.org/custom-nodes/workflow_templates)
*   Registry - Standards: [https://docs.comfy.org/registry/standards](https://docs.comfy.org/registry/standards)
*   Registry - CI/CD: [https://docs.comfy.org/registry/cicd](https://docs.comfy.org/registry/cicd)
*   Registry - Specifications (`pyproject.toml`): [https://docs.comfy.org/registry/specifications](https://docs.comfy.org/registry/specifications)
