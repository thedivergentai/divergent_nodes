# Coding Standards

This document outlines the coding standards to be followed for this ComfyUI custom node project to ensure consistency, readability, and maintainability, especially for AI-assisted development.

## General Principles

*   **Clarity over Brevity:** Write code that is easy to understand, even if slightly more verbose.
*   **Consistency:** Adhere to the standards outlined in this document throughout the codebase.
*   **Modularity:** Break down complex logic into smaller, reusable functions or classes.
*   **Explicitness:** Avoid ambiguity. Use clear names and type hints.
*   **Simplicity:** Prefer simple solutions over complex ones where possible.
*   **DRY (Don't Repeat Yourself):** Avoid code duplication. Utilize shared utilities or helper functions.

## File Structure & Naming

*   **Directories:** Group related nodes or utilities into subdirectories (e.g., `kobold_cpp/`, `shared_utils/`).
*   **Filenames:** Use descriptive, lowercase names with underscores (`snake_case.py`). Node implementation files should clearly indicate the node they contain (e.g., `launcher_node.py`, `token_counter_node.py`).
*   **`__init__.py`:** Each subdirectory acting as a Python package must contain an `__init__.py` file. This file should typically import and expose the necessary classes or functions (like `NODE_CLASS_MAPPINGS`) from modules within that directory.

## Code Formatting & Linting

*   **PEP 8:** Follow the standard Python style guide (PEP 8).
*   **Formatter:** Use `black` for consistent code formatting.
*   **Linter:** Use `ruff` (or `flake8`) for identifying style issues and potential errors.
*   **Configuration:** Linter/formatter configurations (e.g., in `pyproject.toml`) should be used if provided or created to enforce standards automatically.

## Naming Conventions

*   **Variables & Functions:** `snake_case` (lowercase with underscores).
*   **Classes:** `PascalCase` (or `CamelCase`).
*   **Constants:** `UPPER_SNAKE_CASE`.
*   **ComfyUI Node Classes:** Use descriptive `PascalCase` names (e.g., `KoboldCppLauncherNode`).
*   **ComfyUI Mappings:** Use the exact names `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`. Keys in these mappings should be descriptive strings, often matching the class name for `NODE_CLASS_MAPPINGS`.
*   **ComfyUI Node Attributes:** Use the standard uppercase names (`INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, `CATEGORY`, `OUTPUT_NODE`, etc.).

## Modularity & File Length

*   **Maximum File Length:** No Python file should exceed **500 lines** of code (excluding comments and blank lines where reasonable).
*   **Single Responsibility Principle (SRP):** Functions and classes should ideally have one primary responsibility.
*   **Decomposition:** Break down long functions (`execute` methods, complex helpers) into smaller, well-named private helper methods (`_helper_method`) or move logic to separate utility modules/classes.

## Type Hinting

*   **Mandatory:** Add comprehensive Python type hints (`typing` module) to **all** function/method signatures (parameters and return values) and significant variable declarations.
*   **Clarity:** Use specific types where possible (e.g., `List[str]`, `Optional[torch.Tensor]`, `Dict[str, Any]`).
*   **ComfyUI Types:** Use the correct type strings within `INPUT_TYPES` and `RETURN_TYPES` tuples (e.g., `"STRING"`, `"IMAGE"`, `"INT"`).

## Documentation & Comments

*   **Docstrings:**
    *   Provide clear, concise docstrings for all modules, classes, functions, and methods using standard formats (e.g., Google style, reStructuredText).
    *   Explain the purpose, arguments (`Args:`), return values (`Returns:`), and any raised exceptions (`Raises:`).
    *   For ComfyUI nodes, class docstrings should explain the node's overall purpose and usage. `execute` method docstrings should detail the core logic.
*   **Comments:**
    *   Use comments (`#`) sparingly to explain *why* something is done, not *what* is being done (the code should explain the 'what').
    *   Avoid redundant comments that merely restate the code.
    *   Remove commented-out code blocks. Use version control (Git) for history.
    *   **Consistency:** Apply a consistent commenting style. Avoid metacommentary about the development process itself within the code.

## Error Handling & Logging

*   **Logging:** Use the standard `logging` module instead of `print()` for debugging, warnings, and errors. Configure a basic logger if needed.
*   **Exceptions:** Raise specific, informative exceptions where appropriate (e.g., `ValueError`, `FileNotFoundError`, custom exceptions). Avoid catching generic `Exception` unless necessary, and handle specific errors first.
*   **User Feedback:** For errors that should be surfaced to the user via the ComfyUI node output, return a clear error message string prefixed with "ERROR: ". Log the full traceback/details to the console for debugging.

## ComfyUI Specifics

*   **Node Structure:** Adhere strictly to the required ComfyUI node structure (`INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, `CATEGORY`, mappings).
*   **Category Naming:** All node `CATEGORY` attributes **must** be prefixed with `"Divergent Nodes 👽/"` (e.g., `"Divergent Nodes 👽/Utilities"`).
*   **Imports:** Use relative imports (`from .module import Class`) within packages where appropriate.
*   **Dependencies:** Explicitly list all external Python dependencies in `requirements.txt`.

## Testing

*   **Unit Tests:** Write unit tests (using `unittest` or `pytest`) for core logic, helper functions, and utilities where feasible.
*   **Mocking:** Use mocking libraries (`unittest.mock`) to isolate components and test interactions with external services (APIs, file system, ComfyUI internals) without requiring the actual dependencies.
*   **Integration:** While unit tests are preferred, acknowledge that full integration testing often requires running workflows within ComfyUI itself (manually or via tools like `Comfy-Action`).
