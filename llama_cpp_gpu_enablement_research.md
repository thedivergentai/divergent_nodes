# llama-cpp-python GPU Enablement Research

## Enabling GPU Acceleration

Based on the user's feedback (NVIDIA GPU, CUDA 12.4, PyTorch 2.6.0), the focus shifts to enabling GPU acceleration in `llama-cpp-python`.

**Key Considerations:**

*   **`llama-cpp-python` Compilation:** `llama-cpp-python` needs to be compiled with CUDA support to leverage the NVIDIA GPU. If it was installed from a pre-built wheel that doesn't include CUDA, it will default to CPU.
*   **Environment Variables:** Several environment variables can influence `llama-cpp-python`'s behavior regarding CUDA:
    *   `CMAKE_ARGS="-DLLAMA_CUBLAS=on"`: This is the most crucial flag. It tells `llama-cpp-python`'s build system to enable CUBLAS (CUDA Basic Linear Algebra Subprograms) support during compilation.
    *   `FORCE_CMAKE=1`: Forces a reinstallation using CMAKE.
    *   `CUDA_PATH`:  Should point to the CUDA installation directory (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`).
    *   `PATH`:  The system's `PATH` environment variable should include the `bin` directory within the CUDA installation path (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin`). This allows the system to find the necessary CUDA DLLs.
* **Runtime Flags:**
    * `n_gpu_layers`: This parameter, often passed to the `Llama` constructor in `llama-cpp-python`, controls how many layers of the model are offloaded to the GPU. Setting this to a value greater than 0 is essential for GPU utilization. The optimal value depends on the model size and available GPU memory.

**Troubleshooting:**

*   **PyTorch CUDA Check:**  A simple Python script using `torch.cuda.is_available()` can confirm if PyTorch detects the GPU. If this returns `False`, there's a problem with the PyTorch/CUDA setup, which needs to be addressed before `llama-cpp-python` can use the GPU.
*   **Recompilation:** If `llama-cpp-python` wasn't compiled with CUDA support, it needs to be recompiled. This typically involves uninstalling the existing version, setting the necessary environment variables (`CMAKE_ARGS`, `FORCE_CMAKE`, `CUDA_PATH`, `PATH`), and then reinstalling using `pip`.

**Next Steps (in ACT Mode):**

1.  **Check PyTorch CUDA:** Run a Python script to verify `torch.cuda.is_available()`.
2.  **Check Environment Variables:** Inspect the relevant environment variables (`CUDA_PATH`, `PATH`) to ensure they are correctly set.
3.  **Reinstall `llama-cpp-python` (if needed):** If PyTorch sees the GPU but `llama-cpp-python` still doesn't use it, reinstall `llama-cpp-python` with the necessary environment variables set to enable CUBLAS support.
4.  **Test with `n_gpu_layers`:** When loading the model in `llama-cpp-python`, set the `n_gpu_layers` parameter to offload layers to the GPU.
5. **Update Research:** Update the research document.
