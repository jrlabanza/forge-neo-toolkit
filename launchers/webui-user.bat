@echo off

:: ============================================================================
:: Optimized for: RTX 3070 Laptop GPU (8 GB VRAM) + 40 GB system RAM
:: ============================================================================
::
:: Reasoning:
::   --reserve-vram 2      Forge will keep 2 GB VRAM free during sampling.
::                         Without this, free VRAM dropped to ~1.4 GB and
::                         Forge spammed "below safe threshold" warnings every
::                         step. 2 GB is the recommended buffer for 8 GB cards
::                         running SDXL.
::   --pin-shared-memory   CPU<->GPU transfers go through pinned memory pages
::                         instead of pageable ones - measurably faster model
::                         swap on 8 GB cards since UNet, text encoder and VAE
::                         get juggled.
::   --cuda-malloc         async CUDA allocator. Fewer fragmentation stalls
::                         than the default sync allocator.
::   --cuda-stream         use a dedicated CUDA stream for weight offload.
::   --skip-python-version-check  silences the PyTorch 2.10 warning.
::   --disable-gpu-warning silences the per-step "free memory low" spam.
::                         (The actual headroom is now fine thanks to
::                         --reserve-vram 2.)
::   --gradio-allowed-path  StabilityMatrix sets this; preserved for compat.
::
:: NOT used (and why):
::   --xformers           SageAttention 2 is already active and is faster
::                        than xformers on Ampere. Don't enable both.
::   --medvram --lowvram  Not needed at 8 GB VRAM with the reserves above.
::   --no-half            SDXL needs fp16 for VRAM fit; --no-half would OOM.
::
:: Use the venv's python directly - this machine has no system python
:: (Stability Matrix manages its own), and webui.bat's first check would
:: otherwise die on the bare "python" Store alias before reaching the venv.
cd /D "%~dp0"
set PYTHON="%~dp0venv\Scripts\python.exe"

:: GitPython inside Forge needs a git executable; none is on PATH, so point
:: it at the PortableGit that lives in this folder (webui.bat exports it as
:: GIT_PYTHON_GIT_EXECUTABLE).
if exist "%~dp0PortableGit\cmd\git.exe" set "GIT=%~dp0PortableGit\cmd\git.exe"

:: set VENV_DIR=

:: --api: enables the local API on 127.0.0.1 (not exposed to the network).
::        Required by the Job Runner tab (queue / batch re-process / test cards).
set COMMANDLINE_ARGS=--reserve-vram 2 --pin-shared-memory --cuda-malloc --cuda-stream --skip-python-version-check --skip-version-check --disable-gpu-warning --api

:: --xformers --sage --uv
:: --skip-torch-cuda-test --skip-version-check --skip-prepare-environment --skip-install

call webui.bat
