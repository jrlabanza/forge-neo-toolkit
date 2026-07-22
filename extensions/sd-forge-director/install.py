"""Auto-install rembg (background removal) into the venv on launch."""
import launch

if not launch.is_installed("rembg"):
    try:
        launch.run_pip("install rembg", "rembg for sd-forge-director")
    except Exception as exc:
        print(f"[director] rembg install failed (Remove BG will be disabled): {exc}")
