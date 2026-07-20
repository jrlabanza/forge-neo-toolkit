"""
sd-forge-ipadapter-compat
=========================

Three runtime fixes/features for IPAdapter on Forge Neo:

1. Patch `ControlNet.get_control` / `T2IAdapter.get_control` so they no longer
   crash with `AttributeError: 'dict' object has no attribute 'shape'` when the
   conditioning hint is the IPAdapter dict (clip_vision / image / embeds / ...).

2. Hook `IPAdapterPatcher.process_before_every_sampling` and
   `process_after_every_sampling` to temporarily swap `attention_function` from
   SageAttention to PyTorch SDP while IPAdapter is active, then restore Sage
   afterward. (Stability Matrix `--disable-sage` is currently a safer fallback;
   this hook is best-effort.)

3. Register a new preprocessor "IP-Adapter Face (Auto-Crop)" that uses
   InsightFace to detect the largest face in the reference image, crops with
   padding, then encodes with CLIP-ViT-H. Designed for use with
   `ip-adapter-plus-face_sdxl_vit-h` -- produces dramatically better face
   transfer than the generic CLIP-ViT-H preprocessor.

Author: applied by Claude on 2026-05-26
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)
TAG = "[ipadapter-compat]"


# ---------------------------------------------------------------------------
# Fix 1: get_control dict guard
# ---------------------------------------------------------------------------
def _install_get_control_patch():
    try:
        from backend.patcher import controlnet as cn_module
    except ImportError as e:
        logger.error(f"{TAG} could not import backend.patcher.controlnet: {e}")
        return False

    if getattr(cn_module, "_ipadapter_compat_patched", False):
        return True

    def _make_patched(orig_fn):
        def patched(self, x_noisy, t, cond, batched_number):
            if isinstance(self.cond_hint_original, dict):
                if self.previous_controlnet is not None:
                    return self.previous_controlnet.get_control(
                        x_noisy, t, cond, batched_number
                    )
                return None
            return orig_fn(self, x_noisy, t, cond, batched_number)
        return patched

    patched_count = 0
    for cls_name in ("ControlNet", "T2IAdapter"):
        cls = getattr(cn_module, cls_name, None)
        if cls is None:
            continue
        cls.get_control = _make_patched(cls.get_control)
        patched_count += 1

    cn_module._ipadapter_compat_patched = True
    logger.info(f"{TAG} patched get_control on {patched_count} class(es)")
    return True


# ---------------------------------------------------------------------------
# Fix 2: auto-swap attention function while IPAdapter is active
# ---------------------------------------------------------------------------
# backend/nn/*.py do `from backend.attention import attention_function` at
# import time, which captures a *local* reference to the function object
# inside each NN module. Rebinding only `backend.attention.attention_function`
# leaves all of those captured copies pointing at Sage, so Sage keeps running
# in cross-attention and IPAdapter Plus on Illustrious produces NaN -> black
# image. We must walk sys.modules and rebind every captured copy.
_attention_state = {"saved_sites": None, "active": False}


def _enter_ipadapter_mode():
    try:
        from backend import attention as attn_module
    except ImportError:
        return
    sage_fn = getattr(attn_module, "attention_sage", None)
    pytorch_fn = getattr(attn_module, "attention_pytorch", None)
    if sage_fn is None or pytorch_fn is None:
        return
    if attn_module.attention_function is not sage_fn:
        # Already on a safe backend (PyTorch/xformers/flash). Nothing to do.
        return

    sage_vae_fn = getattr(attn_module, "attention_function_vae", None)
    swap_vae = sage_vae_fn is not None and sage_vae_fn is sage_fn

    saved = {"main": [], "vae": []}

    # Canonical binding first.
    saved["main"].append((attn_module, "attention_function", attn_module.attention_function))
    attn_module.attention_function = pytorch_fn
    if swap_vae:
        saved["vae"].append((attn_module, "attention_function_vae", attn_module.attention_function_vae))
        attn_module.attention_function_vae = pytorch_fn

    # Walk every loaded module and rebind copies captured by
    # `from backend.attention import attention_function` at import time.
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or mod is attn_module:
            continue
        fn = getattr(mod, "attention_function", None)
        if fn is sage_fn:
            saved["main"].append((mod, "attention_function", fn))
            try:
                setattr(mod, "attention_function", pytorch_fn)
            except Exception:
                pass
        if swap_vae:
            vfn = getattr(mod, "attention_function_vae", None)
            if vfn is sage_fn:
                saved["vae"].append((mod, "attention_function_vae", vfn))
                try:
                    setattr(mod, "attention_function_vae", pytorch_fn)
                except Exception:
                    pass

    _attention_state["saved_sites"] = saved
    _attention_state["active"] = True
    logger.info(
        f"{TAG} IPAdapter active - swapped Sage->PyTorch "
        f"(main={len(saved['main'])} sites, vae={len(saved['vae'])} sites)"
    )


def _exit_ipadapter_mode():
    if not _attention_state["active"]:
        return
    saved = _attention_state.get("saved_sites") or {}
    for mod, name, orig in reversed(saved.get("main", [])):
        try:
            setattr(mod, name, orig)
        except Exception:
            pass
    for mod, name, orig in reversed(saved.get("vae", [])):
        try:
            setattr(mod, name, orig)
        except Exception:
            pass
    _attention_state["saved_sites"] = None
    _attention_state["active"] = False
    logger.info(f"{TAG} IPAdapter inactive - restored Sage attention")


def _resolve_ipadapter_module():
    """Return the imported forge_ipadapter module (built-in extension)."""
    import importlib
    try:
        return importlib.import_module("scripts.forge_ipadapter")
    except ImportError:
        pass
    forge_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    ip_scripts = os.path.join(
        forge_root, "extensions-builtin", "sd_forge_ipadapter", "scripts"
    )
    if ip_scripts not in sys.path:
        sys.path.insert(0, ip_scripts)
    return importlib.import_module("forge_ipadapter")


def _install_ipadapter_hooks():
    try:
        mod = _resolve_ipadapter_module()
    except Exception as e:
        logger.warning(f"{TAG} could not import sd_forge_ipadapter: {e}")
        return False

    IPAdapterPatcher = getattr(mod, "IPAdapterPatcher", None)
    if IPAdapterPatcher is None:
        logger.warning(f"{TAG} IPAdapterPatcher class not found")
        return False

    if getattr(IPAdapterPatcher, "_compat_hooks_installed", False):
        return True

    orig_before = IPAdapterPatcher.process_before_every_sampling
    orig_after = IPAdapterPatcher.process_after_every_sampling

    def patched_before(self, process, cond, mask, *args, **kwargs):
        _enter_ipadapter_mode()
        return orig_before(self, process, cond, mask, *args, **kwargs)

    def patched_after(self, process, params, *args, **kwargs):
        try:
            return orig_after(self, process, params, *args, **kwargs)
        finally:
            _exit_ipadapter_mode()

    IPAdapterPatcher.process_before_every_sampling = patched_before
    IPAdapterPatcher.process_after_every_sampling = patched_after
    IPAdapterPatcher._compat_hooks_installed = True
    logger.info(f"{TAG} IPAdapter lifecycle hooks installed")
    return True


# ---------------------------------------------------------------------------
# Feature 3: IP-Adapter Face (Auto-Crop) preprocessor
# ---------------------------------------------------------------------------
def _install_face_preprocessor():
    try:
        mod = _resolve_ipadapter_module()
    except Exception as e:
        logger.warning(f"{TAG} face preprocessor: import failed: {e}")
        return False

    Base = getattr(mod, "PreprocessorClipVisionWithInsightFaceForIPAdapter", None)
    if Base is None:
        logger.warning(f"{TAG} face preprocessor: base class not found")
        return False

    try:
        from modules_forge.shared import add_supported_preprocessor
        from modules_forge.utils import numpy_to_pytorch
    except ImportError as e:
        logger.warning(f"{TAG} face preprocessor: Forge plumbing missing: {e}")
        return False

    class PreprocessorIPAdapterFaceCrop(Base):
        """Detect the largest face, crop with ~35% padding, then encode with
        CLIP-ViT-H. Pair with ip-adapter-plus-face_sdxl_vit-h for best results.
        """

        def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
            cropped = input_image
            try:
                insightface = self.load_insightface()
                bgr = input_image[..., ::-1] if input_image.shape[-1] == 3 else input_image
                faces = insightface.get(bgr)
                if faces:
                    def _area(f):
                        b = f.bbox
                        return (b[2] - b[0]) * (b[3] - b[1])
                    face = max(faces, key=_area)
                    x1, y1, x2, y2 = [int(v) for v in face.bbox]
                    w = x2 - x1
                    h = y2 - y1
                    pad_x = int(w * 0.35)
                    pad_y = int(h * 0.35)
                    img_h, img_w = input_image.shape[:2]
                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(img_w, x2 + pad_x)
                    y2 = min(img_h, y2 + pad_y)
                    cropped = input_image[y1:y2, x1:x2]
                    logger.info(
                        f"{TAG} face-crop: cropped to {cropped.shape[1]}x{cropped.shape[0]} from {img_w}x{img_h}"
                    )
                else:
                    logger.info(f"{TAG} face-crop: no face detected, using full image")
            except Exception as e:
                logger.warning(f"{TAG} face-crop failed ({e}), using full image")

            return dict(
                clip_vision=self.load_clipvision(),
                image=numpy_to_pytorch(cropped),
                weight_type="original",
                noise=0.0,
                embeds=None,
                unfold_batch=False,
            )

    preprocessor = PreprocessorIPAdapterFaceCrop(
        name="IP-Adapter Face (Auto-Crop)",
        url="https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors",
        filename="CLIP-ViT-H-14.safetensors",
    )
    add_supported_preprocessor(preprocessor)
    logger.info(f"{TAG} 'IP-Adapter Face (Auto-Crop)' preprocessor registered")
    return True


# ---------------------------------------------------------------------------
# Apply patches
# ---------------------------------------------------------------------------
_install_get_control_patch()

try:
    from modules import script_callbacks as _sc
    def _on_app_started(*_a, **_kw):
        _install_ipadapter_hooks()
        _install_face_preprocessor()
    _sc.on_app_started(_on_app_started)
    logger.info(f"{TAG} extension loaded; hooks deferred to app_started")
except Exception as _e:
    _install_ipadapter_hooks()
    _install_face_preprocessor()
    logger.info(f"{TAG} extension loaded (fallback immediate install): {_e}")
