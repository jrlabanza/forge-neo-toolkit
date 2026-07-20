"""sd-forge-reference-image: NAI-style Vibe / Precise reference for txt2img.

Adds an accordion in txt2img/img2img. Drop image, pick mode, hit Generate.
Hot-swaps SageAttention -> PyTorch SDP during the generation (avoids the
black-image NaN with IPAdapter on Illustrious), then restores afterward.
"""
from __future__ import annotations
import logging
import sys
import traceback
from pathlib import Path

import gradio as gr

try:
    from modules import scripts
except ImportError:
    scripts = None

logger = logging.getLogger(__name__)
TAG = "[reference-image]"

MODE_OFF = "Off"
MODE_VIBE = "Vibe (style + composition)"
MODE_PRECISE = "Precise (face / character)"

MODE_TO_FILENAME = {
    MODE_VIBE: "ip-adapter-plus_sdxl_vit-h.safetensors",
    MODE_PRECISE: "ip-adapter-plus-face_sdxl_vit-h.safetensors",
}


def _find_ipadapter_model(filename):
    try:
        from modules import paths
        models_path = Path(paths.models_path)
    except Exception:
        models_path = Path(__file__).resolve().parents[3] / "models"
    for c in [
        models_path / "controlnet" / "IpAdapter" / filename,
        models_path / "controlnet" / filename,
        models_path / "IpAdapter" / filename,
    ]:
        if c.exists():
            return str(c)
    return None


def _find_clipvision_model():
    try:
        from modules_forge.shared import preprocessor_dir
        p = Path(preprocessor_dir) / "CLIP-ViT-H-14.safetensors"
        if p.exists():
            return str(p)
    except Exception:
        pass
    try:
        from modules import paths
        p = Path(paths.models_path) / "ControlNetPreprocessor" / "CLIP-ViT-H-14.safetensors"
        if p.exists():
            return str(p)
    except Exception:
        pass
    return None


_clipvision_cache = {"path": None, "model": None}


def _load_clipvision():
    path = _find_clipvision_model()
    if not path:
        raise FileNotFoundError(
            "CLIP-ViT-H-14.safetensors not found. Run any IPAdapter generation "
            "via the standard ControlNet panel once to trigger Forge's auto-download."
        )
    if _clipvision_cache["path"] == path:
        return _clipvision_cache["model"]
    from backend.patcher.clipvision import load as load_clipvision
    cv = load_clipvision(path)
    _clipvision_cache.update({"path": path, "model": cv})
    logger.info(f"{TAG} loaded CLIP-ViT-H from {path}")
    return cv


_ipa_cache = {}


def _load_ipadapter_state(model_path):
    if model_path in _ipa_cache:
        return _ipa_cache[model_path]
    if model_path.lower().endswith(".safetensors"):
        import safetensors.torch
        raw = safetensors.torch.load_file(model_path)
    else:
        import torch
        raw = torch.load(model_path, map_location="cpu", weights_only=False)
    state = {"image_proj": {}, "ip_adapter": {}}
    for k, v in raw.items():
        if k.startswith("image_proj."):
            state["image_proj"][k[len("image_proj."):]] = v
        elif k.startswith("ip_adapter."):
            state["ip_adapter"][k[len("ip_adapter."):]] = v
    if not state["ip_adapter"]:
        raise RuntimeError(f"No ip_adapter weights in {model_path}")
    _ipa_cache[model_path] = state
    logger.info(f"{TAG} loaded IPAdapter from {model_path}")
    return state


def _get_opIPAdapterApply():
    import importlib
    try:
        return importlib.import_module("scripts.forge_ipadapter").opIPAdapterApply
    except ImportError:
        ip = Path(__file__).resolve().parents[3] / "extensions-builtin" / "sd_forge_ipadapter" / "scripts"
        if str(ip) not in sys.path:
            sys.path.insert(0, str(ip))
        import importlib
        return importlib.import_module("forge_ipadapter").opIPAdapterApply


def _swap_attention_to_pytorch(p):
    """Force PyTorch attention. SageAttention 2 + IPAdapter Plus + Illustrious
    produces NaN latents (black image). Stash originals on p for postprocess
    to restore.

    NOTE: backend/nn/*.py do `from backend.attention import attention_function`
    at import time, which captures a *local* reference inside each NN module.
    Rebinding only `backend.attention.attention_function` therefore leaves
    those captured references pointing at Sage, and Sage keeps running through
    cross-attention -> NaN. We have to walk sys.modules and rebind every
    captured copy. Same logic for the VAE variant `attention_function_vae`.
    """
    try:
        from backend import attention as a
        sage = getattr(a, "attention_sage", None)
        pyt = getattr(a, "attention_pytorch", None)
        if sage is None or pyt is None:
            return
        if a.attention_function is not sage:
            # Already on a safe backend (PyTorch/xformers/flash). Nothing to do.
            return

        # Optional VAE swap target. Some Sage builds also patch the VAE path;
        # if attention_function_vae also points at a Sage function we swap it.
        sage_vae = getattr(a, "attention_function_vae", None)
        pyt_vae = pyt  # PyTorch SDP is safe for VAE too
        swap_vae = sage_vae is not None and sage_vae is sage

        saved = {"main": [], "vae": []}

        # Rebind the canonical binding first.
        saved["main"].append((a, "attention_function", a.attention_function))
        a.attention_function = pyt
        if swap_vae:
            saved["vae"].append((a, "attention_function_vae", a.attention_function_vae))
            a.attention_function_vae = pyt_vae

        # Walk all currently-loaded modules and rebind every captured copy.
        # We iterate over a snapshot because importing during iteration would
        # mutate sys.modules.
        for mod_name, mod in list(sys.modules.items()):
            if mod is None or mod is a:
                continue
            # attention_function (main UNet/transformer path)
            fn = getattr(mod, "attention_function", None)
            if fn is sage:
                saved["main"].append((mod, "attention_function", fn))
                try:
                    setattr(mod, "attention_function", pyt)
                except Exception:
                    pass
            # attention_function_vae (VAE path) - only swap if it was Sage too
            if swap_vae:
                vfn = getattr(mod, "attention_function_vae", None)
                if vfn is sage:
                    saved["vae"].append((mod, "attention_function_vae", vfn))
                    try:
                        setattr(mod, "attention_function_vae", pyt_vae)
                    except Exception:
                        pass

        p._ref_img_saved_attention = saved
        logger.info(
            f"{TAG} swapped Sage->PyTorch attention "
            f"(main={len(saved['main'])} sites, vae={len(saved['vae'])} sites)"
        )
    except Exception as e:
        logger.warning(f"{TAG} attention swap skipped: {e}")


def _restore_attention(p):
    saved = getattr(p, "_ref_img_saved_attention", None)
    if saved is None:
        return
    try:
        # Restore in reverse insertion order so the canonical binding ends up
        # equal to what we originally saw.
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
        logger.info(f"{TAG} restored attention after gen")
    except Exception as e:
        logger.warning(f"{TAG} restore failed: {e}")
    try:
        delattr(p, "_ref_img_saved_attention")
    except Exception:
        pass


class ReferenceImageScript(scripts.Script if scripts else object):
    def title(self):
        return "Reference Image (NAI-style)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if scripts else True

    def ui(self, is_img2img):
        with gr.Accordion("\U0001F4F8 Reference Image (NovelAI-style)", open=False):
            gr.Markdown(
                "Drop a reference, pick mode, set strength. "
                "IPAdapter is applied automatically with PyTorch attention "
                "(Sage is hot-swapped to avoid NaN; restored after the gen)."
            )
            with gr.Row():
                ref_image = gr.Image(label="Reference image", type="numpy", height=240)
                with gr.Column():
                    mode = gr.Radio(
                        choices=[MODE_OFF, MODE_VIBE, MODE_PRECISE],
                        value=MODE_OFF,
                        label="Mode",
                    )
                    strength = gr.Slider(
                        minimum=0.0, maximum=1.5, step=0.05, value=0.7,
                        label="Reference strength",
                    )
        return [ref_image, mode, strength]

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) < 3:
            return
        ref_image, mode, strength = args[0], args[1], args[2]
        if mode == MODE_OFF or ref_image is None or strength <= 0.0:
            return

        _swap_attention_to_pytorch(p)

        try:
            filename = MODE_TO_FILENAME.get(mode)
            if not filename:
                logger.warning(f"{TAG} unknown mode: {mode!r}")
                return
            model_path = _find_ipadapter_model(filename)
            if not model_path:
                logger.error(f"{TAG} cannot find model {filename}")
                return

            ipadapter = _load_ipadapter_state(model_path)
            clip_vision = _load_clipvision()

            from modules_forge.utils import numpy_to_pytorch
            cond = dict(
                clip_vision=clip_vision,
                image=numpy_to_pytorch(ref_image),
                weight_type="original",
                noise=0.0,
                embeds=None,
                unfold_batch=False,
            )

            opApply = _get_opIPAdapterApply()
            unet = p.sd_model.forge_objects.unet
            (unet,) = opApply(
                ipadapter=ipadapter,
                model=unet,
                weight=float(strength),
                start_at=0.0,
                end_at=1.0,
                faceid_v2=False,
                weight_v2=False,
                attn_mask=None,
                **cond,
            )
            p.sd_model.forge_objects.unet = unet

            try:
                p.extra_generation_params["RefImg Mode"] = mode
                p.extra_generation_params["RefImg Strength"] = strength
                p.extra_generation_params["RefImg Model"] = filename
            except Exception:
                pass

            logger.info(f"{TAG} applied mode={mode!r} strength={strength} model={filename}")

        except Exception as exc:
            logger.error(f"{TAG} apply failed: {exc}\n{traceback.format_exc()}")

    def postprocess(self, p, processed, *args, **kwargs):
        _restore_attention(p)
