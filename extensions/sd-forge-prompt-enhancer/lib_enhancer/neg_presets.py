"""Curated negative-prompt presets.

Each preset is a name -> string mapping. Users pick a name from a Gradio
Dropdown and the string fills the negative-prompt box.

The defaults are tuned for Illustrious / NoobAI XL and similar
Danbooru-trained SDXL checkpoints.
"""

NEGATIVE_PRESETS = {
    "(none)": "",

    "Illustrious — recommended default":
        "lowres, worst quality, bad quality, bad anatomy, bad hands, "
        "missing fingers, extra digits, watermark, signature, artist name, "
        "username, text, jpeg artifacts, blurry, normal quality",

    "Illustrious — strict (anti-deformed)":
        "lowres, worst quality, bad quality, bad anatomy, bad hands, "
        "bad proportions, bad perspective, deformed, mutated, disfigured, "
        "missing fingers, extra digits, extra limbs, missing limbs, "
        "fused fingers, fused hands, malformed limbs, watermark, signature, "
        "artist name, username, text, jpeg artifacts, blurry, "
        "normal quality, ugly",

    "Eyes-focus (for ADetailer 2nd-pass)":
        "blurry eyes, washed out eyes, color bleeding, asymmetric eyes, "
        "muddy eyes, low detail eyes, cross-eyed, lazy eye, "
        "different colored eyes, deformed eyes",

    "Hands-focus":
        "bad hands, missing fingers, extra digits, fused fingers, "
        "malformed hands, deformed hands, wrong hand anatomy, "
        "extra fingers, missing fingers",

    "Anatomy-strict (no body distortions)":
        "bad anatomy, bad proportions, mutated, deformed, disfigured, "
        "extra limbs, missing limbs, fused limbs, malformed body, "
        "long neck, short neck, extra head, extra arms, extra legs, "
        "wrong perspective, bent perspective",

    "Style cleanup (remove jpeg + watermark)":
        "jpeg artifacts, watermark, signature, artist name, username, "
        "text, logo, copyright, sample watermark",

    "Film-grain kill":
        "film grain, grainy, noise, noisy, sensor noise, motion blur, "
        "chromatic aberration, lens flare",

    "Cartoon / sketch kill (force painted)":
        "sketch, line art, monochrome, lineart, simple background, "
        "flat color, cel shading, low detail, unfinished",

    "Background simplify":
        "complex background, busy background, cluttered background, "
        "many objects, distracting background",

    "SFW gatekeeper":
        "nude, naked, topless, nipples, pussy, penis, sex, explicit, "
        "questionable, nsfw",
}


def get(name):
    return NEGATIVE_PRESETS.get(name, "")


def list_names():
    return list(NEGATIVE_PRESETS.keys())
