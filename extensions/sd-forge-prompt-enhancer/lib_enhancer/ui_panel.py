"""Gradio UI panel for the Prompt Enhancer "Pro Tools" features.

All the new functionality (token counter, linter, negative presets, prompt
presets, wildcards, LoRA suggest, CN recipe presets, auto-CN suggest, batch
analyzer, image compare, send-to-ADetailer, recipe card export, generation
history, recently-used artists) is built into ONE sub-tab so the existing
Prompt Enhancer UI is untouched.

Public API:
    build_pro_tools_tab(host_ctx) -> gr.Tab
        host_ctx is a dict carrying references to the host's existing
        widgets (positive_out, negative_out, etc.) so the new tools can
        push results back into the main form. Currently optional — the tab
        works standalone.
"""
import json
import os
import gradio as gr

from . import (
    wildcards, tokens, linter, neg_presets, presets,
    lora_suggest, cn_suggest, batch_analyze, image_compare,
    history, artists_recent, send_adetailer, card_export,
)


def _lora_search_roots():
    """Discover candidate LoRA folders to scan."""
    here = os.path.dirname(os.path.abspath(__file__))
    forge_root = os.path.normpath(os.path.join(here, "..", "..", ".."))
    candidates = [
        os.path.join(forge_root, "models", "Lora"),
        os.path.join(forge_root, "models", "lora"),
        os.path.join(forge_root, "models", "LyCORIS"),
    ]
    return [c for c in candidates if os.path.isdir(c)]


def build_pro_tools_blocks():
    """Build a standalone gr.Blocks containing all Pro Tools accordions.
    Caller registers it as its own top-level tab via on_ui_tabs."""
    with gr.Blocks(analytics_enabled=False) as blocks:
        gr.Markdown(
            "## Pro Tools\n"
            "Power-user features layered on top of the main builder. "
            "Token counter, prompt linter, negative-prompt presets, named "
            "configuration presets, auto-LoRA discovery, batch image "
            "analysis, recipe-card export, and more."
        )

        # =====================================================================
        # 1. Token counter + linter
        # =====================================================================
        with gr.Accordion("Token counter + prompt linter", open=True):
            gr.Markdown(
                "Paste any prompt to see token usage and lint warnings. "
                "Useful before pressing Generate to make sure the CLIP "
                "context isn't overflowing and there are no contradictions."
            )
            with gr.Row():
                lint_pos = gr.Textbox(label="Positive (paste any prompt)",
                                       lines=4, value="")
                lint_neg = gr.Textbox(label="Negative (optional)",
                                       lines=4, value="")
            with gr.Row():
                lint_sfw = gr.Checkbox(label="SFW mode — fail on NSFW tags",
                                        value=False)
                lint_btn = gr.Button("Analyze ↓", variant="primary")
            token_md = gr.Markdown("*(token counter idle)*")
            lint_md  = gr.Markdown("*(lint idle)*")

            def _on_lint(pos, neg, sfw):
                tok_md = tokens.chunk_indicator(pos or "")
                issues = linter.lint(pos or "", neg or "", sfw_mode=bool(sfw))
                return tok_md, linter.format_lint_md(issues)

            lint_btn.click(fn=_on_lint, inputs=[lint_pos, lint_neg, lint_sfw],
                           outputs=[token_md, lint_md])
            # Also update token count live as user types
            lint_pos.change(
                fn=lambda p: tokens.chunk_indicator(p or ""),
                inputs=[lint_pos], outputs=[token_md])

        # =====================================================================
        # 2. Negative prompt presets
        # =====================================================================
        with gr.Accordion("Negative prompt presets", open=False):
            gr.Markdown("Pick a preset to fill the box, then click Copy.")
            with gr.Row():
                neg_choice = gr.Dropdown(
                    label="Preset", choices=neg_presets.list_names(),
                    value=neg_presets.list_names()[0])
                neg_apply_btn = gr.Button("Load →", variant="primary")
            neg_out = gr.Textbox(label="Negative prompt", lines=6,
                                 show_copy_button=True, interactive=True)

            neg_apply_btn.click(fn=lambda name: neg_presets.get(name),
                                inputs=[neg_choice], outputs=[neg_out])

        # =====================================================================
        # 3. Wildcard expander
        # =====================================================================
        with gr.Accordion("Wildcard / variant expander", open=False):
            gr.Markdown(
                "Use `{a|b|c}` syntax. Each click picks a random combination. "
                "Generate multiple variants to A/B test ideas without typing "
                "them out."
            )
            wild_in = gr.Textbox(
                label="Template",
                placeholder="a {red|blue|green} {dress|skirt}, {long|short} hair",
                lines=2,
            )
            wild_n  = gr.Slider(label="How many variants", minimum=1,
                                maximum=20, step=1, value=5)
            wild_btn = gr.Button("Expand", variant="primary")
            wild_out = gr.Textbox(label="Variants", lines=8,
                                   show_copy_button=True)

            def _on_wild(tmpl, n):
                if not wildcards.has_wildcards(tmpl or ""):
                    return "(no {a|b|c} syntax found — try `{red|blue} dress`)"
                outs = wildcards.expand_n(tmpl or "", int(n))
                return "\n".join(outs)

            wild_btn.click(fn=_on_wild, inputs=[wild_in, wild_n],
                           outputs=[wild_out])

        # =====================================================================
        # 4. Prompt-builder presets (save/load named configurations)
        # =====================================================================
        with gr.Accordion("Prompt-builder presets", open=False):
            gr.Markdown(
                "Save a full builder configuration (positive + negative + "
                "settings) under a name. Load it later to skip all the "
                "dropdowns."
            )
            with gr.Row():
                pre_name = gr.Textbox(label="Preset name",
                                       placeholder="e.g. 'My anime portrait'")
                pre_save_btn = gr.Button("Save", variant="primary")
            with gr.Row():
                pre_choice = gr.Dropdown(
                    label="Load preset",
                    choices=presets.list_names("prompt_builder"),
                    value=None)
                pre_load_btn = gr.Button("Load →")
                pre_delete_btn = gr.Button("Delete")
                pre_refresh_btn = gr.Button("Refresh")
            with gr.Row():
                pre_positive = gr.Textbox(label="Positive", lines=4,
                                           show_copy_button=True)
                pre_negative = gr.Textbox(label="Negative", lines=4,
                                           show_copy_button=True)
            pre_status = gr.Markdown("*idle*")

            def _save_preset(name, pos, neg):
                if not name or not name.strip():
                    return ("⚠️ Name required.",
                            gr.update(choices=presets.list_names("prompt_builder")))
                ok = presets.save("prompt_builder", name.strip(),
                                   {"positive": pos or "", "negative": neg or ""})
                msg = "✅ Saved" if ok else "❌ Save failed"
                return msg, gr.update(
                    choices=presets.list_names("prompt_builder"),
                    value=name.strip())

            def _load_preset(name):
                if not name:
                    return "*pick one first*", "", ""
                p = presets.load("prompt_builder", name) or {}
                return ("Loaded **" + name + "**",
                        p.get("positive", ""),
                        p.get("negative", ""))

            def _delete_preset(name):
                if not name:
                    return "*pick one first*", gr.update()
                presets.delete("prompt_builder", name)
                return ("Deleted **" + name + "**",
                        gr.update(choices=presets.list_names("prompt_builder"),
                                  value=None))

            def _refresh_presets():
                return gr.update(choices=presets.list_names("prompt_builder"))

            pre_save_btn.click(fn=_save_preset,
                               inputs=[pre_name, pre_positive, pre_negative],
                               outputs=[pre_status, pre_choice])
            pre_load_btn.click(fn=_load_preset, inputs=[pre_choice],
                               outputs=[pre_status, pre_positive, pre_negative])
            pre_delete_btn.click(fn=_delete_preset, inputs=[pre_choice],
                                 outputs=[pre_status, pre_choice])
            pre_refresh_btn.click(fn=_refresh_presets, outputs=[pre_choice])

        # =====================================================================
        # 5. LoRA auto-suggest from detected tags
        # =====================================================================
        with gr.Accordion("LoRA auto-suggest", open=False):
            gr.Markdown(
                "Index your local LoRA folders, then enter character / artist "
                "tags to get filename matches. Pick a weight and inject "
                "`<lora:name:weight>` into the box, ready to copy."
            )
            with gr.Row():
                lora_index_btn = gr.Button("(Re)build LoRA index",
                                            variant="primary")
                lora_index_md  = gr.Markdown("*not built yet*")
            lora_tags = gr.Textbox(
                label="Tags to match (comma separated)",
                placeholder="saber (fate), monochrome, sketchy"
            )
            lora_weight = gr.Slider(label="Weight to inject", minimum=0.0,
                                     maximum=1.5, step=0.05, value=0.8)
            lora_suggest_btn = gr.Button("Suggest →")
            lora_results_md  = gr.Markdown("*no suggestions yet*")
            lora_inject_box  = gr.Textbox(
                label="Inject syntax (copy into your prompt)",
                lines=4, show_copy_button=True)

            def _rebuild_lora():
                roots = _lora_search_roots()
                n = lora_suggest.refresh_index(roots)
                return "Indexed **{}** files across {} folder(s).".format(
                    n, len(roots))

            def _do_lora_suggest(tags_str, weight):
                tags = [t.strip() for t in (tags_str or "").split(",")
                        if t.strip()]
                if not tags:
                    return "*enter tags first*", ""
                hits = lora_suggest.suggest(tags, top_k=10)
                if not hits:
                    return "*(no matches in indexed LoRAs)*", ""
                lines = ["**Found {}** matches (sorted by confidence):\n"
                         .format(len(hits))]
                injects = []
                for h in hits:
                    lines.append("- `{}` — score `{:.2f}` — {}".format(
                        h["name"], h["score"], h["match_reason"]))
                    injects.append(lora_suggest.inject_syntax(
                        h["name"], float(weight)))
                return "\n".join(lines), " ".join(injects)

            lora_index_btn.click(fn=_rebuild_lora, outputs=[lora_index_md])
            lora_suggest_btn.click(
                fn=_do_lora_suggest,
                inputs=[lora_tags, lora_weight],
                outputs=[lora_results_md, lora_inject_box])

        # =====================================================================
        # 6. ControlNet recipe presets + auto-suggest
        # =====================================================================
        with gr.Accordion("ControlNet recipe presets + auto-suggest", open=False):
            gr.Markdown(
                "Auto-suggest panel reads recently-loaded image metadata and "
                "proposes which CN units to enable. Preset panel saves multi-"
                "unit configurations for reuse."
            )
            gr.Markdown("### Auto-suggest")
            cn_sugg_in = gr.Textbox(
                label="WD14 general tags (comma separated)",
                placeholder="standing, indoors, looking at viewer",
                lines=2)
            cn_sugg_chars = gr.Textbox(
                label="Character tags detected (comma separated)",
                placeholder="alice (alice in wonderland)")
            with gr.Row():
                cn_sugg_w = gr.Slider(label="Source width", minimum=512,
                                       maximum=4096, step=64, value=1024)
                cn_sugg_h = gr.Slider(label="Source height", minimum=512,
                                       maximum=4096, step=64, value=1024)
            cn_sugg_btn = gr.Button("Suggest CN units →", variant="primary")
            cn_sugg_md  = gr.Markdown("*idle*")

            def _do_cn_suggest(tags_str, chars_str, w, h):
                fake = {
                    "general":    [(t.strip(), 0.8) for t in
                                    (tags_str or "").split(",") if t.strip()],
                    "characters": [(t.strip(), 0.9) for t in
                                    (chars_str or "").split(",") if t.strip()],
                }
                s = cn_suggest.suggest(fake, source_size=(int(w), int(h)))
                lines = ["**Recommended units:**\n"]
                for k in ("canny", "depth", "pose", "tile", "ipadapter"):
                    mark = "✓" if s[k] else "·"
                    lines.append("- {} **{}**".format(mark, k.title()))
                lines.append("\n**Reasoning:**\n")
                for r in s["reasoning"]:
                    lines.append("- " + r)
                return "\n".join(lines)

            cn_sugg_btn.click(
                fn=_do_cn_suggest,
                inputs=[cn_sugg_in, cn_sugg_chars, cn_sugg_w, cn_sugg_h],
                outputs=[cn_sugg_md])

            gr.Markdown("---\n### CN recipe presets")
            with gr.Row():
                cn_pre_name = gr.Textbox(label="Recipe name",
                                          placeholder="Portrait — pose+style")
                cn_pre_payload = gr.Textbox(
                    label="Recipe JSON",
                    placeholder='{"canny": true, "depth": false, "pose": true, ...}',
                    lines=3)
                cn_pre_save_btn = gr.Button("Save", variant="primary")
            with gr.Row():
                cn_pre_choice = gr.Dropdown(
                    label="Load recipe",
                    choices=presets.list_names("cn_recipe"))
                cn_pre_load_btn = gr.Button("Load")
                cn_pre_del_btn  = gr.Button("Delete")
                cn_pre_refresh_btn = gr.Button("Refresh")
            cn_pre_status = gr.Markdown("*idle*")

            def _cn_save(name, payload):
                if not name:
                    return ("⚠️ Name required.",
                            gr.update(choices=presets.list_names("cn_recipe")))
                try:
                    parsed = json.loads(payload or "{}")
                except Exception as e:
                    return ("❌ Bad JSON: " + str(e),
                            gr.update(choices=presets.list_names("cn_recipe")))
                presets.save("cn_recipe", name.strip(), parsed)
                return ("✅ Saved " + name.strip(),
                        gr.update(choices=presets.list_names("cn_recipe"),
                                  value=name.strip()))

            def _cn_load(name):
                if not name:
                    return "*pick one*", ""
                payload = presets.load("cn_recipe", name) or {}
                return ("Loaded **" + name + "**",
                        json.dumps(payload, indent=2, ensure_ascii=False))

            def _cn_del(name):
                if not name:
                    return "*pick one*", gr.update()
                presets.delete("cn_recipe", name)
                return ("Deleted " + name,
                        gr.update(choices=presets.list_names("cn_recipe"),
                                  value=None))

            def _cn_refresh():
                return gr.update(choices=presets.list_names("cn_recipe"))

            cn_pre_save_btn.click(fn=_cn_save,
                                   inputs=[cn_pre_name, cn_pre_payload],
                                   outputs=[cn_pre_status, cn_pre_choice])
            cn_pre_load_btn.click(fn=_cn_load, inputs=[cn_pre_choice],
                                   outputs=[cn_pre_status, cn_pre_payload])
            cn_pre_del_btn.click(fn=_cn_del, inputs=[cn_pre_choice],
                                  outputs=[cn_pre_status, cn_pre_choice])
            cn_pre_refresh_btn.click(fn=_cn_refresh,
                                      outputs=[cn_pre_choice])

        # =====================================================================
        # 7. Batch image analyzer
        # =====================================================================
        with gr.Accordion("Batch image analyzer", open=False):
            gr.Markdown(
                "Point at a folder, get a CSV listing every image's size, "
                "mode, embedded prompt source, and PIL-readable metadata."
            )
            batch_folder = gr.Textbox(
                label="Folder path",
                placeholder=r"F:\some\reference\folder")
            with gr.Row():
                batch_recursive = gr.Checkbox(label="Recursive", value=False)
                batch_out = gr.Textbox(
                    label="Output CSV / JSON path",
                    placeholder=r"F:\reports\batch.csv")
                batch_btn = gr.Button("Run", variant="primary")
            batch_status = gr.Markdown("*idle*")

            def _do_batch(folder, recursive, out_path):
                if not folder or not os.path.isdir(folder):
                    return "❌ Folder doesn't exist: `{}`".format(folder)

                def _quick(img):
                    return {
                        "size":     "{}x{}".format(*img.size),
                        "mode":     img.mode,
                        "has_text": "yes" if (getattr(img, "text", None)
                                              or "parameters" in
                                              getattr(img, "info", {})) else "no",
                    }

                result = batch_analyze.batch_run(
                    folder, _quick, out_path or None,
                    recursive=bool(recursive))
                lines = [
                    "**Done** — {} images processed".format(result["count"]),
                ]
                if result["written"]:
                    lines.append("Wrote: `{}`".format(result["written"]))
                if result["errors"]:
                    lines.append("Errors: {}".format(len(result["errors"])))
                return "\n\n".join(lines)

            batch_btn.click(fn=_do_batch,
                            inputs=[batch_folder, batch_recursive, batch_out],
                            outputs=[batch_status])

        # =====================================================================
        # 8. Image compare
        # =====================================================================
        with gr.Accordion("Compare two images side-by-side", open=False):
            gr.Markdown(
                "Drop two PNGs with embedded A1111/Forge/ComfyUI/NovelAI "
                "metadata to diff their prompts and settings."
            )
            with gr.Row():
                comp_a = gr.Image(label="Image A", type="pil",
                                   image_mode="RGBA")
                comp_b = gr.Image(label="Image B", type="pil",
                                   image_mode="RGBA")
            comp_btn = gr.Button("Diff →", variant="primary")
            comp_md  = gr.Markdown("*idle*")

            def _do_compare(a, b):
                # Use the host's _extract_image_metadata via late import
                try:
                    import sys
                    sys.path.insert(0, os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))) + "/scripts")
                    from prompt_enhancer import _extract_image_metadata  # type: ignore
                except Exception:
                    return ("⚠️ Couldn't import the host metadata extractor. "
                            "Make sure the main extension loaded first.")
                meta_a = _extract_image_metadata(a) if a is not None else {}
                meta_b = _extract_image_metadata(b) if b is not None else {}
                # We don't have a host WD14 result here, so pass empty
                return image_compare.compare(meta_a, {}, meta_b, {})

            comp_btn.click(fn=_do_compare, inputs=[comp_a, comp_b],
                           outputs=[comp_md])

        # =====================================================================
        # 9. Send-to-ADetailer
        # =====================================================================
        with gr.Accordion("Send character tags to ADetailer", open=False):
            gr.Markdown(
                "Enter character tag(s) and pick a slot. The Detector Classes "
                "field of that ADetailer unit will be set in ui-config.json so "
                "on next UI reload it's pre-populated. Useful when you want a "
                "character-aware second face pass."
            )
            ad_tags = gr.Textbox(
                label="Tags (comma separated)",
                placeholder="saber (fate), alice (alice in wonderland)")
            ad_slot = gr.Radio(
                label="ADetailer slot", value="2nd",
                choices=["(1st - main)", "2nd", "3rd", "4th"])
            ad_btn  = gr.Button("Push to ui-config.json →", variant="primary")
            ad_md   = gr.Markdown("*idle*")

            def _do_send_ad(tags_str, slot):
                tags = [t.strip() for t in (tags_str or "").split(",")
                        if t.strip()]
                if not tags:
                    return "*enter tag(s) first*"
                slot_arg = "" if slot == "(1st - main)" else slot
                result = send_adetailer.push_character_classes(tags,
                                                                slot=slot_arg)
                if result["ok"]:
                    return ("✅ {} — wrote `{}` to {}".format(
                        result["message"], result.get("classes", ""),
                        result["path"]))
                return "❌ " + result["message"]

            ad_btn.click(fn=_do_send_ad, inputs=[ad_tags, ad_slot],
                         outputs=[ad_md])

        # =====================================================================
        # 10. Recipe card export
        # =====================================================================
        with gr.Accordion("Recipe-card export", open=False):
            gr.Markdown(
                "Generate a shareable PNG containing the thumbnail + prompt "
                "+ settings as one card. Handy for posting workflows."
            )
            with gr.Row():
                card_thumb = gr.Image(label="Thumbnail (your gen)",
                                       type="pil")
                with gr.Column():
                    card_pos = gr.Textbox(label="Positive", lines=4)
                    card_neg = gr.Textbox(label="Negative", lines=2)
                    card_settings = gr.Textbox(
                        label="Settings (key:val, comma separated)",
                        value="Steps: 28, Sampler: Euler a, CFG: 6.0",
                        lines=2)
            card_btn = gr.Button("Render card", variant="primary")
            card_out_img = gr.Image(label="Card", type="pil",
                                     interactive=False)

            def _do_card(thumb, pos, neg, settings_str):
                kv = {}
                for pair in (settings_str or "").split(","):
                    if ":" in pair:
                        k, _, v = pair.partition(":")
                        if k.strip():
                            kv[k.strip()] = v.strip()
                return card_export.build_card(thumb, pos or "", neg or "", kv)

            card_btn.click(fn=_do_card,
                           inputs=[card_thumb, card_pos, card_neg,
                                   card_settings],
                           outputs=[card_out_img])

        # =====================================================================
        # 11. Generation history
        # =====================================================================
        with gr.Accordion("Generation history", open=False):
            gr.Markdown(
                "Every Build/Replicate action is logged here. Browse the "
                "last 50 entries, peek at the payload, or clear the log."
            )
            with gr.Row():
                hist_refresh_btn = gr.Button("Refresh", variant="primary")
                hist_clear_btn   = gr.Button("Clear all")
            hist_md = gr.Markdown("*no entries yet — generate something*")

            def _hist_load():
                entries = history.list_entries(50)
                if not entries:
                    return "*(empty)*"
                lines = []
                for i, e in enumerate(entries):
                    lines.append("- **{}** — {}".format(
                        i, history.format_label(e)))
                return "\n".join(lines)

            def _hist_clear():
                history.clear()
                return "*(cleared)*"

            hist_refresh_btn.click(fn=_hist_load, outputs=[hist_md])
            hist_clear_btn.click(fn=_hist_clear, outputs=[hist_md])

        # =====================================================================
        # 12. Recently-used artists
        # =====================================================================
        with gr.Accordion("Recently-used artists", open=False):
            gr.Markdown(
                "Tracks the artists you've used in the main builder so you "
                "can quick-pick from a short list instead of scrolling 44K "
                "entries."
            )
            with gr.Row():
                recent_refresh_btn = gr.Button("Refresh", variant="primary")
                recent_record_box  = gr.Textbox(
                    label="(advanced) record an artist as recently-used",
                    placeholder="artist name from full dropdown")
                recent_record_btn  = gr.Button("Record")
            recent_md = gr.Markdown("*no artists recorded yet*")

            def _recent_load():
                lst = artists_recent.list_recent()
                if not lst:
                    return "*(empty)*"
                return "\n".join("- {}".format(n) for n in lst)

            def _recent_record(name):
                if not name or not name.strip():
                    return _recent_load()
                artists_recent.record(name.strip())
                return _recent_load()

            recent_refresh_btn.click(fn=_recent_load, outputs=[recent_md])
            recent_record_btn.click(fn=_recent_record,
                                     inputs=[recent_record_box],
                                     outputs=[recent_md])

    return blocks
