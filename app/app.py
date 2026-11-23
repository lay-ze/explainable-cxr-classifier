"""Gradio demo: upload a chest X-ray, get 14 independent probabilities and Grad-CAM heatmaps.

    python app/app.py --checkpoint outputs/run/best_model.pt          # local
    python app/app.py --checkpoint outputs/run/best_model.pt --share  # public link (Colab/Kaggle)

If thresholds.json (written by src.evaluate) sits next to the checkpoint, its per-class
thresholds decide which findings are flagged; otherwise 0.5 is used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
import torch  # noqa: E402

from src.explain import CAMExplainer, overlay  # noqa: E402
from src.labels import CLASSES  # noqa: E402
from src.model import load_checkpoint  # noqa: E402

DISCLAIMER = ("**Research prototype, not a medical device.** Predictions and heatmaps are for "
              "education and model analysis only and must not be used for diagnosis.")


def build_app(checkpoint: str) -> gr.Blocks:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(checkpoint, device)
    img_size = ckpt["config"]["img_size"]
    thr_path = Path(checkpoint).parent / "thresholds.json"
    thresholds = json.loads(thr_path.read_text()) if thr_path.exists() else {c: 0.5 for c in CLASSES}
    explainers = {m: CAMExplainer(model, img_size, m, device) for m in ("gradcam", "gradcam++")}

    def analyse(img, method, top_k, extra_classes):
        if img is None:
            return None, "Upload an X-ray first.", []
        exp = explainers[method]
        x = exp.preprocess(img)
        with torch.no_grad():
            probs = torch.sigmoid(model(x))[0].cpu().numpy()

        order = probs.argsort()[::-1]
        top_k = int(top_k)  # sliders can deliver floats
        chosen = list(dict.fromkeys([int(i) for i in order[:top_k]] + [CLASSES.index(c) for c in extra_classes or []]))
        _, cams = exp.explain(x, chosen)
        gallery = [(overlay(img, cams[c]), f"{CLASSES[c]}  p={probs[c]:.2f}") for c in chosen]

        flagged = [(CLASSES[i], probs[i], thresholds[CLASSES[i]]) for i in order if probs[i] >= thresholds[CLASSES[i]]]
        if flagged:
            md = "### Flagged findings (probability ≥ tuned threshold)\n" + "\n".join(
                f"- **{n}**: {p:.2f} (threshold {t:.2f})" for n, p, t in flagged)
        else:
            md = "### No finding above its threshold"
        md += f"\n\n_Each condition is scored independently (sigmoid), so several can be flagged at once._\n\n{DISCLAIMER}"
        label = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
        return label, md, gallery

    with gr.Blocks(title="Explainable Chest X-ray Classifier") as demo:
        gr.Markdown("# Explainable multi-label chest X-ray classifier\n"
                    f"Backbone: `{ckpt['config']['backbone']}` · trained on NIH ChestX-ray14 · "
                    f"validation mean AUROC {ckpt.get('val_mean_auroc', float('nan')):.3f}\n\n{DISCLAIMER}")
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="Chest X-ray (frontal view)", image_mode="RGB")
                method = gr.Radio(["gradcam", "gradcam++"], value="gradcam", label="Explanation method")
                top_k = gr.Slider(1, 6, value=3, step=1, label="Heatmaps for top-k conditions")
                extra = gr.Dropdown(CLASSES, multiselect=True, label="Also explain these conditions")
                btn = gr.Button("Analyse", variant="primary")
            with gr.Column(scale=1):
                probs_out = gr.Label(num_top_classes=len(CLASSES), label="Probability per condition")
                findings = gr.Markdown()
        gallery = gr.Gallery(label="Where the model is looking (Grad-CAM)", columns=3, height="auto")
        btn.click(analyse, [inp, method, top_k, extra], [probs_out, findings, gallery])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "outputs" / "run" / "best_model.pt"))
    ap.add_argument("--share", action="store_true", help="create a public Gradio link")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    if not Path(a.checkpoint).exists():
        sys.exit(f"Checkpoint not found: {a.checkpoint}\nTrain a model first (see readme.md) or pass --checkpoint.")
    build_app(a.checkpoint).launch(share=a.share, server_port=a.port)


if __name__ == "__main__":
    main()
