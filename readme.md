# Explainable Multi-Label Chest X-ray Classification

B.Tech 3rd-year capstone project. A CNN predicts **14 thoracic conditions** from a frontal chest X-ray (NIH ChestX-ray14). **Grad-CAM / Grad-CAM++** heatmaps show where the model is looking. The heatmaps are then **scored against radiologist bounding boxes** to test whether the model uses real pathology or shortcuts.

## Problem

1. **Multi-label and imbalanced.** One X-ray can show several conditions at once (e.g. Effusion + Cardiomegaly). Each condition is predicted independently with a sigmoid output; softmax is not used. Prevalence ranges from ~18% (Infiltration) down to ~0.2% (Hernia).
2. **Lack of transparency.** A high AUROC doesn't show that the model looks at the lesion. It could be using artifacts, text markers, tubes or hospital-specific patterns. Clinicians need explanations they can verify.

## Method

| Part | Choice | Why |
|---|---|---|
| Data split | Official NIH train_val / test lists, with a **patient-level** validation split | NIH has many follow-up scans per patient. Splitting by image would leak patients and inflate scores. |
| Model | DenseNet-121 (ImageNet-pretrained), 14 sigmoid outputs | The CheXNet baseline. ResNet-50 and EfficientNet-B0 are also supported for comparison. |
| Imbalance | Compared: BCE, **weighted BCE** (`pos_weight = neg/pos`), focal loss, asymmetric loss | Stops rare classes from being ignored. |
| Decisions | **Per-class thresholds** tuned on validation to maximise F1 | A single 0.5 threshold almost never fires for rare classes. |
| Metrics | Per-class AUROC, **AUPRC**, F1, sensitivity, specificity | AUPRC shows rare-class performance more honestly than AUROC. |
| Augmentation | Small affine, brightness/contrast; **no horizontal flip** | A flip puts the heart on the wrong side. |
| Explainability | Grad-CAM and Grad-CAM++, one heatmap per class | Multi-label: each condition has its own explanation. |
| Shortcut check | Pointing game, energy-in-box and IoU against `BBox_List_2017.csv` (~980 boxes, 8 classes), compared with a random-heatmap baseline | Turns "the heatmap looks right" into a number. |

## Project structure

```
src/
  labels.py        14 class names, multi-hot encoding
  dataset.py       metadata loading, patient-level split, transforms, Dataset
  model.py         backbone + 14-way head (features/head split for Grad-CAM)
  losses.py        weighted BCE, focal, asymmetric loss
  train.py         training loop (AMP, cosine LR, early stopping on val AUROC)
  evaluate.py      metrics, threshold tuning, ROC/PR plots   (CLI)
  explain.py       Grad-CAM / Grad-CAM++ and heatmap overlays
  localization.py  heatmap vs bounding-box evaluation        (CLI)
  preprocess.py    optional one-time resize to 256px for faster training
configs/default.yaml
notebooks/01_eda.ipynb           class imbalance, co-occurrence, split check
notebooks/02_train_kaggle.ipynb  full pipeline on a Kaggle/Colab GPU
app/app.py                       Gradio demo
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt   # CPU is fine for the demo; train on a GPU
```

**Dataset:** [NIH ChestX-ray14 on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) (~45 GB). On Kaggle it's mounted at `/kaggle/input/data`, so there's nothing to download.

## Running

The easiest path is to open `notebooks/02_train_kaggle.ipynb` on Kaggle with a GPU and run it top to bottom. The same steps as commands:

```bash
# 0. (optional, recommended) resize images once, into Kaggle's temp folder (not saved as output)
python -m src.preprocess --src /kaggle/input/data --dst /kaggle/temp/nih256 --size 256

# 1. train  (any config key can be overridden: --loss focal --backbone efficientnet_b0 --epochs 5 ...)
python -m src.train --config configs/default.yaml --data-dir /kaggle/temp/nih256 --out-dir outputs/densenet121_wbce

# 2. evaluate on the test split (writes thresholds.json, metrics_test.json, ROC/PR plots)
python -m src.evaluate --checkpoint outputs/densenet121_wbce/best_model.pt --data-dir /kaggle/temp/nih256

# 3. explainability check against bounding boxes (original images give the sharpest overlays)
python -m src.localization --checkpoint outputs/densenet121_wbce/best_model.pt --data-dir /kaggle/input/data --method gradcam

# 4. demo (copy best_model.pt + thresholds.json into one folder locally)
python app/app.py --checkpoint outputs/densenet121_wbce/best_model.pt
```

For a quick test, add `--subset 500 --epochs 1` to the train command.

## Demo

`app/app.py` opens a Gradio page where you upload an X-ray and get:
- an independent probability for each of the 14 conditions,
- the conditions above their tuned threshold (several can be flagged at once),
- Grad-CAM / Grad-CAM++ overlays for the top-k conditions or any you pick.

## Results

_Fill in after training._

| Experiment | Mean AUROC | Mean AUPRC | Mean F1 |
|---|---|---|---|
| DenseNet-121 + BCE | | | |
| DenseNet-121 + weighted BCE | | | |
| DenseNet-121 + focal | | | |

| Class (bbox subset) | Pointing game | Energy in box | Random baseline (box fraction) |
|---|---|---|---|
| | | | |

A CheXNet-style DenseNet-121 at 224px usually reaches a mean test AUROC of around 0.80–0.84 on the official split.

## Limitations

- NIH labels were mined from radiology reports by NLP and are noisy (estimated >90% accurate, lower for some classes).
- Grad-CAM resolution is limited by the 7×7 final feature map at 224px, so small nodules localise poorly.
- Bounding boxes cover only 8 of the 14 classes and fewer than 1,000 images.
- This is a research prototype, not a medical device.

## References

- Wang et al., *ChestX-ray8*, CVPR 2017.
- Rajpurkar et al., *CheXNet*, arXiv:1711.05225, 2017.
- Selvaraju et al., *Grad-CAM*, ICCV 2017. Chattopadhay et al., *Grad-CAM++*, WACV 2018.
- Lin et al., *Focal Loss*, ICCV 2017. Ridnik et al., *Asymmetric Loss*, ICCV 2021.
- DeGrave et al., *AI for radiographic COVID-19 detection selects shortcuts over signal*, Nature Machine Intelligence 2021.
