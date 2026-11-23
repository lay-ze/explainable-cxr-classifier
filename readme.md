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
scripts/make_fake_data.py        tiny fake dataset for testing the code without the real data
configs/default.yaml             all training settings
notebooks/01_eda.ipynb           class imbalance, co-occurrence, split check
notebooks/02_train_kaggle.ipynb  full pipeline on a Kaggle GPU
app/app.py                       Gradio demo
docs/viva-guide.html             study guide and viva questions
```

## How to run

There are three things you might want to do. Do them in this order the first time.

| Goal | Where it runs | Time | Section |
|---|---|---|---|
| Check the code works | Your laptop. No GPU or dataset needed. | ~10 min | [A](#a-quick-check-on-your-laptop) |
| Train the real model | Kaggle (free GPU) | 4–8 h, mostly unattended | [B](#b-train-the-real-model-on-kaggle) |
| Show the demo | Your laptop, after B | 5 min | [C](#c-run-the-demo-with-your-trained-model) |

### Before you start

- **Python 3.10 or newer.** Check with `python --version`. On Windows, install from [python.org](https://www.python.org/downloads/) and tick **"Add python.exe to PATH"** in the installer.
- **Git**, from [git-scm.com](https://git-scm.com/downloads), and a GitHub account that has accepted the invitation to this repository.
- **For training:** a [Kaggle](https://www.kaggle.com) account with a verified phone number. Kaggle only enables GPUs and internet access for verified accounts.

### 1. Get the code (once)

```bash
git clone https://github.com/lay-ze/explainable-cxr-classifier.git
cd explainable-cxr-classifier
```

The repository is private, so Git asks you to sign in to GitHub the first time.

### 2. Set up Python (once)

Run these from the project folder. They create a private Python environment in a `.venv` folder, so nothing is installed system-wide.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

- The `torch` line installs the CPU version of PyTorch (about 200 MB rather than several GB). Your laptop doesn't need a GPU, because training happens on Kaggle.
- When the environment is active, your prompt starts with `(.venv)`. **Every time you open a new terminal**, run the activate line again (`.venv\Scripts\Activate.ps1` or `source .venv/bin/activate`).
- If PowerShell says *"running scripts is disabled on this system"*, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, answer `Y`, and activate again.

### A. Quick check on your laptop

This uses a tiny **fake** dataset (400 noise images) to prove that every part of the code runs on your machine. The scores mean nothing. Run the commands from the project folder with `(.venv)` active:

```bash
python scripts/make_fake_data.py data/fake_nih
python -m src.train --config configs/default.yaml --data-dir data/fake_nih --out-dir outputs/smoke --epochs 2 --batch-size 16 --img-size 128 --num-workers 0
python -m src.evaluate --checkpoint outputs/smoke/best_model.pt --data-dir data/fake_nih --num-workers 0
python -m src.localization --checkpoint outputs/smoke/best_model.pt --data-dir data/fake_nih --bbox-ref-size 256 --max-images 50
python app/app.py --checkpoint outputs/smoke/best_model.pt
```

What you should see:

1. **make_fake_data:** `Fake dataset written to data/fake_nih: 400 images ...`
2. **train:** the first run downloads ImageNet weights (~30 MB, needs internet). Then two epochs of under a minute each, with `train_loss` going down.
3. **evaluate:** a table with one row per disease. **Mass** and **Cardiomegaly** score close to 1.0 because they're the only fake diseases with a visible pattern. The rest are around 0.5 (random), as they should be.
4. **localization:** a `WARNING ... in the training split` line. That's expected for fake data, where the box images are deliberately in the training set; on the real data it should say `OK`.
5. **app:** prints `Running on local URL: http://127.0.0.1:7860`. Open that address in your browser and upload any PNG from `data/fake_nih/images_001/images/`. Press `Ctrl+C` in the terminal to stop it.

If all five steps work, your setup is correct. The `data/` and `outputs/` folders are ignored by Git, so they are never uploaded.

### B. Train the real model on Kaggle

The real dataset ([NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data), ~45 GB) is already on Kaggle, so you never download it.

1. **Zip the code.** From the project folder:
   - Windows: `Compress-Archive -Path src, configs, app, requirements.txt -DestinationPath cxr-capstone-code.zip -Force`
   - macOS / Linux: `zip -r cxr-capstone-code.zip src configs app requirements.txt`
2. **Upload it as a Kaggle dataset.** On kaggle.com, go to **Create → New Dataset**, upload the zip, name it **`cxr-capstone-code`** (the notebook expects this name), keep it **Private**, then **Create**.
3. **Import the notebook.** Go to **Create → New Notebook**, then **File → Import Notebook**, and choose `notebooks/02_train_kaggle.ipynb`.
4. **Add the inputs.** In the right-hand panel, use **Add Input** to add `nih-chest-xrays/data` and your own `cxr-capstone-code`.
5. **Turn on the GPU and internet.** In **Settings**, set Accelerator to **GPU T4 x2** (or **P100**) and switch **Internet** on. The internet is needed to download the ImageNet weights.
6. **Check the paths.** In a new cell, run `!ls /kaggle/input` and `!ls /kaggle/input/*`. The code should be at `/kaggle/input/cxr-capstone-code/src` and the data at `/kaggle/input/data/Data_Entry_2017.csv`. If Kaggle put either one somewhere else, change `CODE_SRC` or `RAW_DATA` in the notebook's first cell.
7. **Run the first three cells by hand:**
   - setup: should print the GPU name, not `none`
   - image resize: about 20–30 minutes
   - 500-image test: should end with a `val_mean_AUROC` line

   If those work, the pipeline works.
8. **Start the full run in the background.** Click **Save Version → Save & Run All (Commit)**. It runs for up to 12 hours, and you can close the browser tab.
9. **Download the results.** When the run finishes, open that version, go to the **Output** tab, and download **`results.zip`**. It contains each experiment's model, metrics, plots and heatmap examples.

Kaggle gives about **30 GPU hours per week**. If you're short, edit the `EXPERIMENTS` cell so that it only runs `densenet121_wbce`, the main model. When the code changes, make a new zip and upload it on your Kaggle dataset's page with **New Version**.

### C. Run the demo with your trained model

1. Unzip `results.zip` into the project folder, so that you have `outputs/densenet121_wbce/best_model.pt` and `outputs/densenet121_wbce/thresholds.json`.
2. Run:
   ```bash
   python app/app.py --checkpoint outputs/densenet121_wbce/best_model.pt
   ```
3. Open http://127.0.0.1:7860 and upload a chest X-ray. Add `--share` to the command to get a temporary public link, for example to open the demo on a phone or another laptop.

### Command reference

Every step can also be run on its own, from the project folder:

```bash
# Resize images once to make training faster (on Kaggle: into /kaggle/temp, which isn't saved as output)
python -m src.preprocess --src /kaggle/input/data --dst /kaggle/temp/nih256 --size 256

# Train
python -m src.train --config configs/default.yaml --data-dir /kaggle/temp/nih256 --out-dir outputs/densenet121_wbce

# Evaluate on the official test split
python -m src.evaluate --checkpoint outputs/densenet121_wbce/best_model.pt --data-dir /kaggle/temp/nih256

# Check the heatmaps against the radiologist boxes (original images give the sharpest overlays)
python -m src.localization --checkpoint outputs/densenet121_wbce/best_model.pt --data-dir /kaggle/input/data --method gradcam

# Demo
python app/app.py --checkpoint outputs/densenet121_wbce/best_model.pt
```

Any setting in `configs/default.yaml` can be changed on the command line:

| Option | Values | Default |
|---|---|---|
| `--loss` | `bce`, `weighted_bce`, `focal`, `asl` | `weighted_bce` |
| `--backbone` | `densenet121`, `resnet50`, `efficientnet_b0` | `densenet121` |
| `--epochs`, `--batch-size`, `--lr` | numbers | `10`, `32`, `0.0001` |
| `--img-size` | input size in pixels | `224` |
| `--subset` | train on only this many images (quick tests) | `0` (all) |
| `--num-workers` | parallel image loaders (use `0` on Windows laptops) | `2` |

What each step writes into the run's folder (e.g. `outputs/densenet121_wbce/`):

| Step | Files |
|---|---|
| train | `best_model.pt`, `last_model.pt`, `history.json` (scores per epoch), `config.yaml` |
| evaluate | `thresholds.json`, `metrics_test.json`, `roc_curves.png`, `pr_curves.png`, `test_predictions.npz` |
| localization | `localization_gradcam/` with `summary.csv`, `per_box.csv`, and `best_*.png` / `worst_*.png` examples |

### Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'src'` | Run commands from the project folder, as `python -m src.train`, not `python src/train.py`. |
| `No module named 'torch'` (or `gradio`, `yaml` ...) | The environment isn't active. Run the activate line from step 2 again. |
| PowerShell: *running scripts is disabled* | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again. |
| Error while downloading `densenet121-....pth` | No internet. On Kaggle, switch **Internet** on in **Settings**. |
| `FileNotFoundError: ... Data_Entry_2017.csv` | `--data-dir` points to the wrong folder; it must contain that CSV. On Kaggle, check with `!ls /kaggle/input`. |
| `CUDA out of memory` | Add `--batch-size 16`. |
| Kaggle training is very slow | The GPU is off (the first cell prints `GPU: none`), or the resize step was skipped. |
| Kaggle run stops when you close the tab | Use **Save & Run All (Commit)**, not the interactive session. |
| `Checkpoint not found` | The path to `best_model.pt` is wrong. Check where you unzipped the results. |
| `Cannot find empty port` | Another program is using port 7860. Add `--port 7861`. |

### Working together on GitHub

```bash
git pull                          # always get your teammates' changes first
# ... edit files ...
git add -A
git commit -m "say what you changed"
git push
```

If `git push` is rejected, someone else pushed first: run `git pull`, then `git push` again. Datasets, `outputs/` and `.venv/` are listed in `.gitignore`, so they are never uploaded. Keep it that way; model files are too large for GitHub.

## Demo

`app/app.py` opens a Gradio page where you upload an X-ray and get:
- an independent probability for each of the 14 conditions,
- the conditions above their tuned threshold (several can be flagged at once),
- Grad-CAM / Grad-CAM++ overlays for the top-k conditions or any you pick.

## Study guide

[docs/viva-guide.html](docs/viva-guide.html) is a plain-language guide to the whole project, with diagrams, a code map and 53 practice viva questions. Download it and open it in a browser (GitHub shows HTML files as source code).

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
