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
scripts/analyse_results.py       confidence intervals and the flagged/missed heatmap split for the report
scripts/render_cases.py          pictures of chosen cases (box + heatmap) and test X-rays for the demo
configs/default.yaml             all training settings
notebooks/01_eda.ipynb           class imbalance, co-occurrence, split check
notebooks/02_train_kaggle.ipynb  full pipeline on a Kaggle GPU
notebooks/03_shortcut_cases.ipynb  pictures of the shortcut candidates (Kaggle, CPU only)
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
| Picture the shortcut cases (optional) | Kaggle, CPU only (no GPU hours) | ~15 min | [D](#d-picture-the-shortcut-candidates) |

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
   - Windows (PowerShell): `tar -a -c -f cxr-capstone-code.zip --exclude=__pycache__ src configs app requirements.txt`
   - macOS / Linux: `zip -r cxr-capstone-code.zip src configs app requirements.txt -x '*__pycache__*'`

   Don't use `Compress-Archive` on Windows. It saves folder paths with backslashes (`src\train.py`), which Kaggle's Linux machines don't read as folders. The zip is listed in `.gitignore`, so it never goes to GitHub.
2. **Upload it as a Kaggle dataset.** On kaggle.com, go to **Create → New Dataset**, drag in the zip, title it **`cxr-capstone-code`**, keep it **Private**, then **Create**. Kaggle unzips it for you.
3. **Import the notebook.** Go to **Create → New Notebook**, then **File → Import Notebook**, and choose `notebooks/02_train_kaggle.ipynb` from the project folder.
4. **Add the inputs.** In the right-hand panel, click **Add Input** twice: search for `nih-chest-xrays/data` (the NIH dataset, ~45 GB, nothing is downloaded) and add it, then add your own `cxr-capstone-code` (under **Your Work** or **Datasets**).
5. **Turn on the GPU and internet.** In the right-hand panel under **Session options** (or **Settings**), set Accelerator to **GPU T4 x2** or **GPU P100** and switch **Internet** on. The internet is needed to download the ImageNet starting weights.
6. **Run the first two cells by hand** (click a cell, press `Shift+Enter`). This takes about 5 minutes:
   - **Setup cell:** prints `code: ...`, `data: ...` and `GPU: Tesla T4` (or `P100`). If it says `Cannot find ...`, that input wasn't added in step 4. If it says `GPU: none`, redo step 5.
   - **Quick test cell:** trains on 500 images for one epoch and ends with `epoch 1: train_loss=...  val_mean_AUROC=...` and `done.`. The score means nothing yet.

   If both work, the whole pipeline works. Don't run the other cells by hand; the full run does them.
7. **Start the full run.** Click **Save Version** (top right), choose **Save & Run All (Commit)**, then **Save**. Kaggle runs every cell from the top on a fresh machine, for up to 12 hours.
8. **Stop the interactive session.** Click the power icon (or **Run → Stop session**). The full run carries on separately; the interactive session would otherwise keep using your GPU hours. You can now close the browser tab.
9. **Check the timing (optional).** About 1.5 hours later (the image resize comes first), open the notebook from **Your Work → Code**, click the running version and read its log. Find the first line `epoch 1: ... (NNNs)` of `densenet121_bce`. NNN × 30 is roughly the total training time in seconds; it should be under about 8 hours.
10. **Download the results.** When the version says it's complete, open it, go to the **Output** tab, and download **`results.zip`**. It contains each experiment's model, metrics, plots and heatmap examples.

Kaggle gives about **30 GPU hours per week**; the full run uses about 4–7. If you're short, edit the `EXPERIMENTS` cell so that it only runs `densenet121_wbce`, the main model, and save a new version. When the code changes, make a new zip and upload it on your Kaggle dataset's page with **New Version**.

### C. Run the demo with your trained model

1. Unzip `results.zip` into the project folder, so that you have `outputs/densenet121_wbce/best_model.pt` and `outputs/densenet121_wbce/thresholds.json`.
2. Run:
   ```bash
   python app/app.py --checkpoint outputs/densenet121_wbce/best_model.pt
   ```
3. Open http://127.0.0.1:7860 and upload a chest X-ray. Add `--share` to the command to get a temporary public link, for example to open the demo on a phone or another laptop.

Real test X-rays to try come out of section D, in `demo_samples/`. You can also download single images from the NIH dataset's page on Kaggle.

### D. Picture the shortcut candidates

`scripts/analyse_results.py` lists the boxes where the model flagged a disease with confidence but put almost no heat inside the radiologist's box. This step draws those cases, using the original X-rays on Kaggle. It runs on the CPU, so it uses none of your GPU hours.

1. **Make the upload zip.** From the project folder, after unzipping `results.zip` and running `python scripts/analyse_results.py`:
   - Windows (PowerShell):
     ```powershell
     New-Item -ItemType Directory -Force trained\densenet121_wbce | Out-Null
     Copy-Item outputs\densenet121_wbce\best_model.pt, outputs\densenet121_wbce\thresholds.json, outputs\densenet121_wbce\localization_gradcam\shortcut_candidates.csv trained\densenet121_wbce\
     tar -a -c -f cxr-shortcut-check.zip --exclude=__pycache__ src configs app scripts requirements.txt trained
     ```
   - macOS / Linux:
     ```bash
     mkdir -p trained/densenet121_wbce
     cp outputs/densenet121_wbce/{best_model.pt,thresholds.json,localization_gradcam/shortcut_candidates.csv} trained/densenet121_wbce/
     zip -r cxr-shortcut-check.zip src configs app scripts requirements.txt trained -x '*__pycache__*'
     ```

   The model goes in `trained/`, not `outputs/`. If it were in `outputs/`, the training notebook would think it was already trained and skip it. `trained/` is listed in `.gitignore`.
2. **Upload it.** On kaggle.com, go to **Create → New Dataset**, drag in `cxr-shortcut-check.zip`, title it `cxr-shortcut-check`, keep it **Private**, then **Create**.
3. **Import the notebook.** Go to **Create → New Notebook → File → Import Notebook**, and choose `notebooks/03_shortcut_cases.ipynb`.
4. **Add the inputs.** Use **Add Input** to add `cxr-shortcut-check` and the NIH dataset, the 45 GB **NIH Chest X-rays** by the National Institutes of Health.
5. **Leave the GPU off.** In **Session options**, set Accelerator to **None**. Internet isn't needed.
6. **Run it.** Click **Save Version → Save & Run All (Commit)**. It takes about 5–15 minutes.
7. **Download the results.** From the version's **Output** tab, download **`shortcut_cases.zip`**. It contains:
   - one picture per case: the X-ray with the radiologist's box on the left, and with the heatmap on the right
   - `overview.png`: all the cases on one page
   - `demo_samples/`: 14 test X-rays (2 per disease, plus 2 with no finding) for trying the demo, with their real labels in `labels.csv`

In each picture, look at where the heat sits. Common shortcuts are a chest drain (a tube, usually entering from the side of the chest), ECG wires, text markers such as "PORTABLE", the shoulder, or the edge of the image.

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

[docs/viva-guide.html](docs/viva-guide.html) is a plain-language guide to the whole project, with diagrams, a code map and 54 practice viva questions. Download it and open it in a browser (GitHub shows HTML files as source code).

## Results

All three models were trained on Kaggle on one Tesla T4, at about 7.7 minutes per epoch; the whole notebook took about 4 hours. Every number below is on the **official NIH test split**: 25,596 images from 2,797 patients, none of them used for training, early stopping or threshold tuning. All 984 bounding-box images are in this split (`localization.py` printed `OK: all 984 box images are in the held-out test split`).

`python scripts/analyse_results.py` reproduces the confidence intervals and the flagged/missed split (about 5 minutes on a laptop).

### 1. Classification: the three losses tie

| Loss | Mean AUROC (95% CI) | Mean AUPRC | Mean F1 | Best validation AUROC | Epochs run (best) |
|---|---|---|---|---|---|
| BCE | 0.801 (0.794–0.808) | 0.269 | 0.329 | 0.835 | 8 (5) |
| Weighted BCE | 0.799 (0.792–0.805) | 0.264 | 0.325 | 0.832 | 10 (8) |
| Focal | 0.798 (0.791–0.805) | 0.266 | 0.327 | 0.834 | 8 (5) |

Confidence intervals come from a paired bootstrap that resamples patients (1,000 resamples), since most patients have several X-rays.

- **No loss wins on mean AUROC.** Weighted BCE minus BCE is −0.002 (95% CI −0.007 to +0.003); focal minus BCE is −0.003 (−0.006 to +0.000). AUROC only measures how images are ranked. Class weights mainly push rare-disease probabilities up rather than reorder them, and the per-class thresholds already correct the decision cut-off.
- **Weighting helps the rarest disease.** Hernia AUPRC rises from 0.270 with BCE to 0.361 with weighted BCE, a difference of +0.09 (95% CI +0.01 to +0.17). The test set has only 32 patients with hernia, so this is a small sample.
- **Validation (0.83) is higher than test (0.80).** The model and thresholds were chosen on validation, and the official test patients are sicker than average: for example, 23.9% have infiltration against 17.7% across the whole dataset.

Weighted BCE is the main model, used for the heatmap check and the demo. It is tied on AUROC and best on the rarest class.

**Per-disease test results.** AUROC for all three losses, with the best in bold. AUPRC, sensitivity and specificity are for weighted BCE at its tuned threshold.

| Condition | Test positives | AUROC BCE | AUROC weighted | AUROC focal | AUPRC | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|
| Atelectasis | 3,279 (12.8%) | **0.767** | 0.754 | 0.766 | 0.311 | 0.53 | 0.81 |
| Cardiomegaly | 1,069 (4.2%) | 0.883 | **0.893** | 0.882 | 0.331 | 0.52 | 0.95 |
| Effusion | 4,658 (18.2%) | 0.820 | **0.824** | 0.822 | 0.504 | 0.51 | 0.89 |
| Infiltration | 6,112 (23.9%) | 0.675 | **0.690** | 0.667 | 0.385 | 0.69 | 0.59 |
| Mass | 1,748 (6.8%) | **0.811** | 0.803 | 0.808 | 0.278 | 0.34 | 0.95 |
| Nodule | 1,623 (6.3%) | **0.750** | 0.729 | 0.732 | 0.203 | 0.41 | 0.87 |
| Pneumonia | 555 (2.2%) | **0.722** | 0.695 | 0.708 | 0.049 | 0.07 | 0.98 |
| Pneumothorax | 2,665 (10.4%) | **0.851** | 0.850 | 0.850 | 0.384 | 0.53 | 0.91 |
| Consolidation | 1,815 (7.1%) | 0.742 | **0.744** | 0.735 | 0.162 | 0.53 | 0.78 |
| Edema | 925 (3.6%) | **0.850** | 0.839 | 0.848 | 0.153 | 0.31 | 0.95 |
| Emphysema | 1,093 (4.3%) | 0.891 | **0.896** | 0.892 | 0.361 | 0.55 | 0.95 |
| Fibrosis | 435 (1.7%) | **0.815** | 0.808 | 0.807 | 0.084 | 0.15 | 0.98 |
| Pleural Thickening | 1,143 (4.5%) | **0.771** | 0.762 | 0.764 | 0.135 | 0.35 | 0.91 |
| Hernia | 86 (0.3%) | 0.868 | **0.903** | 0.893 | 0.361 | 0.40 | 1.00 |

Diseases with clear visual signs score highest: hernia, emphysema and cardiomegaly, at about 0.89–0.90. Infiltration (0.69) and pneumonia (0.70) score lowest; they look vague and their labels are noisy. Pneumonia's F1-tuned threshold catches only 7% of cases, so a screening tool would need a lower threshold for it.

### 2. Does the model look in the right place?

This compares the weighted-BCE model's Grad-CAM heatmaps with the 984 radiologist boxes:

- **Pointing:** the heatmap's hottest point is inside the box.
- **Random baseline:** the box's share of the image, which is what a randomly placed peak would score.
- **Flagged:** the model's probability was above its tuned threshold, so it actually predicted the disease.

| Condition | Boxes | Flagged | Pointing (flagged) | Pointing (missed) | Energy in box | Random baseline | Test AUROC |
|---|---|---|---|---|---|---|---|
| Cardiomegaly | 146 | 66% | **0.99** | 0.76 | 0.51 | 0.18 | 0.893 |
| Pneumonia | 120 | 14% | 0.65 | 0.32 | 0.20 | 0.09 | 0.695 |
| Infiltration | 123 | 76% | 0.50 | 0.14 | 0.18 | 0.10 | 0.690 |
| Mass | 85 | 40% | 0.35 | 0.06 | 0.12 | 0.04 | 0.803 |
| Effusion | 153 | 61% | 0.34 | 0.15 | 0.15 | 0.07 | 0.824 |
| Atelectasis | 180 | 58% | 0.22 | 0.11 | 0.08 | 0.03 | 0.754 |
| Pneumothorax | 98 | 49% | 0.15 | 0.14 | 0.10 | 0.05 | 0.850 |
| Nodule | 79 | 43% | 0.06 | 0.02 | 0.01 | 0.005 | 0.729 |
| **All** | **984** | **53%** | **0.44** | **0.22** | **0.18** | **0.08** | |

Grad-CAM++ gives similar overall numbers: pointing 0.36, energy 0.16 and IoU 0.21, against 0.34, 0.18 and 0.20 for Grad-CAM.

- **Better than random for every disease.** When the model flags a disease, its heatmap peak is inside the radiologist's box 44% of the time, against 8% for a random peak.
- **Cardiomegaly is right for the right reason.** 99% of flagged cases point at the heart.
- **Pneumothorax is the warning sign.** It has the fourth-highest AUROC (0.85) but the weakest localisation relative to chance. Pointing is no better when the model flags it (0.15) than when it misses it (0.14). Part of the score seems to come from evidence outside the collapsed region. This is consistent with published work showing that NIH pneumothorax models partly detect chest drains, the tubes used to treat it (Oakden-Rayner et al., 2020).
- **Nodules are too small for Grad-CAM.** Their boxes cover 0.5% of the image, less than one cell of the 7×7 heatmap grid, so their low scores mostly reflect resolution rather than shortcuts.
- **31 of the 521 flagged boxes (6%) have almost no heat inside the box.** By condition: Nodule 11, Pneumothorax 7, Effusion 4, Atelectasis 4, Mass 3 and Infiltration 2. They are listed in `localization_gradcam/shortcut_candidates.csv` and are the cases to inspect for drains, wires or text markers. [Section D](#d-picture-the-shortcut-candidates) draws them.
- **The saved `worst_*.png` examples are missed diagnoses, not shortcuts.** All six had a probability below the threshold. Five have heat only outside the box, and one has no heatmap at all. The script ranks every box by energy in box, so a disease the model missed counts as "worst".

## Limitations

- NIH labels were mined from radiology reports by NLP and are noisy (estimated >90% accurate, lower for some classes).
- Grad-CAM resolution is limited by the 7×7 final feature map at 224px, so small nodules localise poorly.
- Bounding boxes cover only 8 of the 14 classes and fewer than 1,000 images.
- Each loss was trained once, with one random seed. The bootstrap intervals cover test-set variation, not training variation.
- The 31 shortcut candidates have been found by the numbers but not yet inspected visually.
- All images come from one hospital system; the model has not been tested on X-rays from elsewhere.
- This is a research prototype, not a medical device.

## References

- Wang et al., *ChestX-ray8*, CVPR 2017.
- Rajpurkar et al., *CheXNet*, arXiv:1711.05225, 2017.
- Selvaraju et al., *Grad-CAM*, ICCV 2017. Chattopadhay et al., *Grad-CAM++*, WACV 2018.
- Lin et al., *Focal Loss*, ICCV 2017. Ridnik et al., *Asymmetric Loss*, ICCV 2021.
- DeGrave et al., *AI for radiographic COVID-19 detection selects shortcuts over signal*, Nature Machine Intelligence 2021.
- Oakden-Rayner et al., *Hidden stratification causes clinically meaningful failures in machine learning for medical imaging*, ACM CHIL 2020.
