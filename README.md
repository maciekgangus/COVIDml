# COVIDml — BrixIA Severity Scoring

Automated COVID-19 lung severity scoring from chest X-rays using [RAD-DINO](https://huggingface.co/microsoft/rad-dino) (Microsoft's radiology-specific Vision Transformer) as a frozen feature extractor with lightweight MLP heads.

## Task

The [BrixIA dataset](https://brixia.github.io/) contains 4,694 chest X-rays from COVID-19 patients. Each image is annotated with a **BrixIA score**: six lung zones (A–F, three per lung) each rated 0–3, summing to a global score of 0–18.

```
Right lung │ Left lung
───────────┼──────────
  A  │  B  │  D  │  E     ← upper zones
───────────┼──────────
  C        │  F           ← lower zones
```

Two models are trained:

| Model | Output | Test MAE | Pearson r |
|---|---|---|---|
| **Global head** | Single score 0–18 | 1.58 | 0.884 |
| **Zone head** | 6 scores, each 0–3 | 0.50 / zone | 0.72–0.78 |

For reference, inter-annotator MAE between radiologists on this dataset is ~1.5–2.0.

## Approach

**Linear probing on RAD-DINO** — the backbone is kept fully frozen. CLS-token embeddings (768-dim) are extracted once per image and saved to disk. Two small MLPs are then trained on those embeddings in seconds on CPU.

```
DICOM → 16-bit PNG → RAD-DINO (frozen) → CLS [768] → MLP head → score
```

This works well because RAD-DINO was pretrained on 882,775 chest X-rays (MIMIC-CXR, CheXpert, NIH-CXR, PadChest, BRAX) — the same domain as BrixIA.

## Project Structure

```
COVIDml/
├── src/
│   ├── utils.py                  # shared: paths, metadata, BrixiaDataset
│   ├── 01_extract_dicom.py       # DICOM → 16-bit PNG (parallel, idempotent)
│   ├── 02_analyze_dataset.py     # EDA plots + split summary
│   ├── 03_extract_embeddings.py  # frozen RAD-DINO → .npy CLS tokens
│   ├── 04_train_global.py        # train global score head
│   ├── 05_train_zones.py         # train per-zone head
│   ├── 06_finetune_global.py     # optional: end-to-end fine-tuning
│   └── predict.py                # inference on a single DICOM
├── models/
│   ├── global_head.pt            # global score checkpoint (773KB)
│   ├── global_results.json       # test metrics
│   ├── zone_head.pt              # zone score checkpoint (778KB)
│   └── zone_results.json         # per-zone test metrics
├── notebooks/
│   └── brixia_walkthrough.ipynb  # end-to-end Jupyter walkthrough
└── docs/
    └── superpowers/
        ├── specs/                # design document
        └── plans/               # implementation plan
```

## Setup

```bash
# Python env (requires pyenv)
pyenv virtualenv 3.12.13 covidml
pyenv local covidml
pip install -r requirements.txt
```

## Running the Pipeline

```bash
# 1. Extract DICOMs (already done if you have data/dicom_raw/)
cat osfstorage-archive/data/dicom/*.tar.gz.* | tar -xzf - -C data/dicom_raw

# 2. Convert to PNG (parallel, ~15 min)
python src/01_extract_dicom.py

# 3. EDA plots → data/plots/
python src/02_analyze_dataset.py

# 4. Extract RAD-DINO embeddings once (~2 min on RTX 5070 Ti)
python src/03_extract_embeddings.py

# 5. Train both heads (seconds, CPU only)
python src/04_train_global.py
python src/05_train_zones.py
```

## Inference

```bash
python src/predict.py path/to/image.dcm
```

```
Global score : 4.1 / 18
Zone scores  :
  Zone A: 0/3  ░░░
  Zone B: 1/3  █░░
  Zone C: 2/3  ██░
  Zone D: 0/3  ░░░
  Zone E: 0/3  ░░░
  Zone F: 1/3  █░░

Reconstructed sum: 4 / 18
```

RAD-DINO is downloaded automatically from HuggingFace on first use.

## Results

### Global score (0–18)

| Metric | Linear probe |
|---|---|
| MAE | 1.58 |
| RMSE | 1.98 |
| Exact match | 20.0% |
| Within ±1 | 38.0% |
| Within ±2 | 68.0% |
| Within ±3 | 85.3% |
| Pearson r | 0.884 |

### Per-zone scores (0–3 each)

| Zone | MAE | Exact | Within ±1 | Pearson r |
|---|---|---|---|---|
| A | 0.44 | 66.7% | 90.0% | 0.72 |
| B | 0.51 | 58.7% | 91.3% | 0.77 |
| C | 0.50 | 56.7% | 90.7% | 0.78 |
| D | 0.43 | 66.7% | 90.0% | 0.68 |
| E | 0.55 | 49.3% | 85.3% | 0.75 |
| F | 0.56 | 51.3% | 86.0% | 0.70 |
| **Mean** | **0.50** | **58.2%** | **88.9%** | — |

Test set = 150-image consensus subset with multi-annotator ground truth.

## Optional: End-to-end Fine-tuning

Partially unfreezes the last N RAD-DINO blocks for end-to-end training. On this dataset size (~4,500 images), linear probing outperforms fine-tuning due to overfitting risk.

```bash
# Train (saves to models/finetuned_global.pt)
python src/06_finetune_global.py --mode train --unfreeze 4

# Evaluate saved checkpoint
python src/06_finetune_global.py --mode load
```

## Dataset

[BrixIA: COVID-19 Severity Score Assessment and Database](https://brixia.github.io/)

> **Not for clinical use.** Research purposes only.

## Environment

- Python 3.12.13
- PyTorch 2.11 + CUDA 12.8
- Transformers 5.9
- Tested on RTX 5070 Ti (16GB)
