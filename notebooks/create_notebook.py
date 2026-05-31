"""Generate the brixia_walkthrough.ipynb notebook programmatically."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "covidml",
    "language": "python",
    "name": "covidml",
}

cells = []

def md(text):
    return nbf.v4.new_markdown_cell(text)

def code(text):
    return nbf.v4.new_code_cell(text.strip())

# ── Section 1: Setup ────────────────────────────────────────────────────────
cells.append(md("# BrixIA COVID-19 Severity Scoring\nEnd-to-end walkthrough: DICOM → RAD-DINO embeddings → MLP heads"))
cells.append(code("""
import sys
sys.path.insert(0, '../src')
import warnings; warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

%matplotlib inline
plt.rcParams['figure.dpi'] = 120
"""))

# ── Section 2: Dataset Overview ─────────────────────────────────────────────
cells.append(md("## 1. Dataset Overview"))
cells.append(code("""
from utils import load_metadata, get_splits, ZONE_NAMES

global_df, consensus_df = load_metadata()
train_files, val_files, test_files = get_splits()

print(f"Total images : {len(global_df):,}")
print(f"  Train      : {len(train_files):,}")
print(f"  Val        : {len(val_files):,}")
print(f"  Test       : {len(test_files):,}  (consensus set)")
print()
global_df[["BrixiaScoreGlobal"] + [f"Zone{z}" for z in ZONE_NAMES]].describe().round(2)
"""))

# ── Section 3: EDA plots ─────────────────────────────────────────────────────
cells.append(md("## 2. Exploratory Data Analysis"))
cells.append(code("""
plots_dir = Path('../data/plots')
for fname in ['global_score_dist.png', 'zone_score_dists.png', 'image_dimensions.png']:
    img = mpimg.imread(plots_dir / fname)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(img); ax.axis('off'); ax.set_title(fname)
    plt.tight_layout(); plt.show()
"""))

# ── Section 4: Embeddings ────────────────────────────────────────────────────
cells.append(md("## 3. RAD-DINO Embeddings\n\nRun `python src/03_extract_embeddings.py` first (takes ~2 min on RTX 5070 Ti)."))
cells.append(code("""
emb_dir = Path('../data/embeddings')
splits = {}
for split in ['train', 'val', 'test']:
    splits[split] = {
        'X': np.load(emb_dir / f'embeddings_{split}.npy'),
        'y_global': np.load(emb_dir / f'labels_global_{split}.npy'),
        'y_zones': np.load(emb_dir / f'labels_zones_{split}.npy'),
    }
    print(f"{split:5s}: X={splits[split]['X'].shape}  "
          f"y_global={splits[split]['y_global'].shape}  "
          f"y_zones={splits[split]['y_zones'].shape}")
"""))

# ── Section 5: Training ──────────────────────────────────────────────────────
cells.append(md("## 4. Train MLP Heads\n\nRuns both heads and shows training curves."))
cells.append(code("""
import importlib.util

def load_module(path):
    spec = importlib.util.spec_from_file_location('m', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

global_mod = load_module('../src/04_train_global.py')
global_model, global_results = global_mod.train()
"""))
cells.append(code("""
zone_mod = load_module('../src/05_train_zones.py')
zone_model, zone_results = zone_mod.train()
"""))
cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, results, title in [
    (axes[0], global_results, 'Global Head'),
    (axes[1], zone_results, 'Zone Head'),
]:
    ax.plot(results['history']['val_mae'], label='val MAE')
    ax.plot(results['history']['train_loss'], label='train loss', alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.legend()
plt.tight_layout()
plt.show()
"""))

# ── Section 6: Evaluation ────────────────────────────────────────────────────
cells.append(md("## 5. Test Set Evaluation"))
cells.append(code("""
import json

g = json.load(open('../models/global_results.json'))
z = json.load(open('../models/zone_results.json'))

print("=== Global Head ===")
print(f"  Test MAE  : {g['test_mae']:.4f}")
print(f"  Test RMSE : {g['test_rmse']:.4f}")

print("\\n=== Zone Head ===")
for zone, mae in z['per_zone_mae'].items():
    print(f"  Zone {zone} MAE: {mae:.4f}")
print(f"  Mean  MAE: {z['mean_mae']:.4f}")
"""))
cells.append(code("""
import torch
X_test = torch.tensor(splits['test']['X'])
y_test = splits['test']['y_global']

global_model.eval()
with torch.no_grad():
    preds = global_model(X_test).clamp(0, 18).numpy()

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, preds, alpha=0.6, s=40)
ax.plot([0, 18], [0, 18], 'r--', linewidth=1.5, label='perfect')
ax.set_xlabel('Ground Truth (Global Score)')
ax.set_ylabel('Predicted')
ax.set_title('Global Score: Predicted vs Ground Truth (test set)')
ax.legend()
plt.tight_layout()
plt.show()
"""))

# ── Section 7: Inference ─────────────────────────────────────────────────────
cells.append(md("## 6. Inference on a Single DICOM"))
cells.append(code("""
import torch
import cv2
import numpy as np
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel

def predict_dicom(dcm_path: str):
    \"\"\"Given a DICOM path, return global score and per-zone scores.\"\"\"
    import pydicom
    from utils import ZONE_NAMES, MODELS_DIR

    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)
    if getattr(ds, 'PhotometricInterpretation', '') == 'MONOCHROME1':
        arr = arr.max() - arr
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    arr_u8 = arr.astype(np.uint8)
    img_rgb = np.stack([arr_u8, arr_u8, arr_u8], axis=2)

    processor = AutoImageProcessor.from_pretrained('microsoft/rad-dino')
    encoder = AutoModel.from_pretrained('microsoft/rad-dino', torch_dtype=torch.bfloat16)
    encoder.eval()
    with torch.no_grad():
        pv = processor(images=img_rgb, return_tensors='pt')['pixel_values'].bfloat16()
        cls = encoder(pixel_values=pv).last_hidden_state[:, 0, :].float()

    global_ckpt = torch.load(MODELS_DIR / 'global_head.pt', weights_only=True)
    global_mod = load_module('../src/04_train_global.py')
    g_model = global_mod.GlobalHead()
    g_model.load_state_dict(global_ckpt)
    g_model.eval()
    with torch.no_grad():
        global_pred = g_model(cls).clamp(0, 18).item()

    zone_ckpt = torch.load(MODELS_DIR / 'zone_head.pt', weights_only=True)
    zone_mod = load_module('../src/05_train_zones.py')
    z_model = zone_mod.ZoneHead()
    z_model.load_state_dict(zone_ckpt)
    z_model.eval()
    with torch.no_grad():
        zone_preds = z_model(cls).clamp(0, 3).squeeze(0).round().int().tolist()

    print(f"Global score : {global_pred:.1f} / 18")
    print("Zone scores  :")
    for zone, score in zip(ZONE_NAMES, zone_preds):
        print(f"  Zone {zone}: {score} / 3")

sample_dcm = next(Path('../data/dicom_raw/dicom_clean').glob('*.dcm'))
print(f"Running inference on: {sample_dcm.name}")
predict_dicom(str(sample_dcm))
"""))

nb.cells = cells

out_path = Path(__file__).parent / "brixia_walkthrough.ipynb"
with open(out_path, "w") as f:
    nbf.write(nb, f)
print(f"Notebook written to {out_path}")
