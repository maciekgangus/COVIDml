"""Extract frozen RAD-DINO CLS-token embeddings for all dataset splits.

Runs inference once and saves embeddings + labels as .npy arrays.
Subsequent training loads these arrays directly — no GPU needed.

Outputs in data/embeddings/:
  embeddings_{train,val,test}.npy    shape [N, 768]
  labels_global_{train,val,test}.npy shape [N]
  labels_zones_{train,val,test}.npy  shape [N, 6]

Usage:
    python src/03_extract_embeddings.py
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm
from utils import load_metadata, get_splits, BrixiaDataset, EMBEDDINGS_DIR

MODEL_NAME = "microsoft/rad-dino"
BATCH_SIZE = 64
NUM_WORKERS = 4


def extract_split(model, processor, filenames, global_df, split_name, device):
    dataset = BrixiaDataset(filenames, global_df, processor=processor)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
    )

    all_embeddings, all_global, all_zones = [], [], []

    with torch.no_grad():
        for pixel_values, global_scores, zone_scores in tqdm(loader, desc=f"  {split_name}"):
            pv = pixel_values.to(device, dtype=torch.bfloat16)
            outputs = model(pixel_values=pv)
            # CLS token is first token of last_hidden_state
            cls = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
            all_embeddings.append(cls)
            all_global.append(global_scores.numpy())
            all_zones.append(zone_scores.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)   # [N, 768]
    global_labels = np.concatenate(all_global, axis=0)    # [N]
    zone_labels = np.concatenate(all_zones, axis=0)        # [N, 6]

    np.save(EMBEDDINGS_DIR / f"embeddings_{split_name}.npy", embeddings)
    np.save(EMBEDDINGS_DIR / f"labels_global_{split_name}.npy", global_labels)
    np.save(EMBEDDINGS_DIR / f"labels_zones_{split_name}.npy", zone_labels)

    print(f"  {split_name}: embeddings {embeddings.shape}, "
          f"global_labels {global_labels.shape}, zone_labels {zone_labels.shape}")


def main():
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {MODEL_NAME} ...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Model loaded  ({sum(p.numel() for p in model.parameters()):,} params frozen)")

    global_df, _ = load_metadata()
    train_files, val_files, test_files = get_splits()

    print("\nExtracting embeddings ...")
    t0 = time.time()
    for split_name, files in [
        ("train", train_files),
        ("val", val_files),
        ("test", test_files),
    ]:
        extract_split(model, processor, files, global_df, split_name, device)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
