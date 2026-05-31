"""Train a small MLP head to predict per-zone BrixIA scores (6 × ordinal 0–3).

Loads pre-extracted RAD-DINO embeddings from data/embeddings/.
No GPU required — trains in seconds on CPU.

Outputs:
  models/zone_head.pt        best checkpoint (lowest mean val MAE)
  models/zone_results.json   per-zone metrics + training history

Usage:
    python src/05_train_zones.py
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import numpy as np
from utils import EMBEDDINGS_DIR, MODELS_DIR, ZONE_NAMES

MODELS_DIR.mkdir(parents=True, exist_ok=True)


class ZoneHead(nn.Module):
    """MLP: 768 → 256 → ReLU → Dropout(0.3) → 6 (one regression output per zone)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load(split: str):
    X = torch.tensor(
        np.load(EMBEDDINGS_DIR / f"embeddings_{split}.npy"), dtype=torch.float32
    )
    y = torch.tensor(
        np.load(EMBEDDINGS_DIR / f"labels_zones_{split}.npy"), dtype=torch.float32
    )
    return X, y


def train(
    epochs: int = 150,
    lr: float = 1e-3,
    batch_size: int = 256,
    patience: int = 15,
    seed: int = 42,
):
    torch.manual_seed(seed)
    X_train, y_train = _load("train")
    X_val, y_val = _load("val")
    X_test, y_test = _load("test")

    model = ZoneHead()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_mae": []}
    n = len(X_train)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            preds = model(X_train[idx])
            loss = criterion(preds, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val).clamp(0.0, 3.0)
            val_mae = (val_preds - y_val).abs().mean().item()

        history["train_loss"].append(epoch_loss / n)
        history["val_mae"].append(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / "zone_head.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d} | "
                f"train_loss={epoch_loss/n:.4f} | "
                f"val_mae={val_mae:.4f} | "
                f"best={best_val_mae:.4f}"
            )

    # Final evaluation on test set
    model.load_state_dict(torch.load(MODELS_DIR / "zone_head.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test).clamp(0.0, 3.0)
        per_zone_mae = (test_preds - y_test).abs().mean(dim=0).tolist()
        mean_mae = sum(per_zone_mae) / 6

    print(f"\n=== Test Results ===")
    for zone, mae in zip(ZONE_NAMES, per_zone_mae):
        print(f"  Zone {zone} MAE: {mae:.4f}")
    print(f"  Mean  MAE: {mean_mae:.4f}")
    print(f"Best val MAE: {best_val_mae:.4f}")

    results = {
        "per_zone_mae": dict(zip(ZONE_NAMES, per_zone_mae)),
        "mean_mae": mean_mae,
        "best_val_mae": best_val_mae,
        "history": history,
    }
    with open(MODELS_DIR / "zone_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved models/zone_head.pt and models/zone_results.json")

    return model, results


if __name__ == "__main__":
    train()
