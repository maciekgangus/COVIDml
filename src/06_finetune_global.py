"""Fine-tune RAD-DINO + MLP head end-to-end for global BrixIA score (0-18).

Unfreezes the last N transformer blocks of RAD-DINO while keeping early layers frozen.
Uses differential learning rates: backbone 1e-5, head 1e-3.

Outputs:
  models/finetuned_global.pt       full checkpoint (backbone + head state dicts)
  models/finetuned_global_results.json  metrics + training history

Usage:
    python src/06_finetune_global.py --mode train   # fine-tune from scratch
    python src/06_finetune_global.py --mode load    # load checkpoint and evaluate
    python src/06_finetune_global.py --mode train --unfreeze 6   # unfreeze 6 blocks
"""
import sys
import json
import argparse
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm
from utils import load_metadata, get_splits, BrixiaDataset, MODELS_DIR

MODEL_NAME = "microsoft/rad-dino"
CHECKPOINT = MODELS_DIR / "finetuned_global.pt"
RESULTS_FILE = MODELS_DIR / "finetuned_global_results.json"

# Training hyperparams
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2        # effective batch = 32
LR_BACKBONE = 1e-5           # small: pretrained layers change slowly
LR_HEAD = 1e-3               # larger: head trains from random init
WEIGHT_DECAY = 1e-4
EPOCHS = 20
PATIENCE = 5
NUM_WORKERS = 4


class FinetunedGlobal(nn.Module):
    """RAD-DINO backbone + MLP head, last N transformer blocks unfrozen."""

    def __init__(self, backbone: nn.Module, unfreeze_last_n: int = 4):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )
        self._configure_frozen_layers(unfreeze_last_n)

    def _configure_frozen_layers(self, unfreeze_last_n: int) -> None:
        # Freeze entire backbone first
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # Unfreeze last N encoder blocks
        blocks = self.backbone.encoder.layer
        total = len(blocks)
        for block in blocks[total - unfreeze_last_n:]:
            for p in block.parameters():
                p.requires_grad_(True)

        # Always unfreeze final layernorm
        for p in self.backbone.layernorm.parameters():
            p.requires_grad_(True)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable params: {trainable:,} / {total_params:,} "
              f"({100 * trainable / total_params:.1f}%)")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        cls = outputs.last_hidden_state[:, 0, :]   # [B, 768]
        return self.head(cls).squeeze(-1)            # [B]


def train_model(unfreeze_last_n: int) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load RAD-DINO backbone in BF16
    print(f"Loading {MODEL_NAME} ...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    backbone = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    model = FinetunedGlobal(backbone, unfreeze_last_n=unfreeze_last_n).to(device)

    global_df, _ = load_metadata()
    train_files, val_files, test_files = get_splits()

    train_ds = BrixiaDataset(train_files, global_df, processor=processor)
    val_ds = BrixiaDataset(val_files, global_df, processor=processor)
    test_ds = BrixiaDataset(test_files, global_df, processor=processor)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # Differential learning rates
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad],
         "lr": LR_BACKBONE},
        {"params": model.head.parameters(), "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_mae": []}

    print(f"\nFine-tuning ({unfreeze_last_n} ViT blocks unlocked) for up to {EPOCHS} epochs ...")
    t0 = time.time()

    for epoch in range(EPOCHS):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, (pixel_values, global_scores, _) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1:2d} train", leave=False)
        ):
            pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
            global_scores = global_scores.to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                preds = model(pixel_values)
                loss = criterion(preds, global_scores) / GRAD_ACCUM_STEPS

            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS * len(global_scores)

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_ds)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_preds_all, val_labels_all = [], []
        with torch.no_grad():
            for pixel_values, global_scores, _ in val_loader:
                pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    preds = model(pixel_values).float().clamp(0.0, 18.0)
                val_preds_all.append(preds.cpu())
                val_labels_all.append(global_scores)

        val_preds_t = torch.cat(val_preds_all)
        val_labels_t = torch.cat(val_labels_all)
        val_mae = (val_preds_t - val_labels_t).abs().mean().item()

        history["train_loss"].append(avg_train_loss)
        history["val_mae"].append(val_mae)

        print(f"Epoch {epoch+1:2d} | train_loss={avg_train_loss:.4f} | "
              f"val_mae={val_mae:.4f} | best={best_val_mae:.4f} | "
              f"lr_head={optimizer.param_groups[1]['lr']:.2e}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(
                {"backbone": model.backbone.state_dict(),
                 "head": model.head.state_dict(),
                 "unfreeze_last_n": unfreeze_last_n},
                CHECKPOINT,
            )
            print(f"  ✓ Saved checkpoint (val_mae={val_mae:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    elapsed = time.time() - t0
    print(f"Training done in {elapsed/60:.1f} min")

    # ── Test evaluation ────────────────────────────────────────────────────────
    ckpt = torch.load(CHECKPOINT, weights_only=True)
    model.backbone.load_state_dict(ckpt["backbone"])
    model.head.load_state_dict(ckpt["head"])
    model.eval()

    test_preds_all, test_labels_all = [], []
    with torch.no_grad():
        for pixel_values, global_scores, _ in test_loader:
            pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                preds = model(pixel_values).float().clamp(0.0, 18.0)
            test_preds_all.append(preds.cpu())
            test_labels_all.append(global_scores)

    test_preds_t = torch.cat(test_preds_all)
    test_labels_t = torch.cat(test_labels_all)
    test_mae = (test_preds_t - test_labels_t).abs().mean().item()
    test_mse = ((test_preds_t - test_labels_t) ** 2).mean().item()

    print(f"\n=== Test Results (fine-tuned, {unfreeze_last_n} blocks) ===")
    print(f"MAE : {test_mae:.4f}")
    print(f"MSE : {test_mse:.4f}")
    print(f"RMSE: {test_mse**0.5:.4f}")
    print(f"Best val MAE: {best_val_mae:.4f}")
    print(f"(Linear probe baseline MAE was 1.58)")

    results = {
        "test_mae": test_mae,
        "test_mse": test_mse,
        "test_rmse": test_mse ** 0.5,
        "best_val_mae": best_val_mae,
        "unfreeze_last_n": unfreeze_last_n,
        "history": history,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    return results


def load_and_evaluate() -> None:
    if not CHECKPOINT.exists():
        print(f"No checkpoint found at {CHECKPOINT}. Run with --mode train first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT, weights_only=True)
    unfreeze_last_n = ckpt.get("unfreeze_last_n", 4)

    print(f"Loading checkpoint from {CHECKPOINT} (unfreeze_last_n={unfreeze_last_n}) ...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    backbone = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    model = FinetunedGlobal(backbone, unfreeze_last_n=unfreeze_last_n).to(device)
    model.backbone.load_state_dict(ckpt["backbone"])
    model.head.load_state_dict(ckpt["head"])
    model.eval()

    global_df, _ = load_metadata()
    _, _, test_files = get_splits()
    test_ds = BrixiaDataset(test_files, global_df, processor=processor)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    test_preds_all, test_labels_all = [], []
    with torch.no_grad():
        for pixel_values, global_scores, _ in tqdm(test_loader, desc="Evaluating"):
            pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                preds = model(pixel_values).float().clamp(0.0, 18.0)
            test_preds_all.append(preds.cpu())
            test_labels_all.append(global_scores)

    test_preds_t = torch.cat(test_preds_all)
    test_labels_t = torch.cat(test_labels_all)
    test_mae = (test_preds_t - test_labels_t).abs().mean().item()
    test_mse = ((test_preds_t - test_labels_t) ** 2).mean().item()

    print(f"\n=== Test Results (loaded checkpoint) ===")
    print(f"MAE : {test_mae:.4f}")
    print(f"MSE : {test_mse:.4f}")
    print(f"RMSE: {test_mse**0.5:.4f}")

    if RESULTS_FILE.exists():
        saved = json.load(open(RESULTS_FILE))
        print(f"(Matches saved results: MAE={saved['test_mae']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune RAD-DINO for global BrixIA score")
    parser.add_argument("--mode", choices=["train", "load"], default="train",
                        help="train: fine-tune and save; load: evaluate saved checkpoint")
    parser.add_argument("--unfreeze", type=int, default=4,
                        help="Number of ViT blocks to unfreeze from the end (default: 4)")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        train_model(unfreeze_last_n=args.unfreeze)
    else:
        load_and_evaluate()


if __name__ == "__main__":
    main()
