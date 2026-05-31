"""Run inference on a single DICOM using both trained models.

Loads RAD-DINO once, extracts the CLS embedding, then runs both heads:
  - global_head.pt  → single score 0–18
  - zone_head.pt    → six zone scores 0–3 (zones A–F)

Usage:
    python src/predict.py path/to/image.dcm
    python src/predict.py path/to/image.dcm --device cpu
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import numpy as np
import pydicom
import cv2
from transformers import AutoModel, AutoImageProcessor

from utils import MODELS_DIR, ZONE_NAMES

MODEL_NAME = "microsoft/rad-dino"


class GlobalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ZoneHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 6)
        )

    def forward(self, x):
        return self.net(x)


def load_models(device: torch.device):
    """Load RAD-DINO backbone + both heads. Returns (processor, backbone, global_head, zone_head)."""
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    backbone = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    global_head = GlobalHead()
    global_head.load_state_dict(
        torch.load(MODELS_DIR / "global_head.pt", weights_only=True, map_location="cpu")
    )
    global_head.eval()

    zone_head = ZoneHead()
    zone_head.load_state_dict(
        torch.load(MODELS_DIR / "zone_head.pt", weights_only=True, map_location="cpu")
    )
    zone_head.eval()

    return processor, backbone, global_head, zone_head


def dicom_to_rgb(dcm_path: str) -> np.ndarray:
    """Read a DICOM, normalise to uint8, return (H, W, 3) RGB array."""
    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    arr_u8 = arr.astype(np.uint8)
    return np.stack([arr_u8, arr_u8, arr_u8], axis=2)


def predict(dcm_path: str, device: torch.device | None = None):
    """Return (global_score, zone_scores_dict) for a DICOM file.

    Args:
        dcm_path: path to a .dcm file
        device: torch device (defaults to cuda if available, else cpu)

    Returns:
        global_score: float, predicted severity 0–18
        zone_scores: dict mapping zone name → int score 0–3
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor, backbone, global_head, zone_head = load_models(device)

    img_rgb = dicom_to_rgb(dcm_path)
    pixel_values = processor(images=img_rgb, return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(device, dtype=torch.bfloat16)

    with torch.no_grad():
        cls = backbone(pixel_values=pixel_values).last_hidden_state[:, 0, :].float().cpu()
        global_score = global_head(cls).clamp(0, 18).item()
        zone_raw = zone_head(cls).clamp(0, 3).squeeze(0)
        zone_scores = {z: int(zone_raw[i].round().item()) for i, z in enumerate(ZONE_NAMES)}

    return global_score, zone_scores


def main():
    parser = argparse.ArgumentParser(description="BrixIA severity prediction from DICOM")
    parser.add_argument("dicom", help="Path to .dcm file")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    global_score, zone_scores = predict(args.dicom, device=device)

    print(f"\nGlobal score : {global_score:.1f} / 18")
    print("Zone scores  :")
    for zone, score in zone_scores.items():
        bar = "█" * score + "░" * (3 - score)
        print(f"  Zone {zone}: {score}/3  {bar}")
    print(f"\nReconstructed sum: {sum(zone_scores.values())} / 18")


if __name__ == "__main__":
    main()
