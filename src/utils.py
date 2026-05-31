from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data"
DICOM_DIR = DATA_ROOT / "dicom_raw" / "dicom_clean"
PNG_DIR = DATA_ROOT / "png"
EMBEDDINGS_DIR = DATA_ROOT / "embeddings"
PLOTS_DIR = DATA_ROOT / "plots"
MODELS_DIR = ROOT / "models"
METADATA_GLOBAL = ROOT / "osfstorage-archive" / "data" / "metadata_global_v2.csv"
METADATA_CONSENSUS = ROOT / "osfstorage-archive" / "data" / "metadata_consensus_v1.csv"

ZONE_NAMES = ["A", "B", "C", "D", "E", "F"]


def load_metadata():
    """Load both CSVs and parse BrixiaScore string into per-zone integer columns.

    Returns:
        global_df: DataFrame with columns ZoneA..ZoneF, BrixiaScoreGlobal, is_consensus
        consensus_df: DataFrame with mean/mode per zone for the 150-image test set
    """
    global_df = pd.read_csv(METADATA_GLOBAL, sep=";", encoding="utf-8-sig")
    consensus_df = pd.read_csv(METADATA_CONSENSUS, encoding="utf-8-sig")

    def parse_score(s):
        s = str(int(s)).zfill(6)
        return [int(c) for c in s]

    parsed = global_df["BrixiaScore"].apply(parse_score)
    for i, zone in enumerate(ZONE_NAMES):
        global_df[f"Zone{zone}"] = parsed.apply(lambda x: x[i])

    consensus_filenames = set(consensus_df["Filename"].values)
    global_df["is_consensus"] = global_df["Filename"].isin(consensus_filenames)

    return global_df, consensus_df


def get_splits(seed=42):
    """Split filenames into train / val / test.

    Test = 150-image consensus set (never seen during training).
    Remaining ~4,544 split 90/10 train/val with fixed seed.

    Returns:
        (train_files, val_files, test_files) — lists of Filename strings
    """
    global_df, _ = load_metadata()

    test_df = global_df[global_df["is_consensus"]].copy()
    remaining_df = global_df[~global_df["is_consensus"]].copy()

    train_df, val_df = train_test_split(
        remaining_df, test_size=0.1, random_state=seed, shuffle=True
    )

    return (
        train_df["Filename"].tolist(),
        val_df["Filename"].tolist(),
        test_df["Filename"].tolist(),
    )


class BrixiaDataset(Dataset):
    """PyTorch Dataset for BrixIA chest X-rays.

    Loads 16-bit grayscale PNGs, converts to 8-bit RGB, applies processor or
    transform, returns (pixel_values, global_score, zone_scores).

    Args:
        filenames: list of Filename strings (e.g. "12345.dcm")
        global_df: DataFrame from load_metadata()
        processor: HuggingFace AutoImageProcessor (used by embedding extractor)
        transform: torchvision transforms (used when processor is None)
    """

    def __init__(self, filenames, global_df, processor=None, transform=None):
        self.filenames = filenames
        self.df = global_df.set_index("Filename")
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        png_path = PNG_DIR / fname.replace(".dcm", ".png")

        # Load 16-bit PNG with OpenCV (IMREAD_UNCHANGED preserves uint16)
        arr = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)  # shape (H, W), uint16
        arr_8bit = (arr / 256).astype(np.uint8)
        img_rgb = np.stack([arr_8bit, arr_8bit, arr_8bit], axis=2)  # (H, W, 3)

        if self.processor is not None:
            inputs = self.processor(images=img_rgb, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)
        else:
            img_pil = Image.fromarray(img_rgb)
            pixel_values = self.transform(img_pil)

        row = self.df.loc[fname]
        global_score = torch.tensor(float(row["BrixiaScoreGlobal"]), dtype=torch.float32)
        zone_scores = torch.tensor(
            [float(row[f"Zone{z}"]) for z in ZONE_NAMES], dtype=torch.float32
        )

        return pixel_values, global_score, zone_scores
