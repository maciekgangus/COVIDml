"""Convert BrixIA DICOM files to 16-bit grayscale PNGs.

Handles MONOCHROME1 inversion (some manufacturers store inverted intensity).
Parallelized across all CPU cores. Idempotent — skips existing PNGs.

Usage:
    python src/01_extract_dicom.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import cv2
import pydicom
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from utils import DICOM_DIR, PNG_DIR


def convert_single(dcm_path: Path):
    """Convert one DICOM file to 16-bit PNG. Returns (filename, status_str)."""
    png_path = PNG_DIR / (dcm_path.stem + ".png")
    if png_path.exists():
        return dcm_path.name, "skipped"

    try:
        ds = pydicom.dcmread(str(dcm_path))
        arr = ds.pixel_array.astype(np.float32)

        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr

        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 65535.0
        arr_u16 = arr.astype(np.uint16)

        cv2.imwrite(str(png_path), arr_u16)
        return dcm_path.name, "ok"
    except Exception as exc:
        return dcm_path.name, f"error: {exc}"


def main():
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    dcm_files = sorted(DICOM_DIR.glob("*.dcm"))
    if not dcm_files:
        print(f"No DICOM files found in {DICOM_DIR}")
        print("Make sure extraction from dicom_archive_v2.tar.gz.* is complete.")
        return

    print(f"Found {len(dcm_files)} DICOM files → converting to {PNG_DIR}")
    workers = cpu_count()
    print(f"Using {workers} CPU cores")

    with Pool(workers) as pool:
        results = list(tqdm(
            pool.imap(convert_single, dcm_files),
            total=len(dcm_files),
            desc="DICOM→PNG",
        ))

    ok = sum(1 for _, s in results if s == "ok")
    skipped = sum(1 for _, s in results if s == "skipped")
    errors = [(f, s) for f, s in results if s.startswith("error")]

    print(f"\nDone: {ok} converted, {skipped} skipped, {len(errors)} errors")
    for fname, err in errors[:10]:
        print(f"  {fname}: {err}")


if __name__ == "__main__":
    main()
