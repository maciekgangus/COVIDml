"""Exploratory data analysis for the BrixIA dataset.

Produces plots saved to data/plots/:
  - global_score_dist.png
  - zone_score_dists.png
  - image_dimensions.png

Usage:
    python src/02_analyze_dataset.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils import load_metadata, get_splits, PLOTS_DIR, ZONE_NAMES

sns.set_theme(style="whitegrid", font_scale=1.1)


def plot_global_distribution(global_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    counts = global_df["BrixiaScoreGlobal"].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("BrixiaScoreGlobal (0–18)")
    ax.set_ylabel("Number of images")
    ax.set_title(f"Global Score Distribution  (n={len(global_df):,})")
    ax.set_xticks(range(19))
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "global_score_dist.png", dpi=150)
    plt.close()
    print("Saved global_score_dist.png")


def plot_zone_distributions(global_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=False)
    colors = sns.color_palette("Set2", 4)
    for ax, zone in zip(axes.flat, ZONE_NAMES):
        counts = global_df[f"Zone{zone}"].value_counts().sort_index()
        ax.bar(counts.index, counts.values, color=[colors[i] for i in counts.index], edgecolor="white")
        ax.set_title(f"Zone {zone}")
        ax.set_xlabel("Score (0–3)")
        ax.set_ylabel("Count")
        ax.set_xticks([0, 1, 2, 3])
    plt.suptitle("Per-Zone Score Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "zone_score_dists.png", dpi=150)
    plt.close()
    print("Saved zone_score_dists.png")


def plot_image_dimensions(global_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    global_df["Columns"].hist(bins=40, ax=axes[0], color="#DD8452", edgecolor="white")
    axes[0].set_title("Image Width (Columns)")
    axes[0].set_xlabel("pixels")
    global_df["Rows"].hist(bins=40, ax=axes[1], color="#55A868", edgecolor="white")
    axes[1].set_title("Image Height (Rows)")
    axes[1].set_xlabel("pixels")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "image_dimensions.png", dpi=150)
    plt.close()
    print("Saved image_dimensions.png")


def print_summary(global_df: pd.DataFrame) -> None:
    train_files, val_files, test_files = get_splits()

    print("\n=== Dataset Summary ===")
    print(f"Total images : {len(global_df):,}")
    print(f"  Train      : {len(train_files):,}")
    print(f"  Val        : {len(val_files):,}")
    print(f"  Test       : {len(test_files):,}  (consensus set)")
    print(f"\nModalities   : {dict(global_df['Modality'].value_counts())}")
    print(f"\nManufacturers:")
    for mfr, cnt in global_df["Manufacturer"].value_counts().items():
        print(f"  {mfr:<35} {cnt:>5}")

    print("\nClass imbalance per zone (score: count):")
    for zone in ZONE_NAMES:
        counts = global_df[f"Zone{zone}"].value_counts().sort_index()
        print(f"  Zone {zone}: {dict(counts)}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    global_df, _ = load_metadata()

    plot_global_distribution(global_df)
    plot_zone_distributions(global_df)
    plot_image_dimensions(global_df)
    print_summary(global_df)


if __name__ == "__main__":
    main()
