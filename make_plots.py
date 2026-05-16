from pathlib import Path
from typing import Tuple
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _read_results_file(path: Path) -> pd.DataFrame:
    # Some files are tab-separated but saved as .csv.
    df = pd.read_csv(path, sep="\t")
    if len(df.columns) == 1 and "\t" in df.columns[0]:
        df = pd.read_csv(path, sep="\t", engine="python")
    return df


def _load_all_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for file_path in sorted(results_dir.glob("*_results.csv")):
        m = re.match(
            r"^(?P<model>.+)_(?P<policy>[^_]+)_kcore_(?P<kcore>\d+)_results\.csv$",
            file_path.name,
        )
        if not m:
            continue
        model_name = m.group("model")
        policy = m.group("policy")
        kcore = int(m.group("kcore"))
        df = _read_results_file(file_path)
        df.columns = [c.strip().lower() for c in df.columns]

        # Expected columns: metric, 0 (unliked), 1 (liked)
        for _, row in df.iterrows():
            rows.append(
                {
                    "model": model_name,
                    "policy": policy,
                    "kcore": kcore,
                    "metric": str(row["metric"]).strip().lower(),
                    "unliked": float(row["0 (unliked)"]),
                    "liked": float(row["1 (liked)"]),
                }
            )

    if not rows:
        raise FileNotFoundError("No *_results.csv files found in results/")
    return pd.DataFrame(rows)


def _compute_ylim(values: np.ndarray) -> Tuple[float, float]:
    vmin = float(values.min())
    vmax = float(values.max())
    low = np.floor((vmin - 0.02) * 100) / 100
    high = np.ceil((vmax + 0.06) * 100) / 100
    if high - low < 0.06:
        pad = (0.06 - (high - low)) / 2
        low -= pad
        high += pad
    return max(0.0, low), min(1.0, high)


def _pretty_metric(metric: str) -> str:
    mapping = {
        "f1_score": "F1-Score",
        "precision": "Precision",
        "recall": "Recall",
    }
    return mapping.get(metric, metric.replace("_", " ").title())


def plot_grouped_bars(df: pd.DataFrame) -> None:
    # Keep only core classification metrics.
    df = df[df["metric"].isin(["precision", "recall", "f1_score"])].copy()
    grouped = df.groupby(["policy", "kcore", "metric"], sort=True)
    for (policy, kcore, metric), d0 in grouped:
        d = d0.sort_values("model").copy()
        pretty_metric = _pretty_metric(metric)

        models = d["model"].tolist()
        unliked = d["unliked"].round(2).to_numpy()
        liked = d["liked"].round(2).to_numpy()

        x = np.arange(len(models))
        width = 0.36

        fig, ax = plt.subplots(figsize=(max(10, len(models) * 0.8), 7))
        b1 = ax.bar(x - width / 2, unliked, width, label="0 (unliked)")
        b2 = ax.bar(x + width / 2, liked, width, label="1 (liked)")

        ax.set_title(f"{pretty_metric} | k-core={kcore} | policy={policy}")
        ax.set_xlabel("model")
        ax.set_ylabel(pretty_metric)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.legend()

        random_row = d[d["model"] == "random"]
        if not random_row.empty:
            random_unliked = float(random_row["unliked"].iloc[0])
            ax.axhline(
                y=random_unliked,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label="random unliked",
            )

        all_vals = np.concatenate([unliked, liked])
        y0, y1 = _compute_ylim(all_vals)
        ax.set_ylim(y0, y1)
        ticks = np.arange(y0, y1 + 0.001, 0.01)
        # Avoid overlapping y-axis labels when the range is wide.
        if len(ticks) > 18:
            ticks = ticks[::2]
        ax.set_yticks(ticks)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

        label_offset = (y1 - y0) * 0.015
        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + label_offset,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        
        # Extra bottom/left padding helps avoid axis-label overlap.
        fig.tight_layout()
        fig.subplots_adjust(left=0.12, bottom=0.22)
        out_path = OUTPUT_DIR / f"{metric}_{policy}_kcore_{kcore}_grouped_barplot.pdf"
        fig.savefig(out_path)
        out_path = OUTPUT_DIR / f"{metric}_{policy}_kcore_{kcore}_grouped_barplot.png"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    all_results = _load_all_results(RESULTS_DIR)
    plot_grouped_bars(all_results)
