"""Run Phase 3 ERP-like morphology preservation analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eeg_denoising.evaluation.phase3_helpers import (  # noqa: E402
    morphology_metrics,
    summarize_metric,
)
from eeg_denoising.models.dae import load_dae_checkpoint  # noqa: E402
from eeg_denoising.pipeline.hybrid_pipeline import apply_dae_to_epochs  # noqa: E402
from run_phase3_evaluation import build_test_split  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "results" / "phase3"
CHECKPOINT_PATH = PROJECT_ROOT / "results" / "phase2" / "dae_best_model.pt"
TABLE_PATH = OUTPUT_DIR / "erp_preservation_table.csv"
PLOT_PATH = OUTPUT_DIR / "erp_comparison_plot.png"


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        print("DAE checkpoint not found. Run scripts/train_phase2_dae.py first.")
        return 1

    test_split, manifest = build_test_split(args)
    model = load_dae_checkpoint(args.checkpoint, device=args.device)
    hybrid = apply_dae_to_epochs(test_split["dae_input"], model, device=args.device)
    per_pair = compute_morphology_rows(test_split, manifest, hybrid)
    summary = summarize_morphology(per_pair)

    summary.to_csv(TABLE_PATH, index=False)
    save_erp_like_plot(test_split, manifest, hybrid, PLOT_PATH)

    print(f"Created {TABLE_PATH}")
    print(f"Created {PLOT_PATH}")

    return 0


def compute_morphology_rows(
    test_split: dict,
    manifest: list[dict],
    hybrid,
) -> pd.DataFrame:
    """Compute per-pair morphology metrics for wavelet and hybrid outputs."""
    rows = []
    outputs = {
        "wavelet_only": test_split["dae_input"],
        "hybrid_wavelet_dae": hybrid,
    }

    for pair_id, pair_info in enumerate(manifest):
        clean = test_split["clean"][pair_id]
        for method, output in outputs.items():
            metrics = morphology_metrics(clean, output[pair_id])
            rows.append(
                {
                    "pair_id": pair_id,
                    "artifact_type": pair_info["artifact_type"],
                    "target_snr_db": pair_info["target_snr_db"],
                    "method": method,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def summarize_morphology(per_pair: pd.DataFrame) -> pd.DataFrame:
    """Summarize ERP-like morphology metrics by condition."""
    rows = []
    metrics = [
        "pearson_correlation",
        "peak_amplitude_difference",
        "peak_latency_difference_samples",
    ]

    grouped = per_pair.groupby(["artifact_type", "target_snr_db", "method"])
    for (artifact_type, target_snr_db, method), group in grouped:
        row = {
            "analysis_name": "ERP-like morphology preservation",
            "artifact_type": artifact_type,
            "target_snr_db": target_snr_db,
            "method": method,
            "n_pairs": int(group["pair_id"].nunique()),
        }

        for metric in metrics:
            summary = summarize_metric(group[metric].to_numpy())
            row[f"mean_{metric}"] = summary["mean"]
            row[f"std_{metric}"] = summary["std"]

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["artifact_type", "target_snr_db", "method"]
    )


def save_erp_like_plot(test_split: dict, manifest: list[dict], hybrid, output_path: Path) -> None:
    """Save one representative clean/noisy/wavelet/hybrid morphology plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected_index = choose_representative_pair(manifest)
    clean = test_split["clean"][selected_index]
    noisy = test_split["raw_noisy"][selected_index]
    wavelet = test_split["dae_input"][selected_index]
    hybrid_signal = hybrid[selected_index]

    signals = [
        ("Clean reference", clean),
        ("Noisy input", noisy),
        ("Wavelet-only output", wavelet),
        ("Hybrid output", hybrid_signal),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for axis, (title, signal) in zip(axes, signals):
        axis.plot(signal, linewidth=1.0)
        axis.set_title(title)
        axis.set_ylabel("Amplitude")

    pair_info = manifest[selected_index]
    axes[-1].set_xlabel("Sample index")
    fig.suptitle(
        "ERP-like morphology preservation example: "
        f"{pair_info['artifact_type']} at {pair_info['target_snr_db']:g} dB",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def choose_representative_pair(manifest: list[dict]) -> int:
    """Prefer mixed artifact at 0 dB for the visual example."""
    for index, pair_info in enumerate(manifest):
        if pair_info["artifact_type"] == "mixed" and pair_info["target_snr_db"] == 0.0:
            return index

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--max-clean-epochs", type=int, default=600)
    parser.add_argument("--snr-levels", type=float, nargs="+", default=[-5.0, 0.0, 5.0])
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

