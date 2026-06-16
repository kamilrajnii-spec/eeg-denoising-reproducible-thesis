"""Run the Phase 3 EEGdenoiseNet held-out evaluation matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eeg_denoising.data_loading.load_eegdenoisenet import load_eegdenoisenet  # noqa: E402
from eeg_denoising.evaluation.metrics import rmse, rrmse, snr, snr_gain  # noqa: E402
from eeg_denoising.evaluation.phase3_helpers import summarize_metric  # noqa: E402
from eeg_denoising.models.dae import load_dae_checkpoint  # noqa: E402
from eeg_denoising.pipeline.hybrid_pipeline import apply_dae_to_epochs  # noqa: E402
from eeg_denoising.preprocessing.artifact_mixing import _as_2d_epochs  # noqa: E402
from train_phase2_dae import create_split_pairs, split_clean_epoch_indices  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "results" / "phase3"
CHECKPOINT_PATH = PROJECT_ROOT / "results" / "phase2" / "dae_best_model.pt"
PER_PAIR_PATH = OUTPUT_DIR / "full_evaluation_per_pair.csv"
TABLE_PATH = OUTPUT_DIR / "full_evaluation_table.csv"
SUMMARY_PATH = OUTPUT_DIR / "full_evaluation_summary.md"
BARPLOT_PATH = OUTPUT_DIR / "phase3_condition_barplot.png"


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        print("DAE checkpoint not found. Run scripts/train_phase2_dae.py first.")
        return 1

    test_split, manifest = build_test_split(args)
    model = load_dae_checkpoint(args.checkpoint, device=args.device)
    per_pair = evaluate_methods(test_split, manifest, model, args.device)
    summary = summarize_per_pair(per_pair)

    per_pair.to_csv(PER_PAIR_PATH, index=False)
    summary.to_csv(TABLE_PATH, index=False)
    write_summary_markdown(summary, per_pair, SUMMARY_PATH)
    save_condition_barplot(summary, BARPLOT_PATH)

    print(f"Created {PER_PAIR_PATH}")
    print(f"Created {TABLE_PATH}")
    print(f"Created {SUMMARY_PATH}")
    print(f"Created {BARPLOT_PATH}")

    return 0


def build_test_split(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Build the held-out EEGdenoiseNet test split without leakage."""
    eegdenoisenet = load_eegdenoisenet(PROJECT_ROOT / "data" / "eegdenoisenet")
    clean = _as_2d_epochs(eegdenoisenet["clean_eeg"])[: args.max_clean_epochs]
    eog = _as_2d_epochs(eegdenoisenet["eog"])
    emg = _as_2d_epochs(eegdenoisenet["emg"])

    split_indices = split_clean_epoch_indices(
        n_epochs=clean.shape[0],
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    test_split, manifest = create_split_pairs(
        split_name="test",
        clean=clean,
        eog=eog,
        emg=emg,
        clean_indices=split_indices["test"],
        snr_levels=args.snr_levels,
    )

    return test_split, manifest


def evaluate_methods(
    test_split: dict[str, np.ndarray],
    manifest: list[dict],
    model,
    device: str,
) -> pd.DataFrame:
    """Evaluate noisy, wavelet-only, and hybrid output per pair."""
    clean = test_split["clean"]
    raw_noisy = test_split["raw_noisy"]
    wavelet = test_split["dae_input"]
    hybrid = apply_dae_to_epochs(wavelet, model, device=device)

    method_outputs = {
        "noisy_input": raw_noisy,
        "wavelet_only": wavelet,
        "hybrid_wavelet_dae": hybrid,
    }

    rows = []
    for pair_id, pair_info in enumerate(manifest):
        for method, output in method_outputs.items():
            clean_epoch = clean[pair_id]
            noisy_epoch = raw_noisy[pair_id]
            output_epoch = output[pair_id]

            rows.append(
                {
                    "pair_id": pair_id,
                    "split": pair_info["split"],
                    "clean_epoch_id": pair_info["clean_epoch_id"],
                    "artifact_epoch_id": pair_info["artifact_epoch_id"],
                    "artifact_type": pair_info["artifact_type"],
                    "target_snr_db": pair_info["target_snr_db"],
                    "method": method,
                    "snr_db": snr(clean_epoch, output_epoch),
                    "snr_gain_db": snr_gain(clean_epoch, noisy_epoch, output_epoch),
                    "rmse": rmse(clean_epoch, output_epoch),
                    "rrmse": rrmse(clean_epoch, output_epoch),
                }
            )

    return pd.DataFrame(rows)


def summarize_per_pair(per_pair: pd.DataFrame) -> pd.DataFrame:
    """Create the requested condition-level metric table."""
    rows = []
    grouped = per_pair.groupby(["artifact_type", "target_snr_db", "method"])

    for (artifact_type, target_snr_db, method), group in grouped:
        row = {
            "artifact_type": artifact_type,
            "target_snr_db": target_snr_db,
            "method": method,
            "n_pairs": int(group["pair_id"].nunique()),
        }

        for metric in ["snr_db", "snr_gain_db", "rmse", "rrmse"]:
            metric_summary = summarize_metric(group[metric].to_numpy())
            row[f"mean_{metric}"] = metric_summary["mean"]
            row[f"std_{metric}"] = metric_summary["std"]

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["artifact_type", "target_snr_db", "method"]
    )


def write_summary_markdown(
    summary: pd.DataFrame,
    per_pair: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a short generated Markdown summary."""
    n_pairs = int(per_pair["pair_id"].nunique())
    n_conditions = summary[["artifact_type", "target_snr_db"]].drop_duplicates().shape[0]

    lines = [
        "# Phase 3 Full Evaluation Summary",
        "",
        "This file is generated by `scripts/run_phase3_evaluation.py`.",
        "",
        f"Held-out clean/noisy pairs evaluated per method: {n_pairs}",
        f"Artifact/SNR conditions: {n_conditions}",
        "",
        "Methods evaluated:",
        "",
        "- noisy input",
        "- DWT db4 level-4 wavelet-only",
        "- hybrid wavelet-to-DAE",
        "",
        "ICA is not compared on EEGdenoiseNet because these artifact pairs are "
        "single-channel epochs.",
        "",
        "## Mean SNR Gain by Condition",
        "",
        "| Artifact | SNR level | Method | Mean SNR gain dB | Mean RMSE |",
        "| --- | ---: | --- | ---: | ---: |",
    ]

    for _, row in summary.iterrows():
        lines.append(
            "| {artifact} | {snr_level:g} | {method} | {gain:.6f} | {rmse_value:.6f} |".format(
                artifact=row["artifact_type"],
                snr_level=row["target_snr_db"],
                method=row["method"],
                gain=row["mean_snr_gain_db"],
                rmse_value=row["mean_rmse"],
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_condition_barplot(summary: pd.DataFrame, output_path: Path) -> None:
    """Save a grouped barplot for mean SNR gain by condition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ["noisy_input", "wavelet_only", "hybrid_wavelet_dae"]
    conditions = (
        summary[["artifact_type", "target_snr_db"]]
        .drop_duplicates()
        .sort_values(["artifact_type", "target_snr_db"])
    )
    labels = [
        f"{row.artifact_type}\n{row.target_snr_db:g} dB"
        for row in conditions.itertuples(index=False)
    ]

    x = np.arange(len(labels))
    width = 0.25

    fig, axis = plt.subplots(figsize=(12, 5))
    for index, method in enumerate(methods):
        values = []
        for condition in conditions.itertuples(index=False):
            match = summary[
                (summary["artifact_type"] == condition.artifact_type)
                & (summary["target_snr_db"] == condition.target_snr_db)
                & (summary["method"] == method)
            ]
            values.append(float(match["mean_snr_gain_db"].iloc[0]))

        axis.bar(x + (index - 1) * width, values, width=width, label=method)

    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Mean SNR gain (dB)")
    axis.set_title("Phase 3 held-out SNR gain by artifact and SNR level")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


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

