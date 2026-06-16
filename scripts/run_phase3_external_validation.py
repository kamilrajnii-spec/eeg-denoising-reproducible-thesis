"""Run Phase 3 external validation on PhysioNet Motor Imagery epochs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_denoising.data_loading.load_eegdenoisenet import load_eegdenoisenet  # noqa: E402
from eeg_denoising.evaluation.metrics import rmse, rrmse, snr, snr_gain  # noqa: E402
from eeg_denoising.evaluation.phase3_helpers import summarize_metric  # noqa: E402
from eeg_denoising.models.dae import load_dae_checkpoint  # noqa: E402
from eeg_denoising.pipeline.hybrid_pipeline import apply_dae_to_epochs  # noqa: E402
from eeg_denoising.preprocessing.artifact_mixing import (  # noqa: E402
    _as_2d_epochs,
    create_artifact_pairs,
)
from eeg_denoising.wavelet.dwt_denoising import denoise_epochs_dwt  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "results" / "phase3"
CHECKPOINT_PATH = PROJECT_ROOT / "results" / "phase2" / "dae_best_model.pt"
TABLE_PATH = OUTPUT_DIR / "external_validation_table.csv"
PLOT_PATH = OUTPUT_DIR / "external_validation_plot.png"
PHYSIONET_ROOT = PROJECT_ROOT / "data" / "physionet_mi"


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        print("DAE checkpoint not found. Run scripts/train_phase2_dae.py first.")
        return 1

    clean_epochs, epoch_sources = load_physionet_epochs(args)
    eegdenoisenet = load_eegdenoisenet(PROJECT_ROOT / "data" / "eegdenoisenet")
    eog = _as_2d_epochs(eegdenoisenet["eog"])
    emg = _as_2d_epochs(eegdenoisenet["emg"])

    model = load_dae_checkpoint(args.checkpoint, device=args.device)
    per_pair = evaluate_external_pairs(clean_epochs, epoch_sources, eog, emg, model, args)
    summary = summarize_external(per_pair)

    summary.to_csv(TABLE_PATH, index=False)
    save_external_plot(summary, PLOT_PATH)

    print(f"Created {TABLE_PATH}")
    print(f"Created {PLOT_PATH}")

    return 0


def load_physionet_epochs(args: argparse.Namespace) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Extract 512-sample epochs from one or more PhysioNet EDF files."""
    import mne

    edf_files = sorted(PHYSIONET_ROOT.rglob("*.edf"))[: args.max_files]
    if not edf_files:
        raise FileNotFoundError(
            "Dataset not found. Please download PhysioNet EDF files into "
            f"{PHYSIONET_ROOT}/"
        )

    epochs = []
    sources = []

    for edf_file in edf_files:
        raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
        raw.pick("eeg")
        data = raw.get_data()[: args.max_channels]
        subject_id = edf_file.parent.name

        for channel_index, channel_name in enumerate(raw.ch_names[: args.max_channels]):
            channel_signal = data[channel_index]
            for start in range(0, channel_signal.size - args.epoch_length + 1, args.epoch_length):
                if len(epochs) >= args.max_epochs:
                    return np.asarray(epochs, dtype=float), sources

                end = start + args.epoch_length
                epochs.append(channel_signal[start:end])
                sources.append(
                    {
                        "subject_id": subject_id,
                        "edf_file": str(edf_file.relative_to(PROJECT_ROOT)),
                        "channel": channel_name,
                        "start_sample": start,
                        "end_sample": end,
                    }
                )

    return np.asarray(epochs, dtype=float), sources


def evaluate_external_pairs(
    clean_epochs: np.ndarray,
    epoch_sources: list[dict[str, object]],
    eog: np.ndarray,
    emg: np.ndarray,
    model,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Inject artifacts into PhysioNet epochs and evaluate denoising methods."""
    pairs = create_artifact_pairs(clean_epochs, eog, emg, snr_levels_db=args.snr_levels)
    rows = []

    for pair in pairs:
        wavelet = denoise_epochs_dwt(pair.noisy)
        hybrid = apply_dae_to_epochs(wavelet, model, device=args.device)
        outputs = {
            "noisy_input": pair.noisy,
            "wavelet_only": wavelet,
            "hybrid_wavelet_dae": hybrid,
        }

        for epoch_id, source in enumerate(epoch_sources):
            for method, output in outputs.items():
                rows.append(
                    {
                        "dataset": "PhysioNet Motor Imagery",
                        "subject_id": source["subject_id"],
                        "edf_file": source["edf_file"],
                        "channel": source["channel"],
                        "start_sample": source["start_sample"],
                        "artifact_type": pair.artifact_type,
                        "target_snr_db": pair.target_snr_db,
                        "method": method,
                        "snr_db": snr(pair.clean[epoch_id], output[epoch_id]),
                        "snr_gain_db": snr_gain(
                            pair.clean[epoch_id],
                            pair.noisy[epoch_id],
                            output[epoch_id],
                        ),
                        "rmse": rmse(pair.clean[epoch_id], output[epoch_id]),
                        "rrmse": rrmse(pair.clean[epoch_id], output[epoch_id]),
                    }
                )

    return pd.DataFrame(rows)


def summarize_external(per_pair: pd.DataFrame) -> pd.DataFrame:
    """Summarize external validation by condition and method."""
    rows = []
    grouped = per_pair.groupby(["artifact_type", "target_snr_db", "method"])

    for (artifact_type, target_snr_db, method), group in grouped:
        row = {
            "dataset": "PhysioNet Motor Imagery",
            "artifact_type": artifact_type,
            "target_snr_db": target_snr_db,
            "method": method,
            "n_pairs": int(group.shape[0]),
        }

        for metric in ["snr_db", "snr_gain_db", "rmse", "rrmse"]:
            summary = summarize_metric(group[metric].to_numpy())
            row[f"mean_{metric}"] = summary["mean"]
            row[f"std_{metric}"] = summary["std"]

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["artifact_type", "target_snr_db", "method"]
    )


def save_external_plot(summary: pd.DataFrame, output_path: Path) -> None:
    """Save a barplot of external validation SNR gain."""
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
    axis.set_title("Phase 3 external validation on PhysioNet artifact pairs")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--max-channels", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--epoch-length", type=int, default=512)
    parser.add_argument("--snr-levels", type=float, nargs="+", default=[-5.0, 0.0, 5.0])
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
