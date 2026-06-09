"""Train the Phase 2 DAE using real EEGdenoiseNet artifact pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_denoising.data_loading.load_eegdenoisenet import load_eegdenoisenet  # noqa: E402
from eeg_denoising.models.dae import ConvDAE, write_model_summary  # noqa: E402
from eeg_denoising.preprocessing.artifact_mixing import (  # noqa: E402
    _as_2d_epochs,
    create_artifact_pairs,
)
from eeg_denoising.training.eeg_dataset import (  # noqa: E402
    EEGPairDataset,
    normalize_pairs,
    split_arrays,
)
from eeg_denoising.training.train_dae import TrainingConfig, train_dae_model  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "results" / "phase2"
CHECKPOINT_PATH = OUTPUT_DIR / "dae_best_model.pt"
LOSS_CSV_PATH = OUTPUT_DIR / "training_loss.csv"
LOSS_PLOT_PATH = OUTPUT_DIR / "training_loss_curve.png"
SUMMARY_PATH = OUTPUT_DIR / "model_summary.txt"


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    noisy, clean = build_training_pairs(args)
    noisy, clean, _, _ = normalize_pairs(noisy, clean)
    splits = split_arrays(noisy, clean, seed=args.seed)

    train_loader = DataLoader(
        EEGPairDataset(*splits["train"]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        EEGPairDataset(*splits["validation"]),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = ConvDAE()
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        spectral_weight=args.spectral_weight,
        device=args.device,
        checkpoint_path=CHECKPOINT_PATH,
    )
    history = train_dae_model(model, train_loader, validation_loader, config)

    pd.DataFrame(history).to_csv(LOSS_CSV_PATH, index=False)
    save_loss_plot(history)
    write_model_summary(
        model,
        SUMMARY_PATH,
        extra_lines=[
            f"Training pairs used: {len(noisy)}",
            f"Epochs requested: {args.epochs}",
            f"Batch size: {args.batch_size}",
            f"Best checkpoint: {CHECKPOINT_PATH}",
        ],
    )

    print(f"Created {CHECKPOINT_PATH}")
    print(f"Created {LOSS_CSV_PATH}")
    print(f"Created {LOSS_PLOT_PATH}")
    print(f"Created {SUMMARY_PATH}")

    return 0


def build_training_pairs(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """Create noisy/clean arrays from real EEGdenoiseNet data."""
    eegdenoisenet = load_eegdenoisenet(PROJECT_ROOT / "data" / "eegdenoisenet")

    clean = _as_2d_epochs(eegdenoisenet["clean_eeg"])[: args.max_clean_epochs]
    eog = _as_2d_epochs(eegdenoisenet["eog"])[: args.max_clean_epochs]
    emg = _as_2d_epochs(eegdenoisenet["emg"])[: args.max_clean_epochs]

    pairs = create_artifact_pairs(clean, eog, emg, snr_levels_db=args.snr_levels)

    noisy_arrays = []
    clean_arrays = []
    for pair in pairs:
        noisy_arrays.append(pair.noisy)
        clean_arrays.append(pair.clean)

    return np.vstack(noisy_arrays), np.vstack(clean_arrays)


def save_loss_plot(history: list[dict[str, float]]) -> None:
    frame = pd.DataFrame(history)

    plt.figure(figsize=(8, 4))
    plt.plot(frame["epoch"], frame["train_loss"], marker="o", label="train")
    plt.plot(frame["epoch"], frame["validation_loss"], marker="o", label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Phase 2 DAE training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=150)
    plt.close()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--spectral-weight", type=float, default=0.10)
    parser.add_argument("--max-clean-epochs", type=int, default=600)
    parser.add_argument("--snr-levels", type=float, nargs="+", default=[-5.0, 0.0, 5.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

