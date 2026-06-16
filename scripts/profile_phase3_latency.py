"""Profile Phase 3 latency on a larger held-out segment set."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eeg_denoising.models.dae import load_dae_checkpoint  # noqa: E402
from eeg_denoising.pipeline.hybrid_pipeline import apply_dae_to_epochs  # noqa: E402
from eeg_denoising.wavelet.dwt_denoising import denoise_epochs_dwt  # noqa: E402
from run_phase3_evaluation import build_test_split  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "results" / "phase3"
CHECKPOINT_PATH = PROJECT_ROOT / "results" / "phase2" / "dae_best_model.pt"
SUMMARY_PATH = OUTPUT_DIR / "latency_summary.csv"
BOXPLOT_PATH = OUTPUT_DIR / "latency_boxplot.png"


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        print("DAE checkpoint not found. Run scripts/train_phase2_dae.py first.")
        return 1

    test_split, _ = build_test_split(args)
    noisy_epochs = test_split["raw_noisy"]
    if noisy_epochs.shape[0] < args.segments:
        raise ValueError("Not enough held-out segments for requested latency profiling.")

    measured_epochs = noisy_epochs[: args.segments]
    model = load_dae_checkpoint(args.checkpoint, device=args.device)
    run_warmups(measured_epochs[0], model, args)
    measurements = measure_latency(measured_epochs, model, args.device)
    summary = summarize_latency(measurements, args)

    summary.to_csv(SUMMARY_PATH, index=False)
    save_latency_boxplot(measurements, BOXPLOT_PATH)

    print(f"Created {SUMMARY_PATH}")
    print(f"Created {BOXPLOT_PATH}")

    return 0


def run_warmups(epoch: np.ndarray, model, args: argparse.Namespace) -> None:
    """Run untimed warm-up passes for wavelet and DAE."""
    segment = epoch.reshape(1, -1)
    for _ in range(args.warmup_passes):
        wavelet = denoise_epochs_dwt(segment)
        _ = apply_dae_to_epochs(wavelet, model, device=args.device)


def measure_latency(
    epochs: np.ndarray,
    model,
    device: str,
) -> dict[str, list[float]]:
    """Measure wavelet, DAE-after-wavelet, and total hybrid latency."""
    measurements = {
        "wavelet_only": [],
        "dae_after_wavelet": [],
        "hybrid_total": [],
    }

    for epoch in epochs:
        segment = epoch.reshape(1, -1)

        start = time.perf_counter()
        wavelet = denoise_epochs_dwt(segment)
        measurements["wavelet_only"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        _ = apply_dae_to_epochs(wavelet, model, device=device)
        measurements["dae_after_wavelet"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        wavelet_for_hybrid = denoise_epochs_dwt(segment)
        _ = apply_dae_to_epochs(wavelet_for_hybrid, model, device=device)
        measurements["hybrid_total"].append((time.perf_counter() - start) * 1000.0)

    return measurements


def summarize_latency(
    measurements: dict[str, list[float]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Create latency summary rows with hardware information."""
    cpu = platform.processor() or platform.machine()
    rows = []

    for method, values in measurements.items():
        array = np.asarray(values, dtype=float)
        rows.append(
            {
                "method": method,
                "mean_ms": round(float(np.mean(array)), 6),
                "std_ms": round(float(np.std(array, ddof=1)), 6),
                "median_ms": round(float(np.median(array)), 6),
                "p95_ms": round(float(np.percentile(array, 95)), 6),
                "min_ms": round(float(np.min(array)), 6),
                "max_ms": round(float(np.max(array)), 6),
                "n_segments": int(array.size),
                "warmup_passes": args.warmup_passes,
                "device": args.device,
                "cpu": cpu,
            }
        )

    return pd.DataFrame(rows)


def save_latency_boxplot(
    measurements: dict[str, list[float]],
    output_path: Path,
) -> None:
    """Save a latency distribution boxplot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(measurements.keys())
    values = [measurements[label] for label in labels]

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(values, tick_labels=labels, showfliers=False)
    axis.set_ylabel("Milliseconds per 512-sample segment")
    axis.set_title("Phase 3 latency profile")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--segments", type=int, default=200)
    parser.add_argument("--warmup-passes", type=int, default=20)
    parser.add_argument("--max-clean-epochs", type=int, default=600)
    parser.add_argument("--snr-levels", type=float, nargs="+", default=[-5.0, 0.0, 5.0])
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
