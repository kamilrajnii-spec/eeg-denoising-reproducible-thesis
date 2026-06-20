"""Run Phase 3 paired statistical tests on full evaluation outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_denoising.evaluation.phase3_helpers import (  # noqa: E402
    bonferroni_correct,
    paired_wilcoxon_greater,
)


OUTPUT_DIR = PROJECT_ROOT / "results" / "phase3"
PER_PAIR_PATH = OUTPUT_DIR / "full_evaluation_per_pair.csv"
RESULTS_PATH = OUTPUT_DIR / "statistical_results.csv"
PLOT_PATH = OUTPUT_DIR / "statistical_plot.png"


def main() -> int:
    if not PER_PAIR_PATH.exists():
        print("Missing full_evaluation_per_pair.csv. Run scripts/run_phase3_evaluation.py first.")
        return 1

    per_pair = pd.read_csv(PER_PAIR_PATH)
    results = run_statistics(per_pair)
    results.to_csv(RESULTS_PATH, index=False)
    save_statistics_plot(results, PLOT_PATH)

    print(f"Created {RESULTS_PATH}")
    print(f"Created {PLOT_PATH}")

    return 0


def paired_cohens_d(treatment: np.ndarray, baseline: np.ndarray) -> float:
    """Cohen's d for paired samples: mean difference over its standard deviation.

    Uses the standard deviation of the paired differences (d_z), which is the
    appropriate paired-design effect size to accompany the Wilcoxon test and is
    reproducible from full_evaluation_per_pair.csv.
    """
    differences = np.asarray(treatment, dtype=float) - np.asarray(baseline, dtype=float)
    n = differences.size
    if n < 2:
        return float("nan")
    std = np.std(differences, ddof=1)
    if std == 0:
        return float("nan")
    return float(np.mean(differences) / std)


def run_statistics(per_pair: pd.DataFrame) -> pd.DataFrame:
    """Run Wilcoxon tests for hybrid SNR against noisy and wavelet baselines."""
    rows = []
    grouped = per_pair.groupby(["artifact_type", "target_snr_db"])

    for (artifact_type, target_snr_db), group in grouped:
        pivot = group.pivot(index="pair_id", columns="method", values="snr_db")
        comparisons = [
            ("hybrid_vs_noisy", "noisy_input"),
            ("hybrid_vs_wavelet", "wavelet_only"),
        ]

        for comparison_name, baseline_method in comparisons:
            hybrid_values = pivot["hybrid_wavelet_dae"].to_numpy()
            baseline_values = pivot[baseline_method].to_numpy()
            result = paired_wilcoxon_greater(
                hybrid_values,
                baseline_values,
            )
            rows.append(
                {
                    "artifact_type": artifact_type,
                    "target_snr_db": target_snr_db,
                    "metric": "snr_db",
                    "comparison": comparison_name,
                    "baseline_method": baseline_method,
                    "treatment_method": "hybrid_wavelet_dae",
                    "n_pairs": result.n_pairs,
                    "wilcoxon_statistic": result.statistic,
                    "p_value": result.p_value,
                    "effect_size_r": result.effect_size_r,
                    "cohens_d": paired_cohens_d(hybrid_values, baseline_values),
                }
            )

    frame = pd.DataFrame(rows)
    corrected = bonferroni_correct(frame["p_value"].tolist())
    frame["p_value_bonferroni"] = corrected
    frame["significant_bonferroni_0_05"] = frame["p_value_bonferroni"] < 0.05

    return frame


def save_statistics_plot(results: pd.DataFrame, output_path: Path) -> None:
    """Save an effect-size barplot for the statistical comparisons."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        f"{row.artifact_type}\n{row.target_snr_db:g} dB\n{row.comparison}"
        for row in results.itertuples(index=False)
    ]
    x = np.arange(len(labels))

    fig, axis = plt.subplots(figsize=(13, 5))
    axis.bar(x, results["effect_size_r"])
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_ylabel("Effect size r")
    axis.set_title("Phase 3 Wilcoxon effect sizes for hybrid SNR")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())

