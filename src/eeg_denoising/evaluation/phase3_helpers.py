"""Reusable helpers for Phase 3 evaluation and analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon


EPSILON = 1e-12


@dataclass(frozen=True)
class WilcoxonResult:
    """Container for one paired Wilcoxon signed-rank result."""

    statistic: float
    p_value: float
    effect_size_r: float
    n_pairs: int


def summarize_metric(values: np.ndarray) -> dict[str, float]:
    """Return mean, standard deviation, and count for one metric array."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty metric array.")

    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0

    return {
        "mean": float(np.mean(array)),
        "std": std,
        "n": int(array.size),
    }


def paired_wilcoxon_greater(
    treatment: np.ndarray,
    baseline: np.ndarray,
) -> WilcoxonResult:
    """Run a one-sided paired Wilcoxon test where treatment is expected higher."""
    treatment_array = np.asarray(treatment, dtype=float)
    baseline_array = np.asarray(baseline, dtype=float)

    if treatment_array.shape != baseline_array.shape:
        raise ValueError("treatment and baseline must have the same shape.")

    differences = treatment_array - baseline_array
    nonzero = differences[np.abs(differences) > EPSILON]

    if nonzero.size == 0:
        return WilcoxonResult(
            statistic=0.0,
            p_value=1.0,
            effect_size_r=0.0,
            n_pairs=0,
        )

    result = wilcoxon(
        treatment_array,
        baseline_array,
        alternative="greater",
        zero_method="wilcox",
    )
    statistic = float(result.statistic)
    n_pairs = int(nonzero.size)
    mean_rank_sum = n_pairs * (n_pairs + 1) / 4.0
    rank_sum_std = np.sqrt(n_pairs * (n_pairs + 1) * (2 * n_pairs + 1) / 24.0)

    if rank_sum_std <= EPSILON:
        effect_size = 0.0
    else:
        z_score = (statistic - mean_rank_sum) / rank_sum_std
        effect_size = float(z_score / np.sqrt(n_pairs))

    return WilcoxonResult(
        statistic=statistic,
        p_value=float(result.pvalue),
        effect_size_r=effect_size,
        n_pairs=n_pairs,
    )


def bonferroni_correct(p_values: list[float]) -> list[float]:
    """Apply Bonferroni correction to p-values."""
    n_tests = len(p_values)
    return [min(float(p_value) * n_tests, 1.0) for p_value in p_values]


def pearson_correlation(reference: np.ndarray, observed: np.ndarray) -> float:
    """Return Pearson correlation between two equal-length signals."""
    reference_array = np.asarray(reference, dtype=float).reshape(-1)
    observed_array = np.asarray(observed, dtype=float).reshape(-1)

    if reference_array.shape != observed_array.shape:
        raise ValueError("reference and observed must have the same shape.")

    if np.std(reference_array) <= EPSILON or np.std(observed_array) <= EPSILON:
        return 0.0

    return float(np.corrcoef(reference_array, observed_array)[0, 1])


def peak_amplitude_and_latency(signal: np.ndarray) -> tuple[float, int]:
    """Return absolute peak amplitude and sample index."""
    signal_array = np.asarray(signal, dtype=float).reshape(-1)
    if signal_array.size == 0:
        raise ValueError("signal cannot be empty.")

    index = int(np.argmax(np.abs(signal_array)))
    return float(signal_array[index]), index


def morphology_metrics(reference: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    """Compare waveform morphology using correlation and peak differences."""
    reference_peak, reference_index = peak_amplitude_and_latency(reference)
    observed_peak, observed_index = peak_amplitude_and_latency(observed)

    return {
        "pearson_correlation": pearson_correlation(reference, observed),
        "peak_amplitude_difference": float(observed_peak - reference_peak),
        "peak_latency_difference_samples": int(observed_index - reference_index),
    }

