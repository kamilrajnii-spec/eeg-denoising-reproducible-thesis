from __future__ import annotations

import numpy as np
import pytest

from eeg_denoising.evaluation.phase3_helpers import (
    bonferroni_correct,
    morphology_metrics,
    paired_wilcoxon_greater,
    summarize_metric,
)


def test_summarize_metric_returns_mean_std_and_count() -> None:
    summary = summarize_metric(np.array([1.0, 2.0, 3.0]))

    assert summary["mean"] == 2.0
    assert round(summary["std"], 6) == 1.0
    assert summary["n"] == 3


def test_bonferroni_correction_clips_at_one() -> None:
    corrected = bonferroni_correct([0.01, 0.30, 0.80])

    assert corrected == pytest.approx([0.03, 0.90, 1.0])


def test_paired_wilcoxon_greater_detects_improvement() -> None:
    treatment = np.array([3.0, 4.0, 5.0, 6.0])
    baseline = np.array([1.0, 2.0, 3.0, 4.0])

    result = paired_wilcoxon_greater(treatment, baseline)

    assert result.n_pairs == 4
    assert result.p_value < 0.1
    assert result.effect_size_r > 0.0


def test_morphology_metrics_tracks_peak_and_correlation() -> None:
    reference = np.array([0.0, 1.0, 3.0, 1.0])
    observed = np.array([0.0, 1.0, 2.0, 4.0])

    metrics = morphology_metrics(reference, observed)

    assert metrics["peak_amplitude_difference"] == 1.0
    assert metrics["peak_latency_difference_samples"] == 1
    assert -1.0 <= metrics["pearson_correlation"] <= 1.0
