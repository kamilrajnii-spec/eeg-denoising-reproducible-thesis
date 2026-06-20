"""Controlled artifact mixing for Phase 1 clean/noisy EEG pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPSILON = 1e-12


@dataclass(frozen=True)
class ArtifactPair:
    """One clean/noisy pair created from real clean EEG and artifact epochs."""

    artifact_type: str
    target_snr_db: float
    clean: np.ndarray
    noisy: np.ndarray
    artifact: np.ndarray


def _as_2d_epochs(values: np.ndarray) -> np.ndarray:
    """Convert one or more epochs to shape: epochs x samples."""
    array = np.asarray(values, dtype=float)
    array = np.squeeze(array)

    if array.size == 0:
        raise ValueError("Input data must contain at least one sample.")
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim == 2:
        return array
    if array.ndim > 2:
        return array.reshape(array.shape[0], -1)

    raise ValueError("Input data must contain at least one sample.")


def align_artifact_to_clean(clean: np.ndarray, artifact: np.ndarray) -> np.ndarray:
    """Repeat or crop artifact data so it has the same shape as clean EEG."""
    clean_epochs = _as_2d_epochs(clean)
    artifact_epochs = _as_2d_epochs(artifact)

    n_clean_epochs, n_clean_samples = clean_epochs.shape
    epoch_indices = np.arange(n_clean_epochs) % artifact_epochs.shape[0]
    aligned = artifact_epochs[epoch_indices]

    if aligned.shape[1] == n_clean_samples:
        return aligned
    if aligned.shape[1] > n_clean_samples:
        return aligned[:, :n_clean_samples]

    repeats = int(np.ceil(n_clean_samples / aligned.shape[1]))
    return np.tile(aligned, (1, repeats))[:, :n_clean_samples]


def scale_artifact_to_snr(
    clean: np.ndarray,
    artifact: np.ndarray,
    target_snr_db: float,
) -> np.ndarray:
    """Scale artifact amplitude so clean + artifact has the requested SNR."""
    clean_epochs = _as_2d_epochs(clean)
    artifact_epochs = align_artifact_to_clean(clean_epochs, artifact)

    artifact_epochs = artifact_epochs - np.mean(
        artifact_epochs,
        axis=1,
        keepdims=True,
    )

    clean_power = np.mean(clean_epochs**2, axis=1, keepdims=True)
    artifact_power = np.mean(artifact_epochs**2, axis=1, keepdims=True)

    if np.any(artifact_power <= EPSILON):
        raise ValueError("Artifact power is too close to zero for SNR scaling.")

    target_artifact_power = clean_power / (10.0 ** (target_snr_db / 10.0))
    scale = np.sqrt(target_artifact_power / artifact_power)

    return artifact_epochs * scale


def combine_artifacts_equal_power(
    eog: np.ndarray,
    emg: np.ndarray,
) -> np.ndarray:
    """Combine ocular and myogenic artifacts so each contributes equal power.

    The raw EMG epochs in EEGdenoiseNet carry far more power than the EOG
    epochs (roughly four orders of magnitude). Summing them directly and then
    normalising the sum to a target SNR leaves the ocular component at a
    negligible fraction of the mixed signal. To create a genuinely mixed
    artifact that tests simultaneous ocular and myogenic removal, each
    component is first mean-removed and rescaled to unit RMS (equal power)
    before summation. The resulting mixed artifact is later scaled as a whole
    to hit the requested SNR, preserving the 50/50 power balance between the
    two artifact types.
    """
    eog_epochs = _as_2d_epochs(eog)
    emg_epochs = _as_2d_epochs(emg)

    eog_epochs = eog_epochs - np.mean(eog_epochs, axis=1, keepdims=True)
    emg_epochs = emg_epochs - np.mean(emg_epochs, axis=1, keepdims=True)

    eog_power = np.mean(eog_epochs**2, axis=1, keepdims=True)
    emg_power = np.mean(emg_epochs**2, axis=1, keepdims=True)

    if np.any(eog_power <= EPSILON) or np.any(emg_power <= EPSILON):
        raise ValueError("Artifact power is too close to zero for equal-power mixing.")

    eog_unit = eog_epochs / np.sqrt(eog_power)
    emg_unit = emg_epochs / np.sqrt(emg_power)

    return eog_unit + emg_unit


def mix_artifact(
    clean: np.ndarray,
    artifact: np.ndarray,
    target_snr_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return noisy EEG and the scaled artifact that was added."""
    clean_epochs = _as_2d_epochs(clean)
    scaled_artifact = scale_artifact_to_snr(clean_epochs, artifact, target_snr_db)
    noisy = clean_epochs + scaled_artifact

    return noisy, scaled_artifact


def create_artifact_pairs(
    clean_eeg: np.ndarray,
    eog: np.ndarray,
    emg: np.ndarray,
    snr_levels_db: list[float] | tuple[float, ...] = (-5.0, 0.0, 5.0),
) -> list[ArtifactPair]:
    """Create blink, muscle, and mixed artifact pairs for each SNR level."""
    clean_epochs = _as_2d_epochs(clean_eeg)
    eog_epochs = align_artifact_to_clean(clean_epochs, eog)
    emg_epochs = align_artifact_to_clean(clean_epochs, emg)
    mixed_artifacts = combine_artifacts_equal_power(eog_epochs, emg_epochs)

    pairs: list[ArtifactPair] = []
    artifact_sources = {
        "blink": eog_epochs,
        "muscle": emg_epochs,
        "mixed": mixed_artifacts,
    }

    for target_snr_db in snr_levels_db:
        for artifact_type, artifact in artifact_sources.items():
            noisy, scaled_artifact = mix_artifact(
                clean_epochs,
                artifact,
                float(target_snr_db),
            )
            pairs.append(
                ArtifactPair(
                    artifact_type=artifact_type,
                    target_snr_db=float(target_snr_db),
                    clean=clean_epochs.copy(),
                    noisy=noisy,
                    artifact=scaled_artifact,
                )
            )

    return pairs
