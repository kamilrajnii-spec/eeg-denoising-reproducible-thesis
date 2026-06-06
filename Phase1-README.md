# EEG Denoising Thesis - Phase 1

This repository is the fresh reproducible Phase 1 codebase.

Phase 1 only verifies:

1. dataset loading,
2. artifact-pair creation,
3. metric correctness,
4. reproducible outputs.

No final denoising results are claimed in Phase 1.

## What Phase 1 Does

Phase 1 prepares the foundation for later EEG denoising experiments. It does not train a DAE model, does not compare DWT or ICA methods, and does not report final denoising performance.

The code verifies that:

- EEGdenoiseNet can be loaded from `data/eegdenoisenet/`
- PhysioNet EEG Motor Movement/Imagery EDF files can be loaded from `data/physionet_mi/`
- clean EEG can be mixed with EOG blink artifacts
- clean EEG can be mixed with EMG muscle artifacts
- clean EEG can be mixed with both EOG and EMG artifacts
- SNR, SNR gain, RMSE, and RRMSE are calculated by code

## Repository Structure

```text
data/
  README.md
  eegdenoisenet/
  physionet_mi/

src/
  eeg_denoising/
    data_loading/
      load_eegdenoisenet.py
      load_physionet_mi.py
    preprocessing/
      artifact_mixing.py
    evaluation/
      metrics.py

scripts/
  verify_phase1_datasets.py
  create_phase1_artifact_pairs.py

tests/
  test_metrics.py
  test_artifact_mixing.py

results/
  phase1/
```

## Setup

Create a Python environment, then install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Dataset Setup

Place EEGdenoiseNet files inside:

```text
data/eegdenoisenet/
```

Expected files usually include clean EEG, EOG, and EMG epoch files, for example:

```text
EEG_all_epochs.npy
EOG_all_epochs.npy
EMG_all_epochs.npy
```

Place PhysioNet EEG Motor Movement/Imagery EDF files inside:

```text
data/physionet_mi/
```

The scripts search recursively, so nested folders are okay.

Important: this repository does not generate synthetic fallback data. If a required dataset is missing, the script stops and prints a download/setup message.

## Verify Phase 1 Datasets

```bash
python3 scripts/verify_phase1_datasets.py
```

Expected output when both datasets are present:

```text
EEGdenoiseNet found: yes
Clean EEG shape: ...
EOG shape: ...
EMG shape: ...

PhysioNet found: yes
First EDF loaded: yes
Channels: ...
Sampling rate: ...
Duration: ...
```

## Create Phase 1 Artifact Pairs

```bash
python3 scripts/create_phase1_artifact_pairs.py
```

This creates:

```text
results/phase1/phase1_dataset_check.csv
results/phase1/phase1_artifact_pairs.csv
results/phase1/example_clean_noisy_plot.png
```

Artifact pairs are created at these SNR levels:

- -5 dB
- 0 dB
- +5 dB

Artifact types:

- blink: clean EEG + EOG
- muscle: clean EEG + EMG
- mixed: clean EEG + EOG + EMG

## Run Tests

```bash
python3 -m pytest
```

The tests check metric correctness and controlled artifact mixing behavior.

## Phase 1 Boundaries

This repository does not include:

- final DAE model training
- final DWT results
- final ICA results
- latency claims
- final SNR improvement tables
- clinical dataset claims
- synthetic dataset fallback

Those belong to later phases after the Phase 1 foundation is verified.
