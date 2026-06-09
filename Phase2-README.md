# EEG Denoising Thesis Code

This repository contains the reproducible Phase 1 and Phase 2 code for the EEG
denoising thesis work.

Phase 1 verifies:

1. dataset loading,
2. controlled artifact-pair creation,
3. metric correctness,
4. reproducible Phase 1 outputs.

Phase 2 implements:

1. DWT wavelet denoising baseline,
2. honest ICA baseline checks for compatible multi-channel EEG,
3. a 1-D convolutional DAE,
4. the Wavelet to DAE hybrid pipeline,
5. measured inference-time profiling.

No final thesis results are claimed here. All numbers in `results/` are produced
by scripts and should be regenerated after any code or dataset change.

## Install

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

## Data

Use the real datasets already placed in:

```text
data/eegdenoisenet/
data/physionet_mi/
```

The code does not create synthetic fallback data when a dataset is missing.

## Phase 1 Commands

```bash
python scripts/verify_phase1_datasets.py
python scripts/create_phase1_artifact_pairs.py
```

## Phase 2 Commands

Run the DWT and ICA baseline table:

```bash
python scripts/run_phase2_wavelet_ica_baseline.py
```

Train the DAE:

```bash
python scripts/train_phase2_dae.py
```

Create the hybrid comparison plot:

```bash
python scripts/run_phase2_hybrid_demo.py
```

Measure per-segment inference time:

```bash
python scripts/profile_phase2_latency.py
```

## Phase 2 Outputs

```text
results/phase2/wavelet_ica_baseline_table.csv
results/phase2/training_loss.csv
results/phase2/training_loss_curve.png
results/phase2/model_summary.txt
results/phase2/dae_best_model.pt
results/phase2/inference_time.csv
results/phase2/hybrid_comparison_plot.png
```

## Tests

```bash
pytest -q
```


