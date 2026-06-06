# Data Folder

This folder is for public datasets used in Phase 1.

## EEGdenoiseNet

EEGdenoiseNet files here:

```text
data/eegdenoisenet/data
```

The loader searches for clean EEG, EOG, and EMG files. Common filenames are:

```text
EEG_all_epochs.npy
EOG_all_epochs.npy
EMG_all_epochs.npy
```

EEGdenoiseNet is used because it contains clean EEG epochs plus ocular and muscle artifact epochs.

## PhysioNet EEG Motor Movement/Imagery

PhysioNet EDF files here:

```text
data/physionet_mi/data
```

Nested folders are okay because the loader searches recursively for the first `.edf` file.

## No Synthetic Fallback

If data is missing, scripts stop with a clear message.

They do not generate fake or synthetic data.

