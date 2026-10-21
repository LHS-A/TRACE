# TRACE

Official PyTorch implementation of TRACE, a dual-prototype heterogeneous graph framework for six-class corneal endothelial disease recognition and out-of-distribution rejection.

## Pipeline

1. A shared ProtoCEC-initialized ResNet-50 encoder extracts global descriptors and local tokens.
2. The evidence attention scorer weights local pathological evidence.
3. Per-class biomarker prototypes (`K_b=8`) are built with PAM k-medoids and matched to visual descriptors.
4. Per-class pathology prototypes (`K_p=10`) are learned with attention-weighted spherical k-means.
5. Disease, biomarker, and pathology nodes form a heterogeneous graph with disease-anchor, cross-evidence, and mutual-nearest-neighbour relations.
6. A relation-aware graph transformer performs classification; joint prototype coverage and a Tukey lower fence provide OOD rejection.

## Setup

```bash
pip install -r requirements.txt
```

Set `data.biomarker_csv` and `model.protocec_checkpoint` in `configs/trace.yaml`. The biomarker CSV must contain `path` (or `relative_path`), `patient_id`, `CD`, `AVG`, `SD`, `CV`, `MAX`, and `6A`. Images are arranged as:

```text
Dataset/{BK,CE,FECD,HC,ICE,PPCD}/*
```

## Run

```bash
# Validate inputs and create patient-disjoint five-fold splits.
python scripts/prepare_data.py --config configs/trace.yaml

# Train one fold through encoder, prototype-guided, and graph stages.
python scripts/run_experiment.py --config configs/trace.yaml --experiment fold1

# Run inference with the generated deployment checkpoint.
python scripts/infer.py \
  --checkpoint TRACE_reproduction/experiments/fold1/deployment.pt \
  --image path/to/image.png
```

This repository contains method, training, and inference code only. Datasets, checkpoints, evaluation utilities, comparison methods, and complexity profilers are intentionally excluded.
