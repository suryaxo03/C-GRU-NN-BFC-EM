# Hybrid CNN-GRU Network for Bearing Fault Classification & Edge Deployment

A deep learning pipeline that classifies rotating machinery bearing faults from raw vibration signals, with model compression for deployment on resource-constrained embedded hardware. Built as a final year engineering project.

## Overview

Bearing failures are one of the most common causes of unplanned downtime in rotating machinery. This project builds an end-to-end pipeline that takes raw accelerometer vibration signals and classifies them into one of four conditions *viz.,* **Normal**, **Outer Race fault**, **Inner Race fault**, or **Ball fault** using a hybrid CNN-GRU neural network. The model is then compressed through quantisation and weight approximation to make it viable for deployment on embedded/edge devices, where memory and compute are limited.

## Dataset

- **Source:** [CWRU (Case Western Reserve University) Bearing Dataset](https://engineering.case.edu/bearingdatacenter) which is a widely used public benchmark for bearing fault diagnosis
- **Signal type:** Drive-End (DE) time-domain vibration signals, provided as `.mat` files
- **Classes:** Normal, Outer Race (OR), Inner Race (IR), Ball (BR)
- **Class imbalance:** The raw dataset was heavily skewed (77 OR samples vs. just 4 Normal samples). This was corrected via data augmentation: noise injection, time shifting, amplitude scaling, and time stretching to balance all classes to ~77-78 samples each.

## Methodology

1. **Signal preprocessing** - Raw `.mat` vibration signals are loaded and augmented to resolve class imbalance.
2. **Feature extraction** - Signals are converted to MFCCs (Mel-Frequency Cepstral Coefficients), a standard technique for capturing frequency-domain characteristics of time-series signals. Each sample is reduced to a fixed shape of (39 coefficients × 1280 time steps).
3. **Model architecture** - A hybrid CNN-GRU network:
   - 3× Conv2D blocks (32 → 64 → 128 filters) with max-pooling, for local spectral feature extraction
   - Global Average Pooling to reduce dimensionality
   - 2× stacked GRU layers (64 units each) to capture temporal dependencies
   - A Conv1D refinement layer
   - Dense(128) with dropout, followed by a softmax output over 4 classes
4. **Training** - Adam optimiser (lr=3e-4, gradient clipping), trained for 100 epochs with sparse categorical crossentropy.
5. **Model compression for embedded deployment** - Since the full-precision model is too large for microcontroller-class hardware, two compression strategies were evaluated:
   - **Power-of-2 weight approximation** - reduces the model to ~176K parameters (~3x smaller)
   - **Post-training quantisation (TFLite, FP16)** - converts weights to half-precision for edge inference

## Results

| Model variant | Parameters | Size | Test Accuracy |
|---|---|---|---|
| Full-precision CNN-GRU | 528,206 | 2.01 MB | 100% |
| Quantised (deployed/embedded) | - | 371 KB | 97% |
| TFLite (FP16 quantisation) | 176,070 | 687.78 KB | 98.4% |

The full-precision model achieves perfect classification on the held-out test set. The compressed, deployment-ready variant trades a small amount of accuracy for a **~5.4x reduction in model size**, making it suitable for real-time inference on embedded hardware without a meaningful drop in diagnostic reliability.

Evaluation was carried out using confusion matrices and ROC/AUC curves across all four fault classes to confirm the model wasn't just accurate on average but consistently reliable per class.

## Tech Stack

`Python` · `TensorFlow / Keras` · `TFLite` · `Librosa` (MFCC extraction) · `NumPy` · `Scikit-learn` · `Matplotlib` / `Seaborn`

## Repository Structure

```
├── Quantized_Hybrid_C-GRU_NN.ipynb   # Full pipeline: preprocessing, training, evaluation, compression
└── README.md
```

## Key Takeaways

- Designed a full signal-to-prediction pipeline from raw vibration data, including handling real-world class imbalance.
- Applied MFCC-based feature extraction, more commonly seen in audio processing, to a vibration-based fault diagnosis problem.
- Benchmarked multiple model compression strategies (weight approximation, FP16 quantisation) to meet embedded deployment constraints, rather than optimising for accuracy alone.
