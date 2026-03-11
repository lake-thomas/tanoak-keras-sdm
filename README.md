# Tan Oak NAIP + Climate SDMs

**Authors:** Mark Feinberg, Thomas Lake, 2026

## Overview

This repository contains a Keras/TensorFlow workflow for fitting binary species distribution models that combine **high-resolution NAIP imagery** and **tabular environmental predictors** to estimate the **occurrence probability of Tan Oak** across **Curry County, Oregon**.

---

## Project Goal

The goal of this project is to build a deep learning workflow that predicts Tan Oak occurrence in Curry County, Oregon using:

* **NAIP imagery** sampled as 256 × 256 pixel image chips with 4 bands
* **environmental predictors** derived from normalized WorldClim variables
* **presence/background labels** for supervised binary classification

---

## Repository Structure

tan_oak_keras/
├── main.py
├── data.py
├── models.py
├── train.py
├── config.json
├── environment.yml          # optional, created after environment export
└── README.md

### File descriptions

#### `main.py`

Main training entry point.

* reads the JSON configuration file
* builds train/validation/test datasets
* initializes the requested model type
* compiles and fits the model
* evaluates the model on the test split
* saves trained model outputs to the experiment directory

#### `data.py`

Data loading utilities for Tan Oak imagery and environmental predictors.

* reads the sample CSV
* filters records by `train`, `val`, or `test` split
* loads 4-band TIFF image chips with `rasterio`
* extracts climate predictor variables from the CSV
* creates `tf.data.Dataset` pipelines for model training and evaluation

#### `models.py`

Keras model definitions.

* `image_climate`: combined CNN + climate MLP model
* `image_only`: CNN model using NAIP imagery only
* `climate_only`: MLP model using environmental variables only

#### `train.py`

Training and evaluation utilities.

* model compilation
* callbacks for checkpointing, early stopping, and learning-rate reduction
* model fitting
* test-set evaluation
* saving training history, metrics, confusion matrices, and predictions

#### `config.json`

Configuration file controlling the experiment.

* paths to the CSV and image directory
* output directory
* model type
* batch size, epochs, learning rate, and dropout
* image size
* environmental predictor columns to use

---

## Input Data Format

The workflow expects a CSV with one row per observation and columns such as:

* `chip_id`
* `label`
* `chip_path`
* `WC_01` to `WC_19`
* `block_id`
* `split`

Example:

```text
chip_id,label,chip_path,WC_01,...,WC_19,block_id,split
abs_134,0,abs_134.tif,0.7327,...,0.5771,82_942,val
```

### Required fields

* `chip_path`: image filename relative to the NAIP chip directory
* `label`: binary response variable (`0` = absence, `1` = presence)
* `split`: dataset split (`train`, `val`, or `test`)
* climate predictor columns listed in `config.json`

### Imagery assumptions

* images are stored as **4-band TIFF files**
* each image chip is **256 × 256 pixels**
* imagery is read as `RGB + NIR`
* values are scaled to `[0, 1]` during loading

---

## Installation

### 1. Create a conda environment

Check that conda is available:

```bash
conda --version
```

Create a new environment:

```bash
conda create -n tanoak-keras python=3.10 -y
```

Activate the environment:

```bash
conda activate tanoak-keras
```

### 2. Install core packages

Upgrade packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install the main dependencies:

```bash
pip install tensorflow pandas numpy scikit-learn rasterio matplotlib jupyter
pip install jupyterlab ipykernel
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name tanoak-keras --display-name "Python3.10 (tanoak-keras)"
```

---

## Test the environment

Check the TensorFlow installation:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

Check whether TensorFlow detects a GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## Save the environment

Export the environment for reproducibility:

```bash
conda env export --no-builds > environment.yml
```

---

## Running the demo model

From the project directory, run:

```bash
python main.py --config config.json
```

This will:

* load the configured train/validation/test splits
* train the requested model architecture
* evaluate performance on the test set
* save outputs to the configured experiment directory

---

## Notes

* Future extensions could include class weighting, stronger CNN backbones, map-based prediction outputs, and more detailed experiment tracking.

---
