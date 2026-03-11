"""
Main script to run the NAIP + Climate SDM for Tan Oak.

Usage:

python main.py --config config.json

Authors: Thomas Lake, Mark Feinberg, 2026
"""

import argparse
import json
import os

import tensorflow as tf

from data import TanOakDataModule
from models import build_model
from train import compile_model, fit_model, evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Train Tan Oak NAIP + Climate classifier in Keras")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    experiment_dir = os.path.join(config["output_dir"], config["experiment"])
    os.makedirs(experiment_dir, exist_ok=True)

    print("TensorFlow version:", tf.__version__)
    print("Num GPUs available:", len(tf.config.list_physical_devices("GPU"))) # This should show [] if no GPU is available

    # Alternatively - we might train on the CPU - or use HPC GPU resources to train?

    # Load training, validation, and testing data using the helper class
    data = TanOakDataModule(
        csv_path=config["csv_path"],
        image_dir=config["image_dir"],
        env_features=config.get("env_features"),
        image_size=config.get("image_size", 256),
    )

    train_ds, val_ds, test_ds = data.get_datasets(batch_size=config.get("batch_size", 16)) # Default to 16 samples per training step

    model_type = config.get("model_type", "image_climate") # Specify model type in config or default to image_climate
    
    model = build_model(
        model_type=model_type,
        num_env_features=len(data.env_features),
        image_size=config.get("image_size", 256),
        dropout=config.get("dropout", 0.25),
    )

    # Compile using the appropriate input signature
    model = compile_model(model, learning_rate=config.get("learning_rate", 1e-4))

    print(model.summary())

    # For image_only or climate_only models, convert dataset structure
    if model_type == "image_only":
        train_ds_use = train_ds.map(lambda x, y: (x["image"], y))
        val_ds_use = val_ds.map(lambda x, y: (x["image"], y))
        test_ds_use = test_ds.map(lambda x, y: (x["image"], y))
    elif model_type == "climate_only":
        train_ds_use = train_ds.map(lambda x, y: (x["env"], y))
        val_ds_use = val_ds.map(lambda x, y: (x["env"], y))
        test_ds_use = test_ds.map(lambda x, y: (x["env"], y))
    else:
        train_ds_use, val_ds_use, test_ds_use = train_ds, val_ds, test_ds

    fit_model(
        model=model,
        train_ds=train_ds_use,
        val_ds=val_ds_use,
        epochs=config.get("epochs", 10),
        output_dir=experiment_dir,
    )

    evaluate_model(model, test_ds_use, experiment_dir)

    model.save(os.path.join(experiment_dir, "final_model.keras"))
    print(f"Saved outputs to: {experiment_dir}")


if __name__ == "__main__":
    main()