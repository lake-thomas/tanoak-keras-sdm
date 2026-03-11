"""
Data class module to handle image and climate records for TanOak

"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

class TanOakDataModule:
    """Build tf.data datasets for train/val/test splits."""

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        env_features: Optional[List[str]] = None,
        image_size: int = 256,
    ):
        self.csv_path = csv_path # Path to .CSV file with presence/ background records
        self.image_dir = image_dir # Path to image directory with NAIP tiles
        self.image_size = image_size # NAIP images sampled at 256 x 256

        self.df = pd.read_csv(csv_path)

        default_env_features = [f"WC_{i:02d}" for i in range(1, 20)] # Worldclim variables: WC_01-WC_19
        self.env_features = env_features or default_env_features

        # Relative path to NAIP chip, label (int: 0/1 for pres/bkg), and split (string: train, val, test)
        required_cols = ["chip_path", "label", "split"] + self.env_features
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

    def _load_example(self, chip_path, env, label):
        """Load a single 4-band TIFF chip and climate vector."""
        chip_path = chip_path.numpy().decode("utf-8")
        full_path = os.path.join(self.image_dir, chip_path) # Get full file path for NAIP image

        with rasterio.open(full_path) as src:
            img = src.read().astype(np.float32)  # Read NAIP image as: (Channels, Height, Width) - expect (4, 256, 256)

        # Convert (C, H, W) -> (H, W, C)
        img = np.transpose(img, (1, 2, 0))

        # Scale byte imagery if needed
        if img.max() > 1.0:
            img = img / 255.0

        # Safety checks (is the imagery 4-band?)
        if img.shape[-1] != 4:
            raise ValueError(f"Expected 4-band NAIP image, got shape {img.shape} for {full_path}")

        img = np.clip(img, 0.0, 1.0).astype(np.float32) # Convert image data to float32
        env = np.asarray(env, dtype=np.float32) # Convert env data to float32
        env = np.clip(env, -10.0, 10.0) # Ensure env data is normalized
        label = np.float32(label)

        return img, env, label

    def _tf_load_example(self, chip_path, env, label):
        image, env, label = tf.py_function(
            func=self._load_example,
            inp=[chip_path, env, label],
            Tout=[tf.float32, tf.float32, tf.float32],
        )

        image.set_shape((self.image_size, self.image_size, 4))
        env.set_shape((len(self.env_features),))
        label.set_shape(())

        return {"image": image, "env": env}, label

    def _build_dataset(self, split: str, batch_size: int, training: bool = False):
        df_split = self.df[self.df["split"] == split].reset_index(drop=True) # Split as: train, val, test

        chip_paths = df_split["chip_path"].astype(str).values
        env = df_split[self.env_features].astype(np.float32).values
        labels = df_split["label"].astype(np.float32).values

        ds = tf.data.Dataset.from_tensor_slices((chip_paths, env, labels))

        if training:
            ds = ds.shuffle(buffer_size=len(df_split), reshuffle_each_iteration=True)

        ds = ds.map(self._tf_load_example, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds

    def get_datasets(self, batch_size: int):
        train_ds = self._build_dataset("train", batch_size, training=True)
        val_ds = self._build_dataset("val", batch_size, training=False)
        test_ds = self._build_dataset("test", batch_size, training=False)
        return train_ds, val_ds, test_ds