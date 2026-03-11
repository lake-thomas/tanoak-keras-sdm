"""
Training and evaluation functions for the NAIP + Climate SDM for Tan Oak.

"""

import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def compile_model(model, learning_rate=1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ],
    )
    return model


def get_callbacks(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best_model.keras")

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=5,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
    ]


def fit_model(model, train_ds, val_ds, epochs, output_dir):
    callbacks = get_callbacks(output_dir)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)
    return history


def _extract_y_true(test_ds):
    y_true = []
    for _, labels in test_ds:
        y_true.extend(labels.numpy().tolist())
    return np.asarray(y_true, dtype=np.float32)


def evaluate_model(model, test_ds, output_dir, threshold=0.5):
    """
    Evaluate the model at a threshold of 0.5 on the test dataset and save metrics, confusion matrix, and predictions to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    results = model.evaluate(test_ds, verbose=0)
    metric_names = model.metrics_names
    metrics_dict = {k: float(v) for k, v in zip(metric_names, results)}

    y_true = _extract_y_true(test_ds)
    y_prob = model.predict(test_ds, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    metrics_dict["roc_auc_sklearn"] = float(roc_auc_score(y_true, y_prob))
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)

    with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    pd.DataFrame(cm).to_csv(os.path.join(output_dir, "confusion_matrix.csv"), index=False)

    preds_df = pd.DataFrame({
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
    })

    preds_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

    print("\nTest metrics:")
    for k, v in metrics_dict.items():
        print(f"{k}: {v:.4f}")

    print("\nConfusion matrix:")
    print(cm)

    return metrics_dict, report, cm