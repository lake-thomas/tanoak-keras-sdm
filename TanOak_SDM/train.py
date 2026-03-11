"""
Training and evaluation functions for the NAIP + Climate SDM for Tan Oak.
"""

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
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
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def get_callbacks(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best_model.keras")
    csv_log_path = os.path.join(output_dir, "training_log.csv")

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
        tf.keras.callbacks.CSVLogger(csv_log_path, append=False),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def _save_loss_curves(history, output_dir):
    history_dict = history.history
    if "loss" not in history_dict:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(history_dict["loss"], label="Training loss", linewidth=2)

    if "val_loss" in history_dict:
        plt.plot(history_dict["val_loss"], label="Validation loss", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Binary crossentropy")
    plt.title("Training vs Validation Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=200)
    plt.close()


def _save_training_summary(history, output_dir):
    history_df = pd.DataFrame(history.history)
    summary_path = os.path.join(output_dir, "training_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Tan Oak SDM Training Summary\n")
        f.write("=" * 32 + "\n")
        f.write(f"generated_at_utc: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"epochs_run: {len(history_df)}\n")

        if not history_df.empty:
            if "val_loss" in history_df.columns:
                best_idx = int(history_df["val_loss"].idxmin())
                best_val_loss = float(history_df.loc[best_idx, "val_loss"])
                f.write(f"best_epoch_by_val_loss: {best_idx + 1}\n")
                f.write(f"best_val_loss: {best_val_loss:.6f}\n")

            if "loss" in history_df.columns:
                f.write(f"final_train_loss: {float(history_df['loss'].iloc[-1]):.6f}\n")

            if "val_loss" in history_df.columns:
                f.write(f"final_val_loss: {float(history_df['val_loss'].iloc[-1]):.6f}\n")

            if "accuracy" in history_df.columns:
                f.write(f"final_train_accuracy: {float(history_df['accuracy'].iloc[-1]):.6f}\n")

            if "val_accuracy" in history_df.columns:
                f.write(f"final_val_accuracy: {float(history_df['val_accuracy'].iloc[-1]):.6f}\n")


def fit_model(model, train_ds, val_ds, epochs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
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

    _save_loss_curves(history, output_dir)
    _save_training_summary(history, output_dir)

    return history


def _extract_y_true(test_ds):
    y_true = []
    for _, labels in test_ds:
        y_true.extend(labels.numpy().tolist())
    return np.asarray(y_true, dtype=np.float32)


def _save_confusion_matrix_plot(cm, output_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Absent", "Present"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Absent", "Present"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=200)
    plt.close(fig)


def evaluate_model(model, test_ds, output_dir, threshold=0.5):
    """
    Evaluate the model at a threshold of 0.5 on the test dataset and save
    metrics, confusion matrix, and predictions to the output directory.
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
    report_text = classification_report(y_true, y_pred)

    with open(os.path.join(output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    with open(os.path.join(output_dir, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    pd.DataFrame(cm).to_csv(os.path.join(output_dir, "confusion_matrix.csv"), index=False)
    _save_confusion_matrix_plot(cm, output_dir)

    preds_df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred": y_pred,
        }
    )
    preds_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

    with open(os.path.join(output_dir, "evaluation_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Tan Oak SDM Evaluation Summary\n")
        f.write("=" * 33 + "\n")
        f.write(f"generated_at_utc: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"threshold: {threshold}\n\n")
        f.write("Metrics\n")
        f.write("-" * 20 + "\n")
        for k, v in metrics_dict.items():
            f.write(f"{k}: {v:.6f}\n")
        f.write("\nClassification Report\n")
        f.write("-" * 20 + "\n")
        f.write(report_text)
        f.write("\nConfusion Matrix\n")
        f.write("-" * 20 + "\n")
        f.write(np.array2string(cm))

    print("\nTest metrics:")
    for k, v in metrics_dict.items():
        print(f"{k}: {v:.4f}")

    print("\nConfusion matrix:")
    print(cm)

    return metrics_dict, report, cm