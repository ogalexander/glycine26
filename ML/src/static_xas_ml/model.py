"""PCA-score MLP model for full-spectrum virtual incident spectra."""

from __future__ import annotations

import os
import random

import numpy as np
from sklearn.decomposition import PCA

def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))


def fit_pca(
    y_train_spectra: np.ndarray,
    *,
    n_components: int = 20,
    seed: int = 42,
) -> tuple[PCA, np.ndarray]:
    """Fit a fixed-component PCA on reference training spectra and return train scores."""
    y_train = np.asarray(y_train_spectra, dtype=np.float32)
    max_n = min(int(n_components), y_train.shape[0], y_train.shape[1])
    if max_n < 1:
        raise ValueError("Need at least one sample and one pixel for PCA.")

    pca = PCA(n_components=max_n, svd_solver="full", random_state=int(seed))
    scores = pca.fit_transform(y_train)
    return pca, scores.astype(np.float32)


def build_mlp(
    *,
    input_dim: int,
    output_dim: int,
    hidden_layers: list[int],
    learning_rate: float,
    seed: int,
):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(int(seed))
    layers = [tf.keras.layers.Input(shape=(int(input_dim),))]
    for width in hidden_layers:
        layers.append(tf.keras.layers.Dense(int(width), activation="relu"))
    layers.append(tf.keras.layers.Dense(int(output_dim), activation="linear"))
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def train_mlp(
    model,
    x_train: np.ndarray,
    y_train_scores: np.ndarray,
    x_val: np.ndarray,
    y_val_scores: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    patience: int,
):
    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(patience),
            restore_best_weights=True,
        )
    ]
    return model.fit(
        x_train,
        y_train_scores,
        validation_data=(x_val, y_val_scores),
        epochs=int(epochs),
        batch_size=int(batch_size),
        callbacks=callbacks,
        verbose=2,
    )


def predict_spectra(
    model,
    scaler,
    pca: PCA,
    x: np.ndarray,
    *,
    clip_negative: bool = True,
) -> np.ndarray:
    """Predict full virtual-I0 spectra in the PCA basis."""
    x_scaled = scaler.transform(x)
    scores = model.predict(x_scaled, verbose=0)
    pred = pca.inverse_transform(scores)
    pred = np.asarray(pred, dtype=np.float32)
    if clip_negative:
        pred = np.clip(pred, 0.0, None)
    return pred.astype(np.float32)


def predict_shapes(*args, **kwargs) -> np.ndarray:
    """Backward-compatible alias for older notebooks."""
    kwargs.pop("eps", None)
    kwargs.pop("normalize_area", None)
    return predict_spectra(*args, **kwargs)
