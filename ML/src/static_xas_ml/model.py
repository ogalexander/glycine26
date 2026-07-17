"""PCA-score MLP model for virtual incident spectra."""

from __future__ import annotations

import os
import random

import numpy as np
from sklearn.decomposition import PCA

from .preprocessing import renormalize_shapes


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))


def fit_pca(
    y_train: np.ndarray,
    *,
    variance: float = 0.995,
    max_components: int = 20,
    seed: int = 42,
) -> tuple[PCA, np.ndarray]:
    """Fit PCA on reference training shapes and return train scores."""
    y_train = np.asarray(y_train, dtype=np.float32)
    max_n = min(int(max_components), y_train.shape[0], y_train.shape[1])
    if max_n < 1:
        raise ValueError("Need at least one sample and one pixel for PCA.")

    full = PCA(n_components=max_n, svd_solver="full", random_state=int(seed))
    full.fit(y_train)
    if float(variance) < 1.0:
        cumulative = np.cumsum(full.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumulative, float(variance)) + 1)
    else:
        n_components = max_n
    n_components = min(max(1, n_components), max_n)

    pca = PCA(n_components=n_components, svd_solver="full", random_state=int(seed))
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


def predict_shapes(model, scaler, pca: PCA, x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    x_scaled = scaler.transform(x)
    scores = model.predict(x_scaled, verbose=0)
    pred = pca.inverse_transform(scores)
    return renormalize_shapes(pred, eps=eps)
