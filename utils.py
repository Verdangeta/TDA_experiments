"""Shared utilities for cross-persistence density experiments."""

from __future__ import annotations

import numpy as np
import ot
import tensorflow as tf
import torch
from scipy.stats import entropy
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


def gaussian_kernel(u: np.ndarray) -> np.ndarray:
    """Evaluate the standard two-dimensional Gaussian kernel row-wise."""
    return np.exp(-np.sum(u**2, axis=1) / 2) / (2 * np.pi)


# Backward-compatible aliases used by the original notebooks.
gaus_kernel = gaussian_kernel
kernel = gaussian_kernel


def sym_KL(y_true: np.ndarray, y_pred: np.ndarray) -> tf.Tensor:
    """Symmetric KL divergence used as a density-prediction loss."""
    loss = tf.keras.losses.KLDivergence()
    return (loss(y_true, y_pred) + loss(y_pred, y_true)) / 2


def K_H(h: float, kernel_fn, u: np.ndarray) -> np.ndarray:
    """Scale a two-dimensional kernel by bandwidth ``h``."""
    if h <= 0:
        raise ValueError("Bandwidth h must be positive")
    return kernel_fn(u / h) / (h**2)


def calculate_K_Hs(diagrams: list[np.ndarray], kernel_fn=gaussian_kernel, h: float = 0.1):
    """Precompute pairwise kernel values between all persistence diagrams."""
    if h <= 0:
        raise ValueError("Bandwidth h must be positive")

    n_diagrams = len(diagrams)
    result = np.empty((n_diagrams, n_diagrams), dtype=object)
    result_solo = np.empty((n_diagrams, n_diagrams), dtype=object)

    for i, source in enumerate(diagrams):
        source = np.asarray(source)
        for j, target in enumerate(diagrams):
            target = np.asarray(target)
            values = []
            values_solo = []
            for point in source:
                values.append(K_H(h, kernel_fn, point - target))
                values_solo.append(K_H(h, kernel_fn, point.reshape(1, -1)))
            result[i, j] = np.stack(values) if values else np.empty((0, len(target)))
            result_solo[i, j] = np.stack(values_solo) if values_solo else np.empty((0, 1))
    return result, result_solo


class DenseRagged(tf.keras.layers.Layer):
    """Dense layer applied to the flat values of a ragged tensor."""

    def __init__(self, units: int, use_bias: bool = True, activation: str = "linear", **kwargs):
        super().__init__(**kwargs)
        self._supports_ragged_inputs = True
        self.units = units
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        last_dim = input_shape[-1]
        self.kernel = self.add_weight("kernel", shape=[last_dim, self.units], trainable=True)
        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units], trainable=True)
        else:
            self.bias = None
        super().build(input_shape)

    def call(self, inputs):
        outputs = tf.ragged.map_flat_values(tf.matmul, inputs, self.kernel)
        if self.use_bias:
            outputs = tf.ragged.map_flat_values(tf.nn.bias_add, outputs, self.bias)
        return tf.ragged.map_flat_values(self.activation, outputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "use_bias": self.use_bias,
                "activation": tf.keras.activations.serialize(self.activation),
            }
        )
        return config


class PermopRagged(tf.keras.layers.Layer):
    """Permutation-invariant sum pooling for ragged point-cloud tensors."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._supports_ragged_inputs = True

    def call(self, inputs):
        return tf.math.reduce_sum(inputs, axis=1)


def pdist_gpu(a: np.ndarray, b: np.ndarray, device: str = "cuda:0", max_block_gb: float = 0.2) -> np.ndarray:
    """Compute a pairwise Euclidean distance matrix with torch on a GPU device."""
    a_tensor = torch.tensor(a, dtype=torch.float64)
    b_tensor = torch.tensor(b, dtype=torch.float64)

    approx_size_gb = (a_tensor.shape[0] + b_tensor.shape[0]) * a_tensor.shape[1] / 1e9
    parts = int(approx_size_gb / max_block_gb) + 1 if approx_size_gb > max_block_gb else 1

    distances = np.zeros((a_tensor.shape[0], b_tensor.shape[0]))
    a_device = a_tensor.to(device)
    for part in range(parts):
        start = int(part * b_tensor.shape[0] / parts)
        stop = min(int((part + 1) * b_tensor.shape[0] / parts), b_tensor.shape[0])
        b_device = b_tensor[start:stop].to(device)
        block = torch.cdist(a_device, b_device)
        distances[:, start:stop] = block.cpu()
        del b_device, block
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    del a_device
    return distances


def sep_dist(a: np.ndarray, b: np.ndarray, pdist_device: str = "cpu", n_jobs: int | None = None) -> np.ndarray:
    """Build the asymmetric cross-distance matrix used for Cross-RipsNet inputs."""
    if pdist_device == "cpu":
        jobs = -1 if n_jobs is None else n_jobs
        d1 = pairwise_distances(b, a, n_jobs=jobs)
        d2 = pairwise_distances(b, b, n_jobs=jobs)
    else:
        d1 = pdist_gpu(b, a, device=pdist_device)
        d2 = pdist_gpu(b, b, device=pdist_device)

    size = a.shape[0] + b.shape[0]
    distances = np.zeros((size, size))
    distances[a.shape[0] :, : a.shape[0]] = d1
    distances[a.shape[0] :, a.shape[0] :] = d2
    return distances


def _normalise_distribution(values: np.ndarray, epsilon: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).flatten()
    total = values.sum()
    if total <= 0:
        values = np.full_like(values, 1 / len(values), dtype=np.float64)
    else:
        values = values / total
    values = np.clip(values, epsilon, None)
    return values / values.sum()


def measure_dist(
    dist_initial,
    dist_predicted,
    resolution: int = 50,
    method: str = "Wasserstein",
    epsilon: float = 1e-8,
    verbose: bool = False,
) -> list[float]:
    """Compare batches of vectorized probability densities."""
    initial = np.asarray(dist_initial)
    predicted = np.asarray(dist_predicted)
    if initial.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: {initial.shape} != {predicted.shape}")
    if initial.ndim != 2:
        raise ValueError("Expected two-dimensional arrays shaped as (n_samples, n_bins)")

    if method == "Wasserstein":
        if resolution * resolution != initial.shape[1]:
            raise ValueError("resolution must match the flattened image size")
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        grid_x, grid_y = np.meshgrid(x, y)
        coordinates = np.stack((grid_x.flatten(), grid_y.flatten()), axis=1)
        cost = ot.dist(coordinates, coordinates)
        cost /= cost.max()

    distances: list[float] = []
    iterator = tqdm(range(len(predicted)), disable=not verbose)
    for idx in iterator:
        source = _normalise_distribution(initial[idx], epsilon)
        target = _normalise_distribution(predicted[idx], epsilon)

        if method == "Wasserstein":
            distances.append(float(ot.emd2(target, source, cost)))
        elif method == "KL":
            distances.append(float(entropy(source, target)))
        elif method == "KL_sym":
            distances.append(float((entropy(source, target) + entropy(target, source)) / 2))
        else:
            raise ValueError(f"Unknown method: {method}")

    if verbose:
        print(f"Mean {method} distance: {np.mean(distances):.6f}")
    return distances


def estimate_optimal_bandwidth(diagrams: list[np.ndarray], kernel_fn=gaussian_kernel, h_list=(0.1, 0.2, 0.3)):
    """Select a KDE bandwidth by a simple leave-one-diagram-out score."""
    scores = []
    for h in h_list:
        k_hs, _ = calculate_K_Hs(diagrams, kernel_fn, h=h)
        score = 0.0
        pairs = 0
        for i in range(len(diagrams)):
            for j in range(len(diagrams)):
                if i == j:
                    continue
                n_i = max(len(diagrams[i]), 1)
                n_j = max(len(diagrams[j]), 1)
                score -= 2 * np.sum(k_hs[i, j]) / (n_i * n_j)
                pairs += 1
        scores.append(score / max(pairs, 1))
    best_idx = int(np.argmin(scores))
    return h_list[best_idx], scores
