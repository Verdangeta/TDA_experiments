"""Train Cross-RipsNet variants used in the paper experiments.

The script expects precomputed point clouds, point-cloud pair indexes, and target
cross-persistence density vectors. These large artifacts are intentionally not
stored in the repository; see README.md for the expected directory layout.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
from tqdm import tqdm

from utils import DenseRagged, PermopRagged, measure_dist, sep_dist, sym_KL


TASK_FILES = {
    "synthetic": {
        "clouds": "RipsNet_exp/cross_pd_circles_3000_strat_boost_10_data.npy",
        "indexes": "RipsNet_exp/cross_pd_circles_3000_strat_boost_10_indexes.npy",
        "targets": "RipsNet_exp/cross_pd_circles_3000_strat_boost_10_PI.npy",
        "mid": 2400,
        "end": None,
    },
    "3d_shapes": {
        "clouds": "Data/cross_ripsnet_3d_exp/3d_shapes_pc_train_2500_30_boostrap",
        "indexes": "Data/cross_ripsnet_3d_exp/3d_shapes_indexes_2500_30_boostrap",
        "targets": "Data/cross_ripsnet_3d_exp/3d_shapes_PI_2500_30_boostrap",
        "mid": 2200,
        "end": None,
    },
    "textual": {
        "clouds": "Data/cross_ripsnet_text_exp/human_gpt3_davinci_003_pc_train",
        "indexes": "Data/cross_ripsnet_text_exp/human_gpt3_davinci_003_train_indexes",
        "targets": "Data/cross_ripsnet_text_exp/human_gpt3_davinci_003_PI_10000_50_boostrap",
        "mid": 2000,
        "end": 2500,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_FILES, default="synthetic")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/cross_ripsnet"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dist-nfeatures", type=int, default=60)
    parser.add_argument("--use-pca", action="store_true", help="Reduce point-cloud coordinates before training")
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--pdist-device", default="cuda", help="Use 'cpu' or a torch device such as 'cuda:0'")
    parser.add_argument("--run-name", default="base")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_pickle_or_numpy(path: Path):
    with path.open("rb") as file:
        if path.suffix == ".npy":
            try:
                return np.load(file, allow_pickle=True)
            except Exception:
                file.seek(0)
                return pickle.load(file)
        return pickle.load(file)


def load_task_data(task: str, data_root: Path):
    config = TASK_FILES[task]
    clouds = load_pickle_or_numpy(data_root / config["clouds"])
    indexes = load_pickle_or_numpy(data_root / config["indexes"])
    targets = load_pickle_or_numpy(data_root / config["targets"])

    if task == "synthetic":
        indexes = np.vstack([indexes[:800], indexes[1000:1800], indexes[2000:2800], indexes[800:1000], indexes[1800:2000], indexes[2800:3000]])
        targets = np.vstack([targets[:800], targets[1000:1800], targets[2000:2800], targets[800:1000], targets[1800:2000], targets[2800:3000]])

    clouds = [np.asarray(cloud) for cloud in clouds]
    return clouds, np.asarray(indexes), np.asarray(targets), config["mid"], config["end"]


def create_model_ripsnet(cloud_dim: int, output_dim: int, patience: int):
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-8,
        patience=patience,
        restore_best_weights=True,
    )
    optimizer = tf.keras.optimizers.Adamax(learning_rate=1e-4)
    inputs_1 = tf.keras.Input(shape=(None, cloud_dim), dtype="float32", ragged=True)
    inputs_2 = tf.keras.Input(shape=(None, cloud_dim), dtype="float32", ragged=True)

    joined = keras.layers.Concatenate(axis=1)([inputs_1, inputs_2])
    joined = DenseRagged(units=30, activation="relu")(joined)
    joined = DenseRagged(units=20, activation="relu")(joined)
    joined = DenseRagged(units=10, activation="relu")(joined)
    joined = PermopRagged()(joined)

    hidden = tf.keras.layers.Dense(150, activation="relu", kernel_regularizer=l2(1e-4), activity_regularizer=l2(1e-4))(joined)
    hidden = tf.keras.layers.Dense(200, activation="relu", kernel_regularizer=l2(1e-4), activity_regularizer=l2(1e-4))(hidden)
    hidden = tf.keras.layers.Dense(400, activation="relu", kernel_regularizer=l2(1e-3), activity_regularizer=l2(1e-4))(hidden)
    outputs = density_head(hidden, output_dim)
    model = tf.keras.Model(inputs=[inputs_1, inputs_2], outputs=outputs)
    model.compile(optimizer=optimizer, loss=sym_KL)
    return model, callback


def create_model_base(cloud_dim: int, output_dim: int, patience: int):
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-8,
        patience=patience,
        restore_best_weights=True,
    )
    optimizer = tf.keras.optimizers.Adamax(learning_rate=1e-4)
    inputs_1 = tf.keras.Input(shape=(None, cloud_dim), dtype="float32", ragged=True)
    inputs_2 = tf.keras.Input(shape=(None, cloud_dim), dtype="float32", ragged=True)

    left = point_encoder(inputs_1)
    joined = point_encoder(keras.layers.Concatenate(axis=1)([inputs_1, inputs_2]))
    hidden = keras.layers.Concatenate(axis=-1)([left, joined])
    hidden = tf.keras.layers.Normalization()(hidden)
    hidden = dense_stack(hidden)
    outputs = density_head(hidden, output_dim)
    model = tf.keras.Model(inputs=[inputs_1, inputs_2], outputs=outputs)
    model.compile(optimizer=optimizer, loss=sym_KL)
    return model, callback


def create_model_with_distance_matrix(cloud_dim: int, dist_dim: int, output_dim: int, patience: int):
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-8,
        patience=patience,
        restore_best_weights=True,
    )
    optimizer = tf.keras.optimizers.Adamax(learning_rate=1e-4)
    inputs_1 = tf.keras.Input(shape=(None, cloud_dim), dtype="float32", ragged=True)
    inputs_2 = tf.keras.Input(shape=(None, cloud_dim), dtype="float32", ragged=True)
    inputs_dist = tf.keras.Input(shape=(None, dist_dim), dtype="float32", ragged=True)

    left = point_encoder(inputs_1)
    joined = point_encoder(keras.layers.Concatenate(axis=1)([inputs_1, inputs_2]))
    distances = point_encoder(inputs_dist)
    hidden = keras.layers.Concatenate(axis=-1)([left, joined, distances])
    hidden = tf.keras.layers.Normalization()(hidden)
    hidden = dense_stack(hidden)
    outputs = density_head(hidden, output_dim)
    model = tf.keras.Model(inputs=[inputs_1, inputs_2, inputs_dist], outputs=outputs)
    model.compile(optimizer=optimizer, loss=sym_KL)
    return model, callback


def point_encoder(inputs):
    encoded = DenseRagged(units=30, activation="relu")(inputs)
    encoded = DenseRagged(units=20, activation="relu")(encoded)
    encoded = DenseRagged(units=10, activation="relu")(encoded)
    return PermopRagged()(encoded)


def dense_stack(inputs, dropout_rate: float = 0.2):
    hidden = tf.keras.layers.Dense(150, activation="relu", kernel_regularizer=l2(1e-4), activity_regularizer=l2(1e-4))(inputs)
    hidden = tf.keras.layers.Dropout(dropout_rate)(hidden)
    hidden = tf.keras.layers.Dense(200, activation="relu", kernel_regularizer=l2(1e-4), activity_regularizer=l2(1e-4))(hidden)
    hidden = tf.keras.layers.Dropout(dropout_rate)(hidden)
    hidden = tf.keras.layers.Dense(400, activation="relu", kernel_regularizer=l2(1e-3), activity_regularizer=l2(1e-4))(hidden)
    return tf.keras.layers.Dropout(dropout_rate)(hidden)


def density_head(inputs, output_dim: int):
    outputs = tf.keras.layers.Dense(output_dim, activation="sigmoid")(inputs)
    outputs = tf.keras.layers.Lambda(lambda x: tf.experimental.numpy.clip(x, 1e-8, None))(outputs)
    return tf.keras.layers.Lambda(lambda x: x / (tf.reduce_sum(x, axis=-1, keepdims=True) + 1e-7))(outputs)


def build_ragged_pairs(features, indexes):
    left = tf.ragged.constant([features[i] for i in indexes[:, 0]], ragged_rank=1)
    right = tf.ragged.constant([features[i] for i in indexes[:, 1]], ragged_rank=1)
    return left, right


def compute_distance_features(point_clouds, indexes, n_features: int, pdist_device: str):
    matrices = []
    for idx in tqdm(indexes, desc="cross-distance matrices"):
        cloud_1 = point_clouds[idx[1]]
        cloud_2 = point_clouds[idx[0]]
        distances = sep_dist(cloud_1, cloud_2, pdist_device=pdist_device)
        mean_cross = distances[cloud_1.shape[0] :, : cloud_1.shape[0]].mean()
        distances[: cloud_1.shape[0], : cloud_1.shape[0]] = 0
        distances[distances < mean_cross * 1e-6] = 0
        matrices.append(distances[cloud_1.shape[0] :, :])
    return {
        "pca": tf.ragged.constant([PCA(n_components=n_features).fit_transform(matrix) for matrix in matrices], ragged_rank=1),
        "max": tf.ragged.constant([np.sort(matrix, axis=1)[:, -n_features:] for matrix in matrices], ragged_rank=1),
        "quantile": tf.ragged.constant(
            [np.percentile(matrix, np.linspace(0, 100, n_features), axis=-1, method="nearest").T for matrix in matrices],
            ragged_rank=1,
        ),
    }


def train_model(name, model, train_data, val_data, test_data, y_train, y_val, y_test, callback, epochs: int):
    start_time = time.time()
    history = model.fit(train_data, y_train, epochs=epochs, validation_data=(val_data, y_val), callbacks=[callback], verbose=0)
    train_prediction = model.predict(train_data)
    val_prediction = model.predict(val_data)
    test_prediction = model.predict(test_data)
    metrics = {
        "name": name,
        "train_kl_sym": float(np.mean(measure_dist(y_train, train_prediction, method="KL_sym"))),
        "val_kl_sym": float(np.mean(measure_dist(y_val, val_prediction, method="KL_sym"))),
        "test_kl_sym": float(np.mean(measure_dist(y_test, test_prediction, method="KL_sym"))),
        "fit_time_sec": float(time.time() - start_time),
        "epochs_ran": len(history.history.get("loss", [])),
    }
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    point_clouds, indexes, targets, mid, end = load_task_data(args.task, args.data_root)
    end = len(indexes) if end is None else end

    if args.use_pca:
        reducer = PCA(n_components=args.n_components)
        features = [reducer.fit_transform(cloud) for cloud in point_clouds]
        cloud_dim = args.n_components
    else:
        features = point_clouds
        cloud_dim = point_clouds[0].shape[-1]

    train_indexes = indexes[:mid]
    test_indexes = indexes[mid:end]
    y_all_train = np.vstack([targets[:mid]])
    y_test = np.vstack([targets[mid:end]])
    output_dim = y_all_train.shape[1]

    data_train_1, data_train_2 = build_ragged_pairs(features, train_indexes)
    data_test_1, data_test_2 = build_ragged_pairs(features, test_indexes)
    train_ids, val_ids = train_test_split(list(range(len(y_all_train))), test_size=0.1, random_state=args.seed)

    x_train_1 = tf.gather(data_train_1, train_ids, axis=0)
    x_train_2 = tf.gather(data_train_2, train_ids, axis=0)
    x_val_1 = tf.gather(data_train_1, val_ids, axis=0)
    x_val_2 = tf.gather(data_train_2, val_ids, axis=0)
    y_train = np.take(y_all_train, train_ids, axis=0)
    y_val = np.take(y_all_train, val_ids, axis=0)

    metrics = []
    distance_train = compute_distance_features(point_clouds, train_indexes, args.dist_nfeatures, args.pdist_device)
    distance_test = compute_distance_features(point_clouds, test_indexes, args.dist_nfeatures, args.pdist_device)

    for feature_name in ["pca", "max", "quantile"]:
        model, callback = create_model_with_distance_matrix(cloud_dim, args.dist_nfeatures, output_dim, args.patience)
        train_dist = tf.gather(distance_train[feature_name], train_ids, axis=0)
        val_dist = tf.gather(distance_train[feature_name], val_ids, axis=0)
        metrics.append(
            train_model(
                feature_name,
                model,
                [x_train_1, x_train_2, train_dist],
                [x_val_1, x_val_2, val_dist],
                [data_test_1, data_test_2, distance_test[feature_name]],
                y_train,
                y_val,
                y_test,
                callback,
                args.epochs,
            )
        )

    model, callback = create_model_base(cloud_dim, output_dim, args.patience)
    metrics.append(
        train_model(
            "cross_ripsnet_base",
            model,
            [x_train_1, x_train_2],
            [x_val_1, x_val_2],
            [data_test_1, data_test_2],
            y_train,
            y_val,
            y_test,
            callback,
            args.epochs,
        )
    )

    model, callback = create_model_ripsnet(cloud_dim, output_dim, args.patience)
    metrics.append(
        train_model(
            "ripsnet_baseline",
            model,
            [x_train_1, x_train_2],
            [x_val_1, x_val_2],
            [data_test_1, data_test_2],
            y_train,
            y_val,
            y_test,
            callback,
            args.epochs,
        )
    )

    output_path = args.output_dir / f"{args.run_name}_{args.task}_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
