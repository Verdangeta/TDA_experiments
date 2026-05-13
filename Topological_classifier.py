"""Topological feature generator for the time-series experiments."""

from __future__ import annotations

import numpy as np
import mtd
from gtda.time_series import SingleTakensEmbedding, TakensEmbedding
from scipy.stats import entropy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from tqdm import tqdm


def my_entropy(cross_barcodes, normalize: bool = False) -> np.ndarray:
    """Compute persistence entropy for H0 and H1 cross-barcodes."""
    pers_entropy = np.zeros((len(cross_barcodes), 2))
    sum_lifespan = np.zeros(len(cross_barcodes))

    for barcode_idx, barcode in enumerate(cross_barcodes):
        for hom_dim in [0, 1]:
            if len(barcode) <= hom_dim or len(barcode[hom_dim]) == 0:
                continue
            lifespan_sums = barcode[hom_dim][:, 1] - barcode[hom_dim][:, 0]
            sum_lifespan[barcode_idx] += np.sum(lifespan_sums)
            pers_entropy[barcode_idx, hom_dim] = entropy(lifespan_sums.astype(float), base=2)

    if normalize:
        denominators = np.log2(np.maximum(sum_lifespan, 1.0))
        denominators[denominators == 0] = 1.0
        pers_entropy /= denominators[:, None]

    return pers_entropy


class TopologicalFeatureGenerator(BaseEstimator, TransformerMixin):
    """Generate MTD and entropy features from time-delay embedded series."""

    def __init__(
        self,
        representatives,
        embedding_dimension: int = 50,
        embedding_time_delay: int = 4,
        search_opt_embd: bool = True,
        stride: int = 5,
        n_components: int = 3,
        random_state: int = 42,
        verbose: bool = True,
        pdist_device: str = "cuda",
    ):
        if len(representatives) == 0:
            raise ValueError("At least one representative time series is required")
        if representatives[0].shape[-1] <= (embedding_dimension - 1) * embedding_time_delay:
            raise ValueError("embedding_dimension and embedding_time_delay are too large for the series length")

        self.representatives = representatives
        self.search_opt_embd = search_opt_embd
        self.stride = stride
        self.random_state = random_state
        self.verbose = verbose
        self.pdist_device = pdist_device

        if search_opt_embd:
            search_embedder = SingleTakensEmbedding(
                parameters_type="search",
                n_jobs=-1,
                stride=stride,
                time_delay=embedding_time_delay,
                dimension=embedding_dimension,
            )
            search_embedder.fit(representatives[0].squeeze())
            self.embedding_dimension = search_embedder.dimension_
            self.embedding_time_delay = search_embedder.time_delay_
            if verbose:
                print("Optimal time delay based on mutual information:", search_embedder.time_delay_)
                print("Optimal embedding dimension based on false nearest neighbors:", search_embedder.dimension_)
        else:
            self.embedding_dimension = embedding_dimension
            self.embedding_time_delay = embedding_time_delay

        self.n_components = min(n_components, self.embedding_dimension)
        self.embedder = TakensEmbedding(
            time_delay=self.embedding_time_delay,
            dimension=self.embedding_dimension,
            stride=stride,
        )
        self.pca = PCA(n_components=self.n_components)
        self.persistence = mtd.calc_cross_barcodes
        self.entropy = my_entropy
        np.random.seed(self.random_state)

        transformed = np.asarray(self.embedder.fit_transform(self.representatives))
        self.representatives_transformed = [self.pca.fit_transform(cloud) for cloud in transformed]

        if verbose:
            print(f"The result shape of cloud: {self.representatives_transformed[0].shape}")

    def fit(self, X, y=None, batch_size_L: int = 6000, batch_size_R: int = 6000):
        return self

    def transform(self, X, batch_size_L: int = 6000, batch_size_R: int = 6000) -> np.ndarray:
        all_features = len(self.representatives) * 8 + 4
        ent_features = len(self.representatives) * 4

        embedded = np.asarray(self.embedder.fit_transform(X))
        features = [self.pca.fit_transform(cloud) for cloud in embedded]
        top_features = np.empty([len(features), all_features])

        entropy_offset = 0
        mtd_offset = ent_features
        total_steps = len(self.representatives_transformed) * 2 + 1
        with tqdm(total=total_steps, disable=not self.verbose) as pbar:
            for representative in self.representatives_transformed:
                entropy_offset, mtd_offset = self._append_pair_features(
                    top_features,
                    features,
                    representative,
                    entropy_offset,
                    mtd_offset,
                    batch_size_L,
                    batch_size_R,
                    reverse=False,
                )
                pbar.update(1)

                entropy_offset, mtd_offset = self._append_pair_features(
                    top_features,
                    features,
                    representative,
                    entropy_offset,
                    mtd_offset,
                    batch_size_L,
                    batch_size_R,
                    reverse=True,
                )
                pbar.update(1)

            barcodes = [
                self.persistence(
                    cloud,
                    cloud,
                    batch_size1=batch_size_L,
                    batch_size2=0,
                    pdist_device=self.pdist_device,
                    is_plot=False,
                )
                for cloud in features
            ]
            entropies = self.entropy(barcodes, normalize=False)
            top_features[:, entropy_offset : entropy_offset + 2] = entropies

            mtd_dim_1 = np.array([mtd.get_score(barc, 1, "sum_length") for barc in barcodes])
            mtd_dim_0 = np.array([mtd.get_score(barc, 0, "sum_length") for barc in barcodes])
            top_features[:, mtd_offset] = mtd_dim_1
            top_features[:, mtd_offset + 1] = mtd_dim_0
            pbar.update(1)

        top_features = np.nan_to_num(top_features, posinf=1000, neginf=-1000)
        top_features[top_features > 3e30] = 3e30
        top_features[top_features < -3e30] = -3e30
        return top_features

    def _append_pair_features(
        self,
        top_features: np.ndarray,
        clouds: list[np.ndarray],
        representative: np.ndarray,
        entropy_offset: int,
        mtd_offset: int,
        batch_size_L: int,
        batch_size_R: int,
        reverse: bool,
    ):
        barcodes = []
        for cloud in clouds:
            left, right = (cloud, representative) if reverse else (representative, cloud)
            barcodes.append(
                self.persistence(
                    left,
                    right,
                    batch_size1=batch_size_L,
                    batch_size2=batch_size_R,
                    pdist_device=self.pdist_device,
                    is_plot=False,
                )
            )

        entropies = self.entropy(barcodes, normalize=False)
        top_features[:, entropy_offset : entropy_offset + 2] = entropies
        top_features[:, mtd_offset] = np.array([mtd.get_score(barc, 1, "sum_length") for barc in barcodes])
        top_features[:, mtd_offset + 1] = np.array([mtd.get_score(barc, 0, "sum_length") for barc in barcodes])
        return entropy_offset + 2, mtd_offset + 2

    def inverse_transform(self):
        raise NotImplementedError("Topological features cannot be inverted back to time series")

    def get_feature_names_out(self, input_features=None):
        names = []
        for index_repr in range(len(self.representatives)):
            for suffix in ["", "rw_"]:
                names.append(f"entropies_{suffix}repr_{index_repr}_dim_0")
                names.append(f"entropies_{suffix}repr_{index_repr}_dim_1")

        for index_repr in range(len(self.representatives)):
            for suffix in ["", "rw_"]:
                names.append(f"mtd_{suffix}repr_{index_repr}_dim_1")
                names.append(f"mtd_{suffix}repr_{index_repr}_dim_0")

        return names + ["entropies_solo_0", "entropies_solo_1", "mtd_solo_1", "mtd_solo_0"]
