import logging
from typing import List, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from .tree_structures import Node
from .utils import get_embeddings

logger = logging.getLogger(__name__)

RANDOM_SEED = 42


def _reduce_embeddings(embeddings: np.ndarray, dim: int) -> np.ndarray:
    if len(embeddings) <= 1:
        return embeddings

    target_dim = min(dim, len(embeddings) - 1, embeddings.shape[1])
    if target_dim < 1:
        return embeddings

    reducer = PCA(n_components=target_dim, random_state=RANDOM_SEED)
    return reducer.fit_transform(embeddings)


def _get_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 50) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    if max_clusters <= 1:
        return 1

    cluster_range = np.arange(1, max_clusters + 1)
    bics = []
    for cluster_count in cluster_range:
        model = GaussianMixture(
            n_components=cluster_count,
            random_state=RANDOM_SEED,
        )
        model.fit(embeddings)
        bics.append(model.bic(embeddings))

    return int(cluster_range[int(np.argmin(bics))])


def _gmm_cluster(
    embeddings: np.ndarray,
    threshold: float,
) -> tuple[list[np.ndarray], int]:
    cluster_count = _get_optimal_clusters(embeddings)
    model = GaussianMixture(n_components=cluster_count, random_state=RANDOM_SEED)
    model.fit(embeddings)
    probabilities = model.predict_proba(embeddings)
    labels = [np.where(probability > threshold)[0] for probability in probabilities]
    return labels, cluster_count


def perform_clustering(
    embeddings: np.ndarray,
    dim: int = 10,
    threshold: float = 0.1,
    verbose: bool = False,
) -> List[np.ndarray]:
    reduced = _reduce_embeddings(embeddings, dim)
    global_labels, global_cluster_count = _gmm_cluster(reduced, threshold)

    if verbose:
        logger.info("Global clusters: %s", global_cluster_count)

    local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for cluster_id in range(global_cluster_count):
        member_indices = [
            index
            for index, labels in enumerate(global_labels)
            if cluster_id in labels
        ]
        cluster_embeddings = embeddings[member_indices]

        if len(cluster_embeddings) == 0:
            continue

        if len(cluster_embeddings) <= dim + 1:
            sub_labels = [np.array([0]) for _ in cluster_embeddings]
            local_cluster_count = 1
        else:
            reduced_local = _reduce_embeddings(cluster_embeddings, dim)
            sub_labels, local_cluster_count = _gmm_cluster(reduced_local, threshold)

        if verbose:
            logger.info(
                "Local clusters in global cluster %s: %s",
                cluster_id,
                local_cluster_count,
            )

        for local_id in range(local_cluster_count):
            local_member_indices = [
                member_indices[index]
                for index, labels in enumerate(sub_labels)
                if local_id in labels
            ]
            for index in local_member_indices:
                local_clusters[index] = np.append(
                    local_clusters[index],
                    local_id + total_clusters,
                )

        total_clusters += local_cluster_count

    return local_clusters


class RAPTORClustering:
    """Cluster nodes using GMM over embeddings, following the RAPTOR algorithm."""

    @staticmethod
    def perform_clustering(
        nodes: List[Node],
        embedding_model_name: str,
        max_length_in_cluster: int = 3500,
        tokenizer=None,
        reduction_dimension: int = 10,
        threshold: float = 0.1,
        verbose: bool = False,
    ) -> List[List[Node]]:
        embeddings = np.array(get_embeddings(nodes, embedding_model_name))
        cluster_labels = perform_clustering(
            embeddings,
            dim=reduction_dimension,
            threshold=threshold,
            verbose=verbose,
        )

        node_clusters: List[List[Node]] = []
        unique_labels = np.unique(np.concatenate(cluster_labels))

        for label in unique_labels:
            indices = [
                index
                for index, labels in enumerate(cluster_labels)
                if label in labels
            ]
            cluster_nodes = [nodes[index] for index in indices]

            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            if tokenizer is not None:
                total_length = sum(
                    len(tokenizer.encode(node.text, add_special_tokens=False))
                    for node in cluster_nodes
                )
                if total_length > max_length_in_cluster:
                    if verbose:
                        logger.info(
                            "Reclustering cluster with %s nodes",
                            len(cluster_nodes),
                        )
                    node_clusters.extend(
                        RAPTORClustering.perform_clustering(
                            cluster_nodes,
                            embedding_model_name,
                            max_length_in_cluster=max_length_in_cluster,
                            tokenizer=tokenizer,
                            reduction_dimension=reduction_dimension,
                            threshold=threshold,
                            verbose=verbose,
                        )
                    )
                    continue

            node_clusters.append(cluster_nodes)

        return node_clusters
