"""
Evaluator for BBKNN Algorithm Evolution.

This evaluator measures:
1. Integration quality (how well batches are mixed)
2. Biological variance preservation (how well structure is preserved)
3. Execution speed

BBKNN works by modifying the KNN graph rather than the embedding directly.
We evaluate by computing a spectral embedding from the corrected graph.
"""

import numpy as np
import time
import sys
import os
import importlib.util
from pathlib import Path
from sklearn.manifold import SpectralEmbedding
from typing import Dict, Any, Tuple, Optional


# =============================================================================
# Shared metrics loading via importlib
# =============================================================================

_metrics_module = None


def _load_shared_metrics():
    """
    Load shared metrics module via environment variable.

    Returns the module if successful, None otherwise.
    """
    global _metrics_module
    if _metrics_module is not None:
        return _metrics_module

    metrics_path = os.environ.get("METRICS_MODULE_PATH")
    if metrics_path and Path(metrics_path).exists():
        try:
            spec = importlib.util.spec_from_file_location("shared_metrics", metrics_path)
            metrics = importlib.util.module_from_spec(spec)
            sys.modules["shared_metrics"] = metrics
            spec.loader.exec_module(metrics)
            _metrics_module = metrics
            return metrics
        except Exception:
            pass
    return None


# =============================================================================
# Fallback implementations (used if shared metrics cannot be loaded)
# =============================================================================

def _get_data_dir_fallback() -> Path:
    """Fallback: Get the shared data directory path."""
    env_data_dir = os.environ.get("BBKNN_DATA_DIR")
    if env_data_dir:
        return Path(env_data_dir)

    absolute_path = Path("/Users/wzxu/Projects/Pantheon/pantheon-agents/examples/evolution_batch_correction/data")
    if absolute_path.exists():
        return absolute_path

    try:
        relative_path = Path(__file__).parent.parent / "data"
        if relative_path.exists():
            return relative_path
    except NameError:
        pass

    return absolute_path


def _load_tma_data_fallback(
    data_dir: Path,
    split: str = "train",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback: Load TMA data with real cell type labels."""
    import pandas as pd
    df = pd.read_csv(data_dir / f"tma_8000_{split}.csv")
    X = df.iloc[:, :30].values
    batch_labels = df["donor"].values
    celltype_labels = df["celltype"].values
    return X, batch_labels, celltype_labels


def _compute_batch_mixing_score_fallback(
    X: np.ndarray,
    batch_labels: np.ndarray,
    k: int = 50,
) -> float:
    """Fallback: Compute batch mixing score using k-nearest neighbors."""
    from sklearn.neighbors import NearestNeighbors

    n_cells = X.shape[0]
    unique_batches = np.unique(batch_labels)

    expected_props = np.array([
        np.sum(batch_labels == b) / n_cells
        for b in unique_batches
    ])

    nn = NearestNeighbors(n_neighbors=min(k + 1, n_cells), algorithm="auto")
    nn.fit(X)
    _, indices = nn.kneighbors(X)

    mixing_scores = []
    for i in range(n_cells):
        neighbor_batches = batch_labels[indices[i, 1:]]
        observed_props = np.array([
            np.sum(neighbor_batches == b) / k
            for b in unique_batches
        ])
        score = 1 - np.sqrt(np.mean((observed_props - expected_props) ** 2))
        mixing_scores.append(max(0, score))

    return np.mean(mixing_scores)


def _compute_bio_conservation_score_fallback(
    X_corrected: np.ndarray,
    X_original: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """Fallback: Compute biological structure conservation score."""
    from sklearn.metrics import silhouette_score
    try:
        if len(np.unique(true_labels)) > 1:
            silhouette = silhouette_score(X_corrected, true_labels)
            return (silhouette + 1) / 2
        else:
            return 0.5
    except Exception:
        return 0.5


def _get_default_fitness_weights_fallback() -> Dict[str, float]:
    """Fallback: Get default fitness weights."""
    return {
        "mixing_score": 0.45,
        "bio_conservation_score": 0.45,
        "speed_score": 0.10,
    }


def _error_result_fallback(message: str) -> Dict[str, Any]:
    """Fallback: Create standard error result."""
    return {
        "function_score": 0.0,
        "error": message,
    }


# =============================================================================
# Get functions (use shared metrics if available, fallback otherwise)
# =============================================================================

def _get_functions():
    """Get metric functions, preferring shared module if available."""
    metrics = _load_shared_metrics()

    if metrics:
        return {
            "load_tma_data": metrics.load_tma_data,
            "get_data_dir": lambda: metrics.get_data_dir("BBKNN_DATA_DIR"),
            "compute_batch_mixing_score": metrics.compute_batch_mixing_score,
            "compute_bio_conservation_score": metrics.compute_bio_conservation_score,
            "get_default_fitness_weights": lambda: metrics.get_default_fitness_weights(include_convergence=False),
            "error_result": metrics.error_result,
            "compute_standard_metrics": metrics.compute_standard_metrics,
        }
    else:
        return {
            "load_tma_data": _load_tma_data_fallback,
            "get_data_dir": _get_data_dir_fallback,
            "compute_batch_mixing_score": _compute_batch_mixing_score_fallback,
            "compute_bio_conservation_score": _compute_bio_conservation_score_fallback,
            "get_default_fitness_weights": _get_default_fitness_weights_fallback,
            "error_result": _error_result_fallback,
            "compute_standard_metrics": None,  # Use manual computation
        }


# =============================================================================
# BBKNN-specific functions
# =============================================================================

def graph_to_embedding(
    connectivities,
    n_components: int = 30,
) -> np.ndarray:
    """
    Convert a connectivity graph to an embedding using spectral embedding.

    Args:
        connectivities: Sparse connectivity matrix from BBKNN
        n_components: Number of dimensions for the embedding

    Returns:
        Embedding array (n_cells x n_components)
    """
    conn_sym = (connectivities + connectivities.T) / 2
    se = SpectralEmbedding(
        n_components=n_components,
        affinity='precomputed',
        random_state=42,
    )
    embedding = se.fit_transform(conn_sym.toarray())
    return embedding


def _load_bbknn_module(bbknn_path: Path):
    """Load the BBKNN matrix module from workspace."""
    matrix_path = bbknn_path / "matrix.py"
    spec_matrix = importlib.util.spec_from_file_location("bbknn.matrix", matrix_path)
    matrix_module = importlib.util.module_from_spec(spec_matrix)
    sys.modules["bbknn.matrix"] = matrix_module
    spec_matrix.loader.exec_module(matrix_module)
    return matrix_module


# =============================================================================
# Main evaluation function
# =============================================================================

def evaluate(workspace_path: str) -> Dict[str, Any]:
    """
    Evaluate the BBKNN implementation on TMA training data.

    Args:
        workspace_path: Path to the workspace containing bbknn package

    Returns:
        Dictionary with metrics and fitness_weights
    """
    funcs = _get_functions()
    workspace = Path(workspace_path)

    # Load the bbknn module from workspace
    bbknn_path = workspace / "bbknn"
    if not bbknn_path.exists():
        if (workspace / "matrix.py").exists():
            bbknn_path = workspace
        else:
            return funcs["error_result"]("bbknn package not found")

    try:
        matrix_module = _load_bbknn_module(bbknn_path)
    except Exception as e:
        return funcs["error_result"](f"Failed to load bbknn: {e}")

    # Load TMA training data
    data_dir = funcs["get_data_dir"]()

    try:
        X_train, batch_train, celltype_train = funcs["load_tma_data"](data_dir, split="train")
    except Exception as e:
        return funcs["error_result"](f"Failed to load TMA data: {e}")

    # Run bbknn and measure time
    try:
        start_time = time.time()

        distances, connectivities, params = matrix_module.bbknn(
            pca=X_train,
            batch_list=batch_train,
            neighbors_within_batch=3,
            n_pcs=30,
            computation='cKDTree',
            metric='euclidean',
        )

        execution_time = time.time() - start_time
        X_corrected = graph_to_embedding(connectivities, n_components=30)

    except Exception as e:
        return funcs["error_result"](f"BBKNN execution failed: {e}")

    # Compute metrics
    try:
        if funcs["compute_standard_metrics"]:
            # Use shared compute_standard_metrics
            return funcs["compute_standard_metrics"](
                X_corrected, X_train, batch_train, celltype_train, execution_time
            )
        else:
            # Manual computation using fallback functions
            mixing_score = funcs["compute_batch_mixing_score"](X_corrected, batch_train)
            bio_score = funcs["compute_bio_conservation_score"](X_corrected, X_train, celltype_train)
            speed_score = 1.0 / (1 + execution_time)

            return {
                "mixing_score": mixing_score,
                "bio_conservation_score": bio_score,
                "speed_score": speed_score,
                "execution_time": execution_time,
                "n_cells": len(X_train),
                "n_batches": len(np.unique(batch_train)),
                "fitness_weights": funcs["get_default_fitness_weights"](),
            }

    except Exception as e:
        return funcs["error_result"](f"Metric computation failed: {e}")


def _evaluate_on_split(workspace_path: str, split: str) -> Dict[str, Any]:
    """Evaluate on a specific data split."""
    funcs = _get_functions()
    workspace = Path(workspace_path)

    bbknn_path = workspace / "bbknn"
    if not bbknn_path.exists():
        if (workspace / "matrix.py").exists():
            bbknn_path = workspace
        else:
            return funcs["error_result"]("bbknn package not found")

    try:
        matrix_module = _load_bbknn_module(bbknn_path)
    except Exception as e:
        return funcs["error_result"](f"Failed to load bbknn: {e}")

    data_dir = funcs["get_data_dir"]()

    try:
        X, batch_labels, celltype_labels = funcs["load_tma_data"](data_dir, split=split)
    except Exception as e:
        return funcs["error_result"](f"Failed to load TMA {split} data: {e}")

    try:
        start_time = time.time()

        distances, connectivities, params = matrix_module.bbknn(
            pca=X,
            batch_list=batch_labels,
            neighbors_within_batch=3,
            n_pcs=30,
            computation='cKDTree',
            metric='euclidean',
        )

        execution_time = time.time() - start_time
        X_corrected = graph_to_embedding(connectivities, n_components=30)

    except Exception as e:
        return funcs["error_result"](f"BBKNN execution failed: {e}")

    try:
        if funcs["compute_standard_metrics"]:
            result = funcs["compute_standard_metrics"](
                X_corrected, X, batch_labels, celltype_labels, execution_time
            )
            result["dataset"] = f"tma_{split}"
            return result
        else:
            mixing_score = funcs["compute_batch_mixing_score"](X_corrected, batch_labels)
            bio_score = funcs["compute_bio_conservation_score"](X_corrected, X, celltype_labels)
            speed_score = 1.0 / (1 + execution_time)

            return {
                "mixing_score": mixing_score,
                "bio_conservation_score": bio_score,
                "speed_score": speed_score,
                "execution_time": execution_time,
                "n_cells": len(X),
                "n_batches": len(np.unique(batch_labels)),
                "dataset": f"tma_{split}",
                "fitness_weights": funcs["get_default_fitness_weights"](),
            }

    except Exception as e:
        return funcs["error_result"](f"Metric computation failed: {e}")


def evaluate_on_validation(workspace_path: str) -> Dict[str, Any]:
    """Evaluate on TMA VALIDATION set."""
    return _evaluate_on_split(workspace_path, "val")


def evaluate_on_test(workspace_path: str) -> Dict[str, Any]:
    """Evaluate on TMA TEST set."""
    return _evaluate_on_split(workspace_path, "test")


# Only run main block when executed directly (not via exec())
if __name__ == "__main__" and '__file__' in dir():
    workspace = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("BBKNN Evaluator Test (TMA Data)")
    print("=" * 60)
    print(f"Workspace: {workspace}")

    # Check if shared metrics loaded
    metrics = _load_shared_metrics()
    print(f"Shared metrics loaded: {metrics is not None}")

    print("\n" + "-" * 60)
    print("TRAINING SET Evaluation:")
    print("-" * 60)
    result = evaluate(workspace)
    for key, value in result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, int):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
