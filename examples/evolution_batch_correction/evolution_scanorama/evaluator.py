"""
Evaluator for Scanorama Algorithm Evolution.

This evaluator measures:
1. Integration quality (how well batches are mixed)
2. Biological variance preservation (how well structure is preserved)
3. Execution speed

The combined score balances these metrics for evolution.
"""

import numpy as np
import time
import sys
import os
import importlib.util
from pathlib import Path
from typing import Dict, Any, Tuple, List


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
    env_data_dir = os.environ.get("SCANORAMA_DATA_DIR")
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
            "get_data_dir": lambda: metrics.get_data_dir("SCANORAMA_DATA_DIR"),
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
# Scanorama-specific functions
# =============================================================================

def split_by_batch(
    X: np.ndarray,
    batch_labels: np.ndarray,
) -> Tuple[List[np.ndarray], List[str], np.ndarray]:
    """
    Split data matrix by batch labels for scanorama input.

    Args:
        X: Data matrix (n_cells x n_features)
        batch_labels: Batch assignments

    Returns:
        datasets: List of data matrices, one per batch
        batch_names: List of batch names
        original_order: Indices to reconstruct original order
    """
    unique_batches = np.unique(batch_labels)
    datasets = []
    batch_names = []
    indices_list = []

    for batch in unique_batches:
        mask = batch_labels == batch
        indices = np.where(mask)[0]
        datasets.append(X[mask].copy())
        batch_names.append(str(batch))
        indices_list.append(indices)

    # Create mapping to reconstruct original order
    original_order = np.concatenate(indices_list)

    return datasets, batch_names, original_order


def reconstruct_from_batches(
    datasets: List[np.ndarray],
    original_order: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct full matrix from batch-split datasets.

    Args:
        datasets: List of corrected data matrices
        original_order: Indices to reconstruct original order

    Returns:
        Full corrected matrix in original cell order
    """
    # Concatenate all datasets
    X_concat = np.vstack(datasets)

    # Reconstruct original order
    inverse_order = np.argsort(original_order)
    X_corrected = X_concat[inverse_order]

    return X_corrected


def _load_scanorama_module(scanorama_path: Path):
    """Load the scanorama module from workspace."""
    # Import utils first
    utils_path = scanorama_path / "utils.py"
    spec_utils = importlib.util.spec_from_file_location("scanorama.utils", utils_path)
    utils_module = importlib.util.module_from_spec(spec_utils)
    sys.modules["scanorama.utils"] = utils_module
    spec_utils.loader.exec_module(utils_module)

    # Import main scanorama module
    main_path = scanorama_path / "scanorama.py"
    spec = importlib.util.spec_from_file_location("scanorama.scanorama", main_path)
    scanorama_module = importlib.util.module_from_spec(spec)
    sys.modules["scanorama.scanorama"] = scanorama_module
    spec.loader.exec_module(scanorama_module)

    return scanorama_module


# =============================================================================
# Main evaluation function
# =============================================================================

def evaluate(workspace_path: str) -> Dict[str, Any]:
    """
    Evaluate the Scanorama implementation on TMA training data.

    This is the main evaluation function called by Pantheon Evolution.
    Uses real cell type labels for accurate biological conservation scoring.

    Args:
        workspace_path: Path to the workspace containing scanorama package

    Returns:
        Dictionary with metrics and fitness_weights
    """
    funcs = _get_functions()
    workspace = Path(workspace_path)

    # Load the scanorama module from workspace
    # CodebaseSnapshot may place files directly in workspace root or in scanorama/ subdirectory
    scanorama_path = workspace / "scanorama"
    if not scanorama_path.exists():
        # Fallback: files might be directly in workspace root
        if (workspace / "scanorama.py").exists():
            scanorama_path = workspace
        else:
            return funcs["error_result"]("scanorama package not found")

    try:
        scanorama_module = _load_scanorama_module(scanorama_path)
    except Exception as e:
        return funcs["error_result"](f"Failed to load scanorama: {e}")

    # Load TMA training data with real cell type labels
    data_dir = funcs["get_data_dir"]()

    try:
        X_train, batch_train, celltype_train = funcs["load_tma_data"](data_dir, split="train")
    except Exception as e:
        return funcs["error_result"](f"Failed to load TMA data: {e}")

    # Split data by batch for scanorama input
    datasets, batch_names, original_order = split_by_batch(X_train, batch_train)

    # Create fake gene names (scanorama expects them, but we use PCA features)
    n_features = X_train.shape[1]
    genes_list = [[f"PC{i+1}" for i in range(n_features)] for _ in datasets]

    # Run scanorama and measure time
    try:
        start_time = time.time()

        # Use integrate function which returns low-dimensional embeddings
        # Use dimred=30 (same as input) to ensure dense arrays are returned
        # (numpy 2.x can't concatenate sparse matrices with np.concatenate)
        # Use approx=False to avoid annoy dependency (uses sklearn NearestNeighbors)
        datasets_corrected, _ = scanorama_module.integrate(
            datasets,
            genes_list,
            dimred=30,  # Same as input dims; ensures dense arrays are used
            verbose=0,
            knn=20,
            sigma=15,
            alpha=0.10,
            approx=False,  # Use exact NN (no annoy dependency)
            seed=42,
        )

        execution_time = time.time() - start_time

        # Reconstruct full matrix in original order
        X_corrected = reconstruct_from_batches(datasets_corrected, original_order)

    except Exception as e:
        return funcs["error_result"](f"Scanorama execution failed: {e}")

    # Correctness check: verify that the algorithm actually corrected the data
    correction_magnitude = np.abs(X_train - X_corrected).mean()
    if correction_magnitude < 0.01:
        result = funcs["error_result"]("No meaningful correction applied (data unchanged)")
        result["correction_magnitude"] = correction_magnitude
        return result

    # Compute metrics on training data using real cell type labels
    try:
        if funcs["compute_standard_metrics"]:
            # Use shared compute_standard_metrics
            result = funcs["compute_standard_metrics"](
                X_corrected, X_train, batch_train, celltype_train, execution_time
            )
            result["correction_magnitude"] = correction_magnitude
            return result
        else:
            # Manual computation using fallback functions
            mixing_score = funcs["compute_batch_mixing_score"](X_corrected, batch_train)
            bio_score = funcs["compute_bio_conservation_score"](X_corrected, X_train, celltype_train)
            speed_score = 1.0 / (1 + execution_time)

            return {
                "mixing_score": mixing_score,
                "bio_conservation_score": bio_score,
                "speed_score": speed_score,
                "correction_magnitude": correction_magnitude,
                "execution_time": execution_time,
                "n_cells": len(X_train),
                "n_batches": len(datasets),
                "fitness_weights": funcs["get_default_fitness_weights"](),
            }

    except Exception as e:
        return funcs["error_result"](f"Metric computation failed: {e}")


def _evaluate_on_split(workspace_path: str, split: str) -> Dict[str, Any]:
    """
    Evaluate the Scanorama implementation on a specific TMA data split.

    Args:
        workspace_path: Path to the workspace containing scanorama package
        split: Which split to evaluate on ("train", "val", or "test")

    Returns:
        Dictionary with metrics
    """
    funcs = _get_functions()
    workspace = Path(workspace_path)

    # Load the scanorama module from workspace
    # CodebaseSnapshot may place files directly in workspace root or in scanorama/ subdirectory
    scanorama_path = workspace / "scanorama"
    if not scanorama_path.exists():
        # Fallback: files might be directly in workspace root
        if (workspace / "scanorama.py").exists():
            scanorama_path = workspace
        else:
            return funcs["error_result"]("scanorama package not found")

    try:
        scanorama_module = _load_scanorama_module(scanorama_path)
    except Exception as e:
        return funcs["error_result"](f"Failed to load scanorama: {e}")

    # Load TMA data
    data_dir = funcs["get_data_dir"]()

    try:
        X, batch_labels, celltype_labels = funcs["load_tma_data"](data_dir, split=split)
    except Exception as e:
        return funcs["error_result"](f"Failed to load TMA {split} data: {e}")

    # Split data by batch
    datasets, batch_names, original_order = split_by_batch(X, batch_labels)
    n_features = X.shape[1]
    genes_list = [[f"PC{i+1}" for i in range(n_features)] for _ in datasets]

    # Run scanorama
    try:
        start_time = time.time()

        datasets_corrected, _ = scanorama_module.integrate(
            datasets,
            genes_list,
            dimred=30,  # Same as input dims; ensures dense arrays are used
            verbose=0,
            knn=20,
            sigma=15,
            alpha=0.10,
            approx=False,  # Use exact NN (no annoy dependency)
            seed=42,
        )

        execution_time = time.time() - start_time
        X_corrected = reconstruct_from_batches(datasets_corrected, original_order)

    except Exception as e:
        return funcs["error_result"](f"Scanorama execution failed: {e}")

    # Compute metrics
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
                "n_batches": len(datasets),
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
    print("Scanorama Evaluator Test (TMA Data)")
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
