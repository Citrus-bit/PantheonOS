"""
Shared evaluation metrics for batch correction algorithm evolution.

This module provides common evaluation functions used by Harmony, BBKNN, and Scanorama
evaluators for measuring batch correction quality:

1. Data loading utilities for TMA datasets
2. Batch mixing score (kNN-based)
3. Biological conservation score (silhouette-based)
4. Standard fitness weights and metric computation
5. Utility functions for evaluator implementation

Usage:
    Evaluators can load this module via importlib using the METRICS_MODULE_PATH
    environment variable set by run_evolution.py.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from typing import Dict, Any, Tuple, Optional


def load_tma_data(
    data_dir: Path,
    split: str = "train",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load TMA data with real cell type labels.

    Args:
        data_dir: Directory containing the data files
        split: Which split to load ("train", "val", or "test")

    Returns:
        X: Features (n_cells x 30)
        batch_labels: Batch labels (donor)
        celltype_labels: True cell type labels
    """
    df = pd.read_csv(data_dir / f"tma_8000_{split}.csv")
    X = df.iloc[:, :30].values  # PC1-PC30
    batch_labels = df["donor"].values
    celltype_labels = df["celltype"].values

    return X, batch_labels, celltype_labels


def compute_batch_mixing_score(
    X: np.ndarray,
    batch_labels: np.ndarray,
    k: int = 50,
) -> float:
    """
    Compute batch mixing score using k-nearest neighbors.

    Measures how well different batches are mixed in the embedding.
    Higher score = better mixing.

    Args:
        X: Embedding (n_cells x n_features)
        batch_labels: Batch assignments
        k: Number of neighbors

    Returns:
        Mixing score in [0, 1]
    """
    n_cells = X.shape[0]
    unique_batches = np.unique(batch_labels)

    # Expected proportion of each batch
    expected_props = np.array([
        np.sum(batch_labels == b) / n_cells
        for b in unique_batches
    ])

    # Find k nearest neighbors
    nn = NearestNeighbors(n_neighbors=min(k + 1, n_cells), algorithm="auto")
    nn.fit(X)
    _, indices = nn.kneighbors(X)

    # For each cell, compute batch proportions in neighborhood
    mixing_scores = []
    for i in range(n_cells):
        neighbor_batches = batch_labels[indices[i, 1:]]  # Exclude self
        observed_props = np.array([
            np.sum(neighbor_batches == b) / k
            for b in unique_batches
        ])

        # Compare to expected (lower deviation = better mixing)
        score = 1 - np.sqrt(np.mean((observed_props - expected_props) ** 2))
        mixing_scores.append(max(0, score))

    return np.mean(mixing_scores)


def compute_bio_conservation_score(
    X_corrected: np.ndarray,
    X_original: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """
    Compute biological structure conservation score using silhouette score.

    Measures how well the biological clusters are separated after correction.
    Higher silhouette = better separation of cell types.

    Args:
        X_corrected: Corrected embedding
        X_original: Original embedding (unused, kept for API compatibility)
        true_labels: True biological labels (cell types)

    Returns:
        Silhouette score normalized to [0, 1]
    """
    try:
        if len(np.unique(true_labels)) > 1:
            # Compute silhouette score on corrected data
            silhouette = silhouette_score(X_corrected, true_labels)
            # Normalize from [-1, 1] to [0, 1]
            return (silhouette + 1) / 2
        else:
            return 0.5
    except Exception:
        return 0.5


def get_default_fitness_weights(include_convergence: bool = False) -> Dict[str, float]:
    """
    Get default fitness weights for batch correction evaluation.

    Args:
        include_convergence: If True, include convergence_score (for Harmony)

    Returns:
        Dictionary with metric weights
    """
    if include_convergence:
        return {
            "mixing_score": 0.45,
            "bio_conservation_score": 0.45,
            "speed_score": 0.05,
            "convergence_score": 0.05,
        }
    else:
        return {
            "mixing_score": 0.45,
            "bio_conservation_score": 0.45,
            "speed_score": 0.10,
        }


def get_data_dir(env_var: str = "DATA_DIR") -> Path:
    """
    Get the shared data directory path.

    Tries in order:
    1. Environment variable specified by env_var
    2. Hardcoded absolute path (for exec() context)
    3. Relative path from __file__ (for direct execution)

    Args:
        env_var: Name of environment variable containing data path

    Returns:
        Path to data directory
    """
    # Check environment variable first
    env_data_dir = os.environ.get(env_var)
    if env_data_dir:
        return Path(env_data_dir)

    # Hardcoded absolute path (required for exec() context where __file__ is not defined)
    absolute_path = Path("/Users/wzxu/Projects/Pantheon/pantheon-agents/examples/evolution_batch_correction/data")
    if absolute_path.exists():
        return absolute_path

    # Try relative path (works when running directly with __file__ defined)
    try:
        relative_path = Path(__file__).parent / "data"
        if relative_path.exists():
            return relative_path
    except NameError:
        pass  # __file__ not defined in exec() context

    return absolute_path


def compute_standard_metrics(
    X_corrected: np.ndarray,
    X_original: np.ndarray,
    batch_labels: np.ndarray,
    celltype_labels: np.ndarray,
    execution_time: float,
    include_convergence: bool = False,
    convergence_score: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute all standard batch correction metrics.

    Args:
        X_corrected: Corrected embedding
        X_original: Original embedding
        batch_labels: Batch assignments
        celltype_labels: True cell type labels
        execution_time: Time taken to run the algorithm
        include_convergence: If True, include convergence_score
        convergence_score: Pre-computed convergence score (required if include_convergence)

    Returns:
        Dictionary with all metrics and fitness_weights
    """
    mixing_score = compute_batch_mixing_score(X_corrected, batch_labels)
    bio_score = compute_bio_conservation_score(X_corrected, X_original, celltype_labels)
    speed_score = 1.0 / (1 + execution_time)

    result = {
        "mixing_score": mixing_score,
        "bio_conservation_score": bio_score,
        "speed_score": speed_score,
        "execution_time": execution_time,
        "n_cells": len(X_original),
        "n_batches": len(np.unique(batch_labels)),
        "fitness_weights": get_default_fitness_weights(include_convergence),
    }

    if include_convergence and convergence_score is not None:
        result["convergence_score"] = convergence_score

    return result


def error_result(message: str) -> Dict[str, Any]:
    """
    Create standard error result for evaluators.

    Args:
        message: Error description

    Returns:
        Dictionary with function_score=0.0 and error message
    """
    return {
        "function_score": 0.0,
        "error": message,
    }
