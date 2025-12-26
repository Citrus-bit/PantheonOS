"""
Evaluator for Harmony Algorithm Evolution.

This evaluator measures:
1. Integration quality (how well batches are mixed)
2. Biological variance preservation (how well structure is preserved)
3. Execution speed
4. Convergence behavior

The combined score balances these metrics for evolution.
"""

import numpy as np
import time
import sys
import importlib.util
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from typing import Dict, Any, Tuple


def generate_test_data(
    n_cells: int = 2000,
    n_features: int = 50,
    n_batches: int = 3,
    n_clusters: int = 5,
    batch_effect_strength: float = 2.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic single-cell data with batch effects.

    Args:
        n_cells: Total number of cells
        n_features: Number of features (PCs)
        n_batches: Number of batches
        n_clusters: Number of biological clusters
        batch_effect_strength: Strength of batch effects
        random_state: Random seed

    Returns:
        X: Data matrix (n_cells x n_features)
        batch_labels: Batch assignments
        true_labels: True biological cluster labels
    """
    np.random.seed(random_state)

    cells_per_batch = n_cells // n_batches

    X_list = []
    batch_list = []
    label_list = []

    # Generate cluster centers
    cluster_centers = np.random.randn(n_clusters, n_features) * 3

    for batch_idx in range(n_batches):
        # Batch-specific offset
        batch_offset = np.random.randn(n_features) * batch_effect_strength

        for cluster_idx in range(n_clusters):
            # Cells per cluster per batch
            n = cells_per_batch // n_clusters

            # Generate cells around cluster center with batch effect
            cells = (
                cluster_centers[cluster_idx]
                + np.random.randn(n, n_features) * 0.5
                + batch_offset
            )

            X_list.append(cells)
            batch_list.extend([batch_idx] * n)
            label_list.extend([cluster_idx] * n)

    X = np.vstack(X_list)
    batch_labels = np.array(batch_list)
    true_labels = np.array(label_list)

    return X, batch_labels, true_labels


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
    n_batches = len(unique_batches)

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

        # Compare to expected (lower KL divergence = better mixing)
        # Use simple correlation instead for stability
        score = 1 - np.sqrt(np.mean((observed_props - expected_props) ** 2))
        mixing_scores.append(max(0, score))

    return np.mean(mixing_scores)


def compute_bio_conservation_score(
    X_corrected: np.ndarray,
    X_original: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """
    Compute biological structure conservation score.

    Measures how well the biological clusters are preserved after correction.

    Args:
        X_corrected: Corrected embedding
        X_original: Original embedding
        true_labels: True biological labels

    Returns:
        Conservation score in [0, 1]
    """
    try:
        # Silhouette score on corrected data
        if len(np.unique(true_labels)) > 1:
            silhouette_corrected = silhouette_score(X_corrected, true_labels)
            silhouette_original = silhouette_score(X_original, true_labels)

            # Normalize to [0, 1] (silhouette is in [-1, 1])
            score_corrected = (silhouette_corrected + 1) / 2
            score_original = (silhouette_original + 1) / 2

            # We want corrected to be at least as good as original
            # Bonus if it's better, penalty if worse
            if score_original > 0:
                ratio = score_corrected / score_original
                return min(1.0, ratio)
            else:
                return score_corrected
        else:
            return 0.5
    except Exception:
        return 0.5


def compute_convergence_score(objectives: list) -> float:
    """
    Compute convergence behavior score.

    Rewards fast and stable convergence.

    Args:
        objectives: List of objective values over iterations

    Returns:
        Convergence score in [0, 1]
    """
    if len(objectives) < 2:
        return 0.5

    # Check if converged (objective stabilized)
    final_change = abs(objectives[-1] - objectives[-2]) / (abs(objectives[-2]) + 1e-8)

    # Reward small final change
    convergence_quality = np.exp(-final_change * 10)

    # Reward fewer iterations (faster convergence)
    speed_bonus = 1.0 / (1 + len(objectives) / 10)

    return 0.7 * convergence_quality + 0.3 * speed_bonus


def evaluate(workspace_path: str) -> Dict[str, Any]:
    """
    Evaluate the Harmony implementation.

    This is the main evaluation function called by Pantheon Evolution.

    Args:
        workspace_path: Path to the workspace containing harmony.py

    Returns:
        Dictionary with metrics including 'combined_score'
    """
    workspace = Path(workspace_path)

    # Load the harmony module from workspace
    harmony_path = workspace / "harmony.py"
    if not harmony_path.exists():
        return {
            "combined_score": 0.0,
            "error": "harmony.py not found",
        }

    try:
        spec = importlib.util.spec_from_file_location("harmony", harmony_path)
        harmony_module = importlib.util.module_from_spec(spec)
        sys.modules["harmony"] = harmony_module
        spec.loader.exec_module(harmony_module)
    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": f"Failed to load harmony.py: {e}",
        }

    # Generate test data
    X, batch_labels, true_labels = generate_test_data(
        n_cells=2000,
        n_features=50,
        n_batches=3,
        n_clusters=5,
        random_state=42,
    )

    # Run harmony and measure time
    try:
        start_time = time.time()
        hm = harmony_module.run_harmony(
            X,
            batch_labels,
            n_clusters=50,
            max_iter=10,
            random_state=42,
        )
        execution_time = time.time() - start_time

        X_corrected = hm.Z_corr
        objectives = hm.objectives

    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": f"Harmony execution failed: {e}",
        }

    # Compute metrics
    try:
        # Batch mixing (higher = better, weight: 0.4)
        mixing_score = compute_batch_mixing_score(X_corrected, batch_labels)

        # Biological conservation (higher = better, weight: 0.3)
        bio_score = compute_bio_conservation_score(X_corrected, X, true_labels)

        # Speed score (faster = better, weight: 0.2)
        # Baseline ~1 second, reward sub-second
        speed_score = 1.0 / (1 + execution_time)

        # Convergence score (weight: 0.1)
        conv_score = compute_convergence_score(objectives)

        # Combined score
        combined_score = (
            0.4 * mixing_score +
            0.3 * bio_score +
            0.2 * speed_score +
            0.1 * conv_score
        )

        return {
            "combined_score": combined_score,
            "mixing_score": mixing_score,
            "bio_conservation_score": bio_score,
            "speed_score": speed_score,
            "convergence_score": conv_score,
            "execution_time": execution_time,
            "iterations": len(objectives),
        }

    except Exception as e:
        return {
            "combined_score": 0.1,  # Partial credit for running
            "error": f"Metric computation failed: {e}",
        }


if __name__ == "__main__":
    # Test the evaluator locally
    import os

    # Use current directory as workspace
    workspace = os.path.dirname(os.path.abspath(__file__))
    result = evaluate(workspace)

    print("Evaluation Results:")
    print("-" * 40)
    for key, value in result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
