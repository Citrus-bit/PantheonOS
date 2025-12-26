"""
Harmony Algorithm for Data Integration.

This is a simplified implementation of the Harmony algorithm for integrating
multiple high-dimensional datasets. It uses fuzzy k-means clustering and
linear corrections to remove batch effects while preserving biological structure.

Reference:
    Korsunsky et al., "Fast, sensitive and accurate integration of single-cell
    data with Harmony", Nature Methods, 2019.

This implementation is designed to be optimized by Pantheon Evolution.
"""

import numpy as np
from sklearn.cluster import KMeans
from typing import Optional, Tuple, List


class Harmony:
    """
    Harmony algorithm for batch effect correction.

    Attributes:
        Z_corr: Corrected embedding after harmonization
        Z_orig: Original embedding
        R: Soft cluster assignments (cells x clusters)
        objectives: History of objective function values
    """

    def __init__(
        self,
        n_clusters: int = 100,
        theta: float = 2.0,
        sigma: float = 0.1,
        lamb: float = 1.0,
        max_iter: int = 10,
        max_iter_kmeans: int = 20,
        epsilon_cluster: float = 1e-5,
        epsilon_harmony: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        """
        Initialize Harmony.

        Args:
            n_clusters: Number of clusters for k-means
            theta: Diversity clustering penalty parameter
            sigma: Width of soft k-means clusters
            lamb: Ridge regression penalty
            max_iter: Maximum iterations of Harmony algorithm
            max_iter_kmeans: Maximum iterations for clustering step
            epsilon_cluster: Convergence threshold for clustering
            epsilon_harmony: Convergence threshold for Harmony
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.theta = theta
        self.sigma = sigma
        self.lamb = lamb
        self.max_iter = max_iter
        self.max_iter_kmeans = max_iter_kmeans
        self.epsilon_cluster = epsilon_cluster
        self.epsilon_harmony = epsilon_harmony
        self.random_state = random_state

        # Will be set during fit
        self.Z_orig = None
        self.Z_corr = None
        self.R = None
        self.Y = None  # Cluster centroids
        self.Phi = None  # Batch membership matrix
        self.objectives = []

    def fit(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray,
    ) -> "Harmony":
        """
        Fit Harmony to the data.

        Args:
            X: Data matrix (n_cells x n_features), typically PCA coordinates
            batch_labels: Batch labels for each cell (n_cells,)

        Returns:
            self with Z_corr containing corrected coordinates
        """
        n_cells, n_features = X.shape

        # Store original
        self.Z_orig = X.copy()
        self.Z_corr = X.copy()

        # Create batch membership matrix (one-hot encoding)
        unique_batches = np.unique(batch_labels)
        n_batches = len(unique_batches)
        self.Phi = np.zeros((n_batches, n_cells))
        for i, batch in enumerate(unique_batches):
            self.Phi[i, batch_labels == batch] = 1

        # Compute batch proportions
        self.batch_props = self.Phi.sum(axis=1) / n_cells

        # Initialize clusters
        self._init_clusters()

        # Main Harmony loop
        self.objectives = []
        for iteration in range(self.max_iter):
            # Clustering step
            self._cluster()

            # Correction step
            self._correct()

            # Check convergence
            obj = self._compute_objective()
            self.objectives.append(obj)

            if iteration > 0:
                obj_change = abs(self.objectives[-2] - self.objectives[-1])
                if obj_change < self.epsilon_harmony:
                    break

        return self

    def _init_clusters(self):
        """Initialize cluster centroids using k-means."""
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=1,
            max_iter=25,
        )
        kmeans.fit(self.Z_corr)
        self.Y = kmeans.cluster_centers_.T  # (n_features x n_clusters)

        # Initialize soft assignments
        self._update_R()

    def _cluster(self):
        """Run clustering iterations."""
        for _ in range(self.max_iter_kmeans):
            # Update centroids
            self._update_centroids()

            # Update soft assignments
            R_old = self.R.copy() if self.R is not None else None
            self._update_R()

            # Check convergence
            if R_old is not None:
                r_change = np.abs(self.R - R_old).max()
                if r_change < self.epsilon_cluster:
                    break

    def _update_centroids(self):
        """Update cluster centroids."""
        # Weighted average of cells
        weights = self.R  # (n_clusters x n_cells)
        weights_sum = weights.sum(axis=1, keepdims=True) + 1e-8

        # Y = Z @ R.T / sum(R)
        self.Y = (self.Z_corr.T @ weights.T) / weights_sum.T

    def _update_R(self):
        """Update soft cluster assignments with diversity penalty."""
        n_cells = self.Z_corr.shape[0]

        # Compute distances to centroids
        # dist[k, i] = ||z_i - y_k||^2
        dist = self._compute_distances()

        # Soft assignments (before diversity correction)
        R = np.exp(-dist / self.sigma)

        # Apply diversity penalty
        # Penalize clusters that are dominated by a single batch
        if self.theta > 0:
            # Compute expected batch proportions per cluster
            # O[b, k] = sum_i(R[k,i] * Phi[b,i]) / sum_i(R[k,i])
            R_sum = R.sum(axis=1, keepdims=True) + 1e-8
            O = (R @ self.Phi.T) / R_sum  # (n_clusters x n_batches)

            # Diversity penalty
            # penalty[k] = sum_b(theta * O[k,b] * log(O[k,b] / batch_props[b]))
            expected = self.batch_props[np.newaxis, :]  # (1 x n_batches)
            penalty = self.theta * np.sum(
                O * np.log((O + 1e-8) / (expected + 1e-8)),
                axis=1,
                keepdims=True,
            )

            # Apply penalty
            R = R * np.exp(-penalty)

        # Normalize to get probabilities
        R = R / (R.sum(axis=0, keepdims=True) + 1e-8)

        self.R = R

    def _compute_distances(self) -> np.ndarray:
        """Compute squared distances from cells to centroids."""
        # ||z - y||^2 = ||z||^2 + ||y||^2 - 2 * z @ y
        Z_sq = np.sum(self.Z_corr ** 2, axis=1, keepdims=True)  # (n_cells x 1)
        Y_sq = np.sum(self.Y ** 2, axis=0, keepdims=True)  # (1 x n_clusters)
        cross = self.Z_corr @ self.Y  # (n_cells x n_clusters)

        dist = Z_sq + Y_sq - 2 * cross  # (n_cells x n_clusters)
        return dist.T  # (n_clusters x n_cells)

    def _correct(self):
        """Apply linear correction to remove batch effects."""
        n_cells = self.Z_corr.shape[0]
        n_features = self.Z_corr.shape[1]
        n_batches = self.Phi.shape[0]

        # For each cluster, compute and apply correction
        for k in range(self.n_clusters):
            # Get cells in this cluster (soft membership)
            weights = self.R[k, :]  # (n_cells,)

            # Skip if cluster is empty
            if weights.sum() < 1e-8:
                continue

            # Weighted design matrix: [1, Phi.T] with weights
            # We want to regress out batch effects
            W = np.diag(weights)

            # Design matrix: intercept + batch indicators (drop first for identifiability)
            design = np.vstack([
                np.ones(n_cells),
                self.Phi[1:, :],  # Drop first batch as reference
            ]).T  # (n_cells x n_batches)

            # Weighted least squares with ridge penalty
            # beta = (X'WX + lambda*I)^-1 X'WZ
            XWX = design.T @ W @ design
            XWX += self.lamb * np.eye(XWX.shape[0])

            XWZ = design.T @ W @ self.Z_corr

            try:
                beta = np.linalg.solve(XWX, XWZ)
            except np.linalg.LinAlgError:
                continue

            # Remove batch effects (keep intercept)
            # Z_corr = Z - Phi.T @ beta[1:, :]
            batch_effect = design[:, 1:] @ beta[1:, :]

            # Apply correction weighted by cluster membership
            self.Z_corr -= weights[:, np.newaxis] * batch_effect

    def _compute_objective(self) -> float:
        """Compute the Harmony objective function."""
        # Clustering objective (within-cluster variance)
        dist = self._compute_distances()
        cluster_obj = np.sum(self.R * dist)

        # Diversity objective (entropy of batch distribution per cluster)
        R_sum = self.R.sum(axis=1, keepdims=True) + 1e-8
        O = (self.R @ self.Phi.T) / R_sum
        expected = self.batch_props[np.newaxis, :]
        diversity_obj = self.theta * np.sum(
            O * np.log((O + 1e-8) / (expected + 1e-8))
        )

        return cluster_obj + diversity_obj

    def transform(self, X: np.ndarray, batch_labels: np.ndarray) -> np.ndarray:
        """
        Transform new data using fitted model.

        Args:
            X: New data matrix (n_cells x n_features)
            batch_labels: Batch labels for new cells

        Returns:
            Corrected coordinates
        """
        # This is a simplified transform - in practice would need more work
        return X


def run_harmony(
    X: np.ndarray,
    batch_labels: np.ndarray,
    n_clusters: int = 100,
    theta: float = 2.0,
    sigma: float = 0.1,
    lamb: float = 1.0,
    max_iter: int = 10,
    random_state: Optional[int] = None,
) -> Harmony:
    """
    Run Harmony algorithm.

    Args:
        X: Data matrix (n_cells x n_features), typically PCA coordinates
        batch_labels: Batch labels for each cell
        n_clusters: Number of clusters
        theta: Diversity penalty parameter
        sigma: Soft clustering width
        lamb: Ridge regression penalty
        max_iter: Maximum iterations
        random_state: Random seed

    Returns:
        Fitted Harmony object with Z_corr attribute containing corrected data

    Example:
        >>> X = np.random.randn(1000, 50)  # 1000 cells, 50 PCs
        >>> batch = np.repeat([0, 1, 2], [300, 400, 300])
        >>> hm = run_harmony(X, batch)
        >>> X_corrected = hm.Z_corr
    """
    hm = Harmony(
        n_clusters=n_clusters,
        theta=theta,
        sigma=sigma,
        lamb=lamb,
        max_iter=max_iter,
        random_state=random_state,
    )
    hm.fit(X, batch_labels)
    return hm
