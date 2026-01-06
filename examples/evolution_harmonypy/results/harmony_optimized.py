# File: harmony.py
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
from typing import Optional


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
        self.Y = None  # Cluster centroids (n_features x n_clusters)
        self.Phi = None  # Batch membership matrix (n_batches x n_cells)
        self.objectives = []

        # Cached / precomputed for performance
        self._design = None          # (n_cells x n_covariates), covariates = intercept + (n_batches-1)
        self._design_T = None        # transpose view
        self._I_ridge = None         # ridge identity matrix (n_covariates x n_covariates)
        self._sigma_inv = None       # 1/sigma
        self._expected_batch = None  # (1 x n_batches)

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
        self.Phi = np.zeros((n_batches, n_cells), dtype=np.float64)
        for i, batch in enumerate(unique_batches):
            self.Phi[i, batch_labels == batch] = 1.0

        # Compute batch proportions
        self.batch_props = self.Phi.sum(axis=1) / n_cells
        self._expected_batch = self.batch_props[np.newaxis, :]  # (1 x n_batches)

        # Precompute design matrix once: intercept + batch indicators (drop first)
        # design: (n_cells x (1 + n_batches - 1)) = (n_cells x n_batches)
        self._design = np.empty((n_cells, n_batches), dtype=np.float64)
        self._design[:, 0] = 1.0
        if n_batches > 1:
            self._design[:, 1:] = self.Phi[1:, :].T
        self._design_T = self._design.T

        # Cache ridge identity and sigma inverse
        self._I_ridge = np.eye(self._design.shape[1], dtype=np.float64)
        self._sigma_inv = 1.0 / float(self.sigma)

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
        # Compute distances to centroids: dist (n_clusters x n_cells)
        dist = self._compute_distances()

        # Soft assignments (before diversity correction)
        # Use cached 1/sigma to avoid division in hot path
        R = np.exp(-dist * self._sigma_inv)

        # Apply diversity penalty
        # Penalize clusters that are dominated by a single batch
        if self.theta > 0:
            R_sum = R.sum(axis=1, keepdims=True) + 1e-8
            O = (R @ self.Phi.T) / R_sum  # (n_clusters x n_batches)

            # penalty[k] = theta * KL(O_k || expected)
            expected = self._expected_batch
            penalty = self.theta * np.sum(
                O * np.log((O + 1e-8) / (expected + 1e-8)),
                axis=1,
                keepdims=True,
            )

            # Apply penalty (broadcast to cells)
            R *= np.exp(-penalty)

        # Normalize to get probabilities
        R /= (R.sum(axis=0, keepdims=True) + 1e-8)
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
        # Precomputed design: intercept + (n_batches-1) indicators, with reference batch dropped
        design = self._design
        design_T = self._design_T
        I_ridge = self._I_ridge
        Z = self.Z_corr

        # For each cluster, compute and apply correction
        # Implement weighted ridge without forming diag(W):
        # X'WX = X' (w * X), X'WZ = X' (w * Z)
        for k in range(self.n_clusters):
            w = self.R[k, :]  # (n_cells,)
            w_sum = float(w.sum())
            if w_sum < 1e-8:
                continue

            # Weighted design and response
            Xw = design * w[:, None]          # (n_cells x n_cov)
            XWX = design_T @ Xw               # (n_cov x n_cov)
            XWX += self.lamb * I_ridge

            Z_w = Z * w[:, None]              # (n_cells x n_features)
            XWZ = design_T @ Z_w              # (n_cov x n_features)

            try:
                beta = np.linalg.solve(XWX, XWZ)
            except np.linalg.LinAlgError:
                continue

            # Remove batch effects (keep intercept): effect = X[:,1:] @ beta[1:,:]
            if beta.shape[0] > 1:
                batch_effect = design[:, 1:] @ beta[1:, :]
                Z -= w[:, None] * batch_effect

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
