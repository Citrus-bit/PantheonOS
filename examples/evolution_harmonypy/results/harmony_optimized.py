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
        self.Y = None  # Cluster centroids
        self.Phi = None  # (deprecated) Batch membership matrix (removed for perf/memory)
        self.objectives = []

        # Batch bookkeeping (set during fit)
        self.unique_batches = None
        self.batch_id = None  # (n_cells,) integer ids in [0, n_batches)
        self.batch_indices = None  # List[np.ndarray], indices per batch
        self.batch_props = None  # (n_batches,)

        # Removed dense batch indicator matrix (_H/_Ht) to avoid O(N*B) memory/time
        self._H = None
        self._Ht = None

        # Diversity smoothing (EMA) cache
        self._O_ema = None  # (K,B) exponential moving average of O
        self._O_ema_beta = 0.8

        # Caches
        self._dist = None  # (n_clusters x n_cells)
        self._Z_sq = None  # (n_cells x 1) cache of ||Z_corr||^2
        self._dist_valid = False
        self._Z_version = 0
        self._Y_version = 0
        self._dist_Z_version = -1
        self._dist_Y_version = -1

        # Diversity sufficient statistics cache (computed in _update_R, reused in objective)
        self._R_version = 0
        self._last_O_num = None  # (K,B)
        self._last_R_sum = None  # (K,1)
        self._last_O_num_R_version = -1

        # Cluster convergence sampling indices (set during fit)
        self._cluster_sample_idx = None

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
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")
        if self.sigma < 1e-3:
            raise ValueError("sigma must be >= 1e-3 to avoid degenerate (nearly hard) assignments")
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be > 0")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be > 0")
        if self.max_iter_kmeans <= 0:
            raise ValueError("max_iter_kmeans must be > 0")
        if self.lamb < 0:
            raise ValueError("lamb must be >= 0")
        if self.theta < 0:
            raise ValueError("theta must be >= 0")

        if X.ndim != 2:
            raise ValueError("X must be a 2D array (n_cells x n_features)")
        if batch_labels.shape[0] != X.shape[0]:
            raise ValueError("batch_labels must have the same length as X has rows")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf")

        n_cells, _ = X.shape

        # Store original (use float32 for speed/memory; compute sensitive ops in float64 where needed)
        X = np.asarray(X, dtype=np.float32)
        self.Z_orig = X.copy()
        self.Z_corr = X.copy()

        # Invalidate caches
        self._dist = None
        self._Z_sq = None
        self._dist_valid = False
        self._Z_version += 1

        # Batch bookkeeping (no dense Phi)
        self.unique_batches, inv = np.unique(batch_labels, return_inverse=True)
        n_batches = len(self.unique_batches)
        if n_batches < 1:
            raise ValueError("No batches found in batch_labels")

        self.batch_id = inv.astype(np.int64, copy=False)
        self.batch_indices = [np.flatnonzero(self.batch_id == b) for b in range(n_batches)]
        counts = np.bincount(self.batch_id, minlength=n_batches).astype(np.float64)
        self.batch_props = counts / float(n_cells)

        # Do NOT build dense (N x B) one-hot matrix; use index reductions instead
        self._H = None
        self._Ht = None

        # Deprecated (kept for backward compatibility; no longer used)
        self.Phi = None

        # Sample indices for cheaper clustering convergence checks
        rng = np.random.default_rng(self.random_state)
        sample_size = int(min(1000, n_cells))
        self._cluster_sample_idx = (
            rng.choice(n_cells, size=sample_size, replace=False) if sample_size > 0 else None
        )

        # Initialize clusters
        self._init_clusters()

        # Initialize diversity EMA after clusters exist (K is known)
        # Will be updated on first _update_R call in the loop.
        self._O_ema = None

        # Main Harmony loop
        self.objectives = []
        for iteration in range(self.max_iter):
            # Anneal theta (warmup -> ramp -> slight taper)
            if self.max_iter <= 1:
                theta_t = float(self.theta)
            else:
                t = float(iteration) / float(self.max_iter - 1)
                warmup = 0.2
                ramp_end = 0.8
                if t < warmup:
                    s = 0.3 + 0.7 * (t / warmup)  # 0.3 -> 1.0
                elif t < ramp_end:
                    s = 1.0
                else:
                    s = 1.0 - 0.1 * ((t - ramp_end) / max(1e-8, (1.0 - ramp_end)))  # 1.0 -> 0.9
                theta_t = float(self.theta) * float(s)

            # Clustering step
            self._cluster(theta_t=theta_t)

            # Correction step
            self._correct(iteration=iteration)

            # Check convergence
            obj = self._compute_objective(theta_t=theta_t)
            self.objectives.append(obj)

            if iteration > 0:
                prev = self.objectives[-2]
                curr = self.objectives[-1]
                denom = abs(prev) + 1e-8
                rel_change = abs(prev - curr) / denom
                if rel_change < self.epsilon_harmony:
                    break

        return self

    def _init_clusters(self):
        """Initialize cluster centroids using k-means."""
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init="auto",
            max_iter=self.max_iter_kmeans,
        )
        kmeans.fit(self.Z_corr)
        self.Y = kmeans.cluster_centers_.astype(np.float32, copy=False).T  # (n_features x n_clusters)
        self._Y_version += 1
        self._dist_valid = False

        # Initialize soft assignments
        self._update_R()

    def _cluster(self, theta_t: Optional[float] = None):
        """Run clustering iterations."""
        for it in range(self.max_iter_kmeans):
            # Update centroids
            self._update_centroids()

            # Update soft assignments
            R_old = self.R.copy() if self.R is not None else None
            self._update_R(theta_t=theta_t)

            # Check convergence on a sample (robust statistic) + periodic broader check
            if R_old is not None and self._cluster_sample_idx is not None:
                idx = self._cluster_sample_idx
                diff = np.abs(self.R[:, idx] - R_old[:, idx])

                # Use high percentile on sample (less likely to hide a subset of movers)
                r_change = float(np.quantile(diff, 0.95))

                # Every few iterations, tighten check with max on a larger sample
                if (it + 1) % 5 == 0:
                    rng = np.random.default_rng(self.random_state)
                    n_cells = self.R.shape[1]
                    big_n = int(min(5000, n_cells))
                    big_idx = rng.choice(n_cells, size=big_n, replace=False) if big_n > 0 else idx
                    r_change = max(r_change, float(np.max(np.abs(self.R[:, big_idx] - R_old[:, big_idx]))))

                if r_change < self.epsilon_cluster:
                    break

    def _update_centroids(self):
        """Update cluster centroids."""
        # Weighted average of cells
        weights = self.R  # (n_clusters x n_cells)
        weights_sum = weights.sum(axis=1, keepdims=True) + 1e-8

        # Y = Z @ R.T / sum(R)
        self.Y = (self.Z_corr.T @ weights.T) / weights_sum.T
        self._Y_version += 1
        self._dist_valid = False

    def _batch_sums_from_R(self, R: np.ndarray) -> np.ndarray:
        """Compute per-batch sums of responsibilities: O_num[:, b] = sum_i in batch b R[:, i]."""
        if self.batch_indices is None:
            raise ValueError("batch_indices not initialized; call fit() first")
        K = R.shape[0]
        B = len(self.batch_indices)
        out = np.zeros((K, B), dtype=np.float32)
        for b, idx_b in enumerate(self.batch_indices):
            if idx_b.size == 0:
                continue
            out[:, b] = R[:, idx_b].sum(axis=1, dtype=np.float32)
        return out

    def _update_R(self, theta_t: Optional[float] = None):
        """Update soft cluster assignments with diversity penalty."""
        # Compute distances to centroids
        # dist[k, i] = ||z_i - y_k||^2
        dist = self._compute_distances()

        # Work fully in log-space for stability and to avoid extra exp/multiply passes
        # Gaussian kernel: exp(-dist / (2*sigma^2))
        logR = (-dist / (2.0 * (self.sigma ** 2))).astype(np.float32, copy=False)  # (K,N)

        # Apply batch-conditional diversity penalty (index-reduced, smoothed, size-aware)
        theta_use = float(self.theta if theta_t is None else theta_t)
        if (
            theta_use > 0
            and self.batch_id is not None
            and self.batch_props is not None
            and self.batch_indices is not None
        ):
            eps = 1e-8

            # First-pass (no penalty) soft weights, needed to compute O
            logR0 = logR - logR.max(axis=0, keepdims=True)
            R0 = np.exp(logR0, dtype=np.float32)

            R_sum = R0.sum(axis=1, keepdims=True).astype(np.float32, copy=False) + eps  # (K,1)

            # O_num via index reductions (K,B)
            O_num = self._batch_sums_from_R(R0)
            O = O_num / R_sum  # (K,B)
            expected = self.batch_props.astype(np.float32, copy=False)[np.newaxis, :]  # (1,B)

            # Floor O to avoid extreme penalties when O ~= 0
            o_floor = max(1e-4, 1.0 / (10.0 * float(self.Z_corr.shape[0])))
            O = np.maximum(O, np.float32(o_floor))

            # Smooth O across iterations (EMA) to reduce oscillation
            if self._O_ema is None or self._O_ema.shape != O.shape:
                self._O_ema = O.copy()
            else:
                beta = float(self._O_ema_beta)
                self._O_ema = (beta * self._O_ema + (1.0 - beta) * O).astype(np.float32, copy=False)

            O_use = self._O_ema

            # Cluster-size aware weight to downweight tiny/noisy clusters
            Nk = R_sum[:, 0]  # (K,)
            c = 50.0
            w_k = (Nk / (Nk + c)).astype(np.float32, copy=False)  # (K,)

            # Stabilized log-space adjustment with clamp
            L = 10.0
            log_adjust = theta_use * (np.log(expected + eps) - np.log(O_use + eps))  # (K,B)
            log_adjust = (log_adjust * w_k[:, None]).astype(np.float32, copy=False)
            log_adjust = np.clip(log_adjust, -L, L).astype(np.float32, copy=False)

            # Add penalty in log-space per-cell via batch lookup
            logR = logR + log_adjust[:, self.batch_id]

        # Stabilize once and exponentiate
        logR = logR - logR.max(axis=0, keepdims=True)
        R = np.exp(logR, dtype=np.float32)

        # Normalize to get probabilities
        R = R / (R.sum(axis=0, keepdims=True) + 1e-8)

        self.R = R
        self._R_version += 1

        # Cache sufficient statistics for objective reuse (versioning fixed)
        if (
            theta_use > 0
            and self.batch_id is not None
            and self.batch_props is not None
            and self.batch_indices is not None
        ):
            self._last_R_sum = self.R.sum(axis=1, keepdims=True).astype(np.float32, copy=False) + 1e-8
            self._last_O_num = self._batch_sums_from_R(self.R.astype(np.float32, copy=False))
            self._last_O_num_R_version = self._R_version

    def _compute_distances(self) -> np.ndarray:
        """Compute squared distances from cells to centroids."""
        # ||z - y||^2 = ||z||^2 + ||y||^2 - 2 * z @ y
        if (
            self._dist_valid
            and self._dist is not None
            and self._dist_Z_version == self._Z_version
            and self._dist_Y_version == self._Y_version
        ):
            return self._dist

        if self._Z_sq is None:
            self._Z_sq = np.sum(self.Z_corr ** 2, axis=1, keepdims=True).astype(np.float32, copy=False)  # (n_cells x 1)
        Z_sq = self._Z_sq  # (N,1) float32
        Y_sq = np.sum(self.Y ** 2, axis=0, keepdims=True).astype(np.float32, copy=False)  # (1,K) float32

        # Compute (K,N) directly to avoid transpose/memory traffic
        cross = (self.Y.T @ self.Z_corr.T).astype(np.float32, copy=False)  # (K,N)

        dist = (Z_sq.T + Y_sq.T - 2.0 * cross).astype(np.float32, copy=False)  # (K,N)

        self._dist = dist
        self._dist_valid = True
        self._dist_Z_version = self._Z_version
        self._dist_Y_version = self._Y_version
        return dist

    def _correct(self, iteration: int = 0):
        """Apply mean-based correction to remove batch effects (fast + stable)."""
        if self.batch_indices is None:
            return

        eps = 1e-8
        Z = self.Z_corr  # (N,F) float32

        # Damping/step size: stronger early, decays to reduce late oscillations/overcorrection
        alpha0 = 0.8
        alpha = max(0.3, alpha0 * (0.9 ** float(iteration)))

        # Compute cluster means in float32/float64 mix (avoid casting large matrices repeatedly)
        R = self.R.astype(np.float32, copy=False)  # (K,N)
        Nk = R.sum(axis=1, dtype=np.float64) + eps  # (K,)
        mu_k = (R @ Z).astype(np.float32, copy=False) / Nk[:, None].astype(np.float32, copy=False)  # (K,F)

        # Compute a single global cap per iteration on a subsample (cheap + stable)
        rng = np.random.default_rng(self.random_state)
        n_cells = Z.shape[0]
        sample_n = int(min(2000, n_cells))
        sample_idx = rng.choice(n_cells, size=sample_n, replace=False) if sample_n > 0 else None
        global_cap = None
        if sample_idx is not None and sample_idx.size > 0:
            C_sample = (R[:, sample_idx].T @ mu_k).astype(np.float32, copy=False)  # (n_s,F)
            norms_s = np.linalg.norm(C_sample, axis=1) + 1e-12
            global_cap = float(np.quantile(norms_s, 0.95))

        for b, idx_b in enumerate(self.batch_indices):
            if idx_b.size == 0:
                continue

            Rb = R[:, idx_b]  # (K, n_b) float32

            Nk_b = Rb.sum(axis=1, dtype=np.float64) + eps  # (K,)
            mu_kb = (Rb @ Z[idx_b]).astype(np.float32, copy=False) / Nk_b[:, None].astype(np.float32, copy=False)  # (K,F)

            Delta_kb = (mu_kb - mu_k).astype(np.float32, copy=False)  # (K,F)
            shrink_kb = (Nk_b / (Nk_b + float(self.lamb))).astype(np.float32, copy=False)  # (K,)

            # Reorder: Delta2 = Delta * shrink, then Cb = Rb.T @ Delta2 (avoids n_b x K transient)
            Delta2 = (Delta_kb * shrink_kb[:, None]).astype(np.float32, copy=False)  # (K,F)
            Cb = (Rb.T @ Delta2).astype(np.float32, copy=False)  # (n_b,F)

            # Apply a single global cap (if available) for consistency across batches
            if global_cap is not None and global_cap > 0:
                norms = np.linalg.norm(Cb, axis=1) + 1e-12
                scale = np.minimum(1.0, global_cap / norms).astype(np.float32, copy=False)
                Cb = Cb * scale[:, None]

            Z[idx_b] -= (alpha * Cb).astype(Z.dtype, copy=False)

        # Invalidate caches because Z_corr changed
        self._dist = None
        self._Z_sq = None
        self._dist_valid = False
        self._Z_version += 1

    def _compute_objective(self, theta_t: Optional[float] = None) -> float:
        """Compute the Harmony objective function."""
        dist = self._compute_distances()
        cluster_obj = float(np.sum(self.R * dist))

        theta_use = float(self.theta if theta_t is None else theta_t)

        # Diversity objective (entropy of batch distribution per cluster), computed via index reductions
        if theta_use <= 0 or self.batch_indices is None:
            return cluster_obj

        eps = 1e-8
        expected = self.batch_props.astype(np.float32, copy=False)[np.newaxis, :]  # (1,B)

        # Reuse cached sufficient statistics from _update_R when possible
        if (
            self._last_O_num is not None
            and self._last_R_sum is not None
            and self._last_O_num_R_version == self._R_version
        ):
            O_num = self._last_O_num
            R_sum = self._last_R_sum
        else:
            R_sum = self.R.sum(axis=1, keepdims=True).astype(np.float32, copy=False) + eps  # (K,1)
            O_num = self._batch_sums_from_R(self.R.astype(np.float32, copy=False))

        O = O_num / R_sum  # (K,B)
        diversity_obj = float(
            theta_use * np.sum(O * np.log((O + eps) / (expected + eps)))
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
