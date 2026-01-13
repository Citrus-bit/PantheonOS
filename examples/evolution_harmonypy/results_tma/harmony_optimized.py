# File: harmony.py
# harmonypy - A data alignment algorithm.
# Copyright (C) 2018  Ilya Korsunsky
#               2019  Kamil Slowikowski <kslowikowski@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans
import logging

# create logger
logger = logging.getLogger('harmonypy')
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def get_device(device=None):
    """Get the appropriate device for PyTorch operations."""
    if device is not None:
        return torch.device(device)

    # Check for available accelerators
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def run_harmony(
    data_mat: np.ndarray,
    meta_data: pd.DataFrame,
    vars_use,
    theta=None,
    lamb=None,
    sigma=0.1,
    nclust=None,
    tau=0,
    block_size=0.05,
    max_iter_harmony=10,
    max_iter_kmeans=20,
    epsilon_cluster=1e-5,
    epsilon_harmony=1e-4,
    alpha=0.2,
    verbose=True,
    random_state=0,
    device=None
):
    """Run Harmony batch effect correction.

    This is a PyTorch implementation matching the R package formulas.
    Supports CPU and GPU (CUDA, MPS) acceleration.

    Parameters
    ----------
    data_mat : np.ndarray
        PCA embedding matrix (cells x PCs or PCs x cells)
    meta_data : pd.DataFrame
        Metadata with batch variables (cells x variables)
    vars_use : str or list
        Column name(s) in meta_data to use for batch correction
    theta : float or list, optional
        Diversity penalty parameter(s). Default is 2 for each batch.
    lamb : float or list, optional
        Ridge regression penalty. Default is 1 for each batch.
        If -1, lambda is estimated automatically (matches R package).
    sigma : float, optional
        Kernel bandwidth for soft clustering. Default is 0.1.
    nclust : int, optional
        Number of clusters. Default is min(N/30, 100).
    tau : float, optional
        Protection against overcorrection. Default is 0.
    block_size : float, optional
        Proportion of cells to update in each block. Default is 0.05.
    max_iter_harmony : int, optional
        Maximum Harmony iterations. Default is 10.
    max_iter_kmeans : int, optional
        Maximum k-means iterations per Harmony iteration. Default is 20.
    epsilon_cluster : float, optional
        K-means convergence threshold. Default is 1e-5.
    epsilon_harmony : float, optional
        Harmony convergence threshold. Default is 1e-4.
    alpha : float, optional
        Alpha parameter for lambda estimation (when lamb=-1). Default is 0.2.
    verbose : bool, optional
        Print progress messages. Default is True.
    random_state : int, optional
        Random seed for reproducibility. Default is 0.
    device : str, optional
        Device to use ('cpu', 'cuda', 'mps'). Default is auto-detect.

    Returns
    -------
    Harmony
        Harmony object with corrected data in Z_corr attribute.
    """
    N = meta_data.shape[0]
    if data_mat.shape[1] != N:
        data_mat = data_mat.T

    assert data_mat.shape[1] == N, \
       "data_mat and meta_data do not have the same number of cells"

    if nclust is None:
        nclust = int(min(round(N / 30.0), 100))

    if isinstance(sigma, float) and nclust > 1:
        sigma = np.repeat(sigma, nclust)

    if isinstance(vars_use, str):
        vars_use = [vars_use]

    # Build concatenated batch indicator matrix (one-hot encoded across all vars_use)
    # plus per-variable one-hot matrices for correct/fast diversity penalty.
    Phi_list = []
    Pr_list = []
    phi_n = meta_data[vars_use].describe().loc['unique'].to_numpy().astype(int)

    for v in vars_use:
        Phi_v = pd.get_dummies(meta_data[v]).to_numpy().T.astype(np.float32)  # (Bv, N)
        Phi_list.append(Phi_v)
        Pr_list.append((Phi_v.sum(axis=1) / N).astype(np.float32))

    # Concatenated Phi for ridge regression design
    phi = np.concatenate(Phi_list, axis=0).astype(np.float32)

    # Theta handling - diversity penalty is per-variable
    if theta is None:
        theta_list = [np.repeat(2.0, Phi_v.shape[0]).astype(np.float32) for Phi_v in Phi_list]
    elif isinstance(theta, (float, int)):
        theta_list = [np.repeat(float(theta), Phi_v.shape[0]).astype(np.float32) for Phi_v in Phi_list]
    else:
        theta_arr = np.asarray(theta, dtype=np.float32)
        if theta_arr.ndim == 0:
            theta_list = [np.repeat(float(theta_arr), Phi_v.shape[0]).astype(np.float32) for Phi_v in Phi_list]
        elif len(theta_arr) == len(phi_n):
            # One theta per variable -> expand to categories within each variable
            theta_list = [np.repeat(float(theta_arr[i]), Phi_list[i].shape[0]).astype(np.float32)
                          for i in range(len(Phi_list))]
        else:
            # Assume concatenated per-category theta across all vars_use
            assert len(theta_arr) == int(np.sum(phi_n)), "each batch category must have a theta"
            theta_list = []
            off = 0
            for i in range(len(Phi_list)):
                Bv = Phi_list[i].shape[0]
                theta_list.append(theta_arr[off:off + Bv].astype(np.float32))
                off += Bv

    # Lambda handling (matches R package) - still applies to full concatenated design
    lambda_estimation = False
    if lamb is None:
        lamb = np.repeat([1] * len(phi_n), phi_n).astype(np.float32)
        lamb = np.insert(lamb, 0, 0).astype(np.float32)
    elif lamb == -1:
        lambda_estimation = True
        lamb = np.zeros(1, dtype=np.float32)
    elif isinstance(lamb, (float, int)):
        lamb = np.repeat([lamb] * len(phi_n), phi_n).astype(np.float32)
        lamb = np.insert(lamb, 0, 0).astype(np.float32)
    elif len(lamb) == len(phi_n):
        lamb = np.repeat([lamb], phi_n).astype(np.float32)
        lamb = np.insert(lamb, 0, 0).astype(np.float32)
    else:
        lamb = np.asarray(lamb, dtype=np.float32)
        if len(lamb) == np.sum(phi_n):
            lamb = np.insert(lamb, 0, 0).astype(np.float32)

    # Tau scaling per-variable (matches conceptual intent better than on concatenated mega-batch)
    if tau > 0:
        theta_list_tau = []
        for i, Phi_v in enumerate(Phi_list):
            N_b_v = Phi_v.sum(axis=1).astype(np.float32)
            theta_v = theta_list[i].astype(np.float32)
            theta_v = theta_v * (1 - np.exp(-(N_b_v / (nclust * tau)) ** 2))
            theta_list_tau.append(theta_v.astype(np.float32))
        theta_list = theta_list_tau

    # For backward-compatible logging / downstream usage keep a concatenated theta as well
    theta = np.concatenate(theta_list, axis=0).astype(np.float32)

    # Number of items in each category (concatenated)
    N_b = phi.sum(axis=1)
    Pr_b = (N_b / N).astype(np.float32)

    # Get device
    device_obj = get_device(device)

    if verbose:
        logger.info(f"Running Harmony (PyTorch on {device_obj})")
        logger.info("  Parameters:")
        logger.info(f"    max_iter_harmony: {max_iter_harmony}")
        logger.info(f"    max_iter_kmeans: {max_iter_kmeans}")
        logger.info(f"    epsilon_cluster: {epsilon_cluster}")
        logger.info(f"    epsilon_harmony: {epsilon_harmony}")
        logger.info(f"    nclust: {nclust}")
        logger.info(f"    block_size: {block_size}")
        if lambda_estimation:
            logger.info(f"    lamb: dynamic (alpha={alpha})")
        else:
            logger.info(f"    lamb: {lamb[1:]}")
        logger.info(f"    theta: {theta}")
        logger.info(f"    sigma: {sigma[:5]}..." if len(sigma) > 5 else f"    sigma: {sigma}")
        logger.info(f"    verbose: {verbose}")
        logger.info(f"    random_state: {random_state}")
        logger.info(f"  Data: {data_mat.shape[0]} PCs Ã— {N} cells")
        logger.info(f"  Batch variables: {vars_use}")

    # Set random seeds
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # Optional: enable faster matmul on CUDA (TF32) with minimal numerical impact
    if device_obj.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Ensure data_mat is a proper numpy array
    if hasattr(data_mat, 'values'):
        data_mat = data_mat.values
    data_mat = np.asarray(data_mat, dtype=np.float32)

    ho = Harmony(
        data_mat, phi, Pr_b, sigma.astype(np.float32),
        theta, lamb, alpha, lambda_estimation,
        max_iter_harmony, max_iter_kmeans,
        epsilon_cluster, epsilon_harmony, nclust, block_size, verbose,
        random_state, device_obj,
        Phi_list=Phi_list, Pr_list=Pr_list, theta_list=theta_list
    )

    return ho


class Harmony:
    """Harmony class for batch effect correction using PyTorch.

    Supports CPU and GPU acceleration.
    """

    def __init__(
            self, Z, Phi, Pr_b, sigma, theta, lamb, alpha, lambda_estimation,
            max_iter_harmony, max_iter_kmeans,
            epsilon_kmeans, epsilon_harmony, K, block_size, verbose,
            random_state, device,
            Phi_list=None, Pr_list=None, theta_list=None
    ):
        self.device = device

        # Convert to PyTorch tensors on device
        # Store with underscore prefix internally, expose as properties returning NumPy arrays
        self._Z_corr = torch.tensor(Z, dtype=torch.float32, device=device)
        self._Z_orig = torch.tensor(Z, dtype=torch.float32, device=device)

        # Simple L2 normalization (safe: avoid NaNs if a column norm is 0)
        _norm = torch.linalg.norm(self._Z_orig, ord=2, dim=0).clamp_min(1e-8)
        self._Z_cos = self._Z_orig / _norm

        # Batch indicators (concatenated design for ridge)
        self._Phi = torch.tensor(Phi, dtype=torch.float32, device=device)
        self._Pr_b = torch.tensor(Pr_b, dtype=torch.float32, device=device)

        # Per-variable Phi for diversity penalty (fast gather/scatter path always valid per-variable)
        self._Phi_list = []
        self._Pr_list = []
        self._theta_list = []
        self._phi_idx_list = []
        if Phi_list is not None:
            for i, Phi_v in enumerate(Phi_list):
                Phi_v_t = torch.tensor(Phi_v, dtype=torch.float32, device=device)  # (Bv, N)
                self._Phi_list.append(Phi_v_t)
                if Pr_list is None:
                    Pr_v = (Phi_v_t.sum(dim=1) / float(Phi_v_t.shape[1])).to(torch.float32)
                else:
                    Pr_v = torch.tensor(Pr_list[i], dtype=torch.float32, device=device)
                self._Pr_list.append(Pr_v)

                if theta_list is None:
                    theta_v = torch.full((Phi_v_t.shape[0],), 2.0, dtype=torch.float32, device=device)
                else:
                    theta_v = torch.tensor(theta_list[i], dtype=torch.float32, device=device)
                self._theta_list.append(theta_v)

                with torch.no_grad():
                    # Each Phi_v is one-hot (per variable), so argmax is safe and enables gather/scatter.
                    self._phi_idx_list.append(torch.argmax(Phi_v_t, dim=0).to(torch.long))

        # Backward compatibility: only used in objective/cross-entropy legacy path if needed
        self._phi_idx = None
        with torch.no_grad():
            phi_col_sum = self._Phi.sum(dim=0)
            if torch.all(phi_col_sum == 1):
                self._phi_idx = torch.argmax(self._Phi, dim=0).to(torch.long)

        self.N = self._Z_corr.shape[1]
        self.B = Phi.shape[0]
        self.d = self._Z_corr.shape[0]

        # Build batch index for fast ridge correction
        self._batch_index = []
        for b in range(self.B):
            idx = torch.where(self._Phi[b, :] > 0)[0]
            self._batch_index.append(idx)

        # Create Phi_moe with intercept
        ones = torch.ones(1, self.N, dtype=torch.float32, device=device)
        self._Phi_moe = torch.cat([ones, self._Phi], dim=0)

        # Cache per-batch Z slices (Z_orig is constant) to reduce indexing overhead in ridge step
        self._Zb_list = [self._Z_orig[:, idx].contiguous() for idx in self._batch_index]

        self.window_size = 3
        self.epsilon_kmeans = epsilon_kmeans
        self.epsilon_harmony = epsilon_harmony

        self._lamb = torch.tensor(lamb, dtype=torch.float32, device=device)
        self.alpha = alpha
        self.lambda_estimation = lambda_estimation
        self._sigma = torch.tensor(sigma, dtype=torch.float32, device=device)
        self.block_size = block_size
        self.K = K
        self.max_iter_harmony = max_iter_harmony
        self.max_iter_kmeans = max_iter_kmeans
        self.verbose = verbose
        self._theta = torch.tensor(theta, dtype=torch.float32, device=device)

        self.objective_harmony = []
        self.objective_kmeans = []
        self.objective_kmeans_dist = []
        self.objective_kmeans_entropy = []
        self.objective_kmeans_cross = []
        self.kmeans_rounds = []

        self.allocate_buffers()
        self.init_cluster(random_state)
        self.harmonize(self.max_iter_harmony, self.verbose)

    # =========================================================================
    # Properties - Return NumPy arrays for inspection and tutorials
    # =========================================================================

    @property
    def Z_corr(self):
        """Corrected embedding matrix (N x d). Batch effects removed."""
        return self._Z_corr.cpu().numpy().T

    @property
    def Z_orig(self):
        """Original embedding matrix (N x d). Input data before correction."""
        return self._Z_orig.cpu().numpy().T

    @property
    def Z_cos(self):
        """L2-normalized embedding matrix (N x d). Used for clustering."""
        return self._Z_cos.cpu().numpy().T

    @property
    def R(self):
        """Soft cluster assignment matrix (N x K). R[i,k] = P(cell i in cluster k)."""
        return self._R.cpu().numpy().T

    @property
    def Y(self):
        """Cluster centroids matrix (d x K). Columns are cluster centers."""
        return self._Y.cpu().numpy()

    @property
    def O(self):
        """Observed batch-cluster counts (K x B). O[k,b] = sum of R[k,:] for batch b."""
        return self._O.cpu().numpy()

    @property
    def E(self):
        """Expected batch-cluster counts (K x B). E[k,b] = cluster_size[k] * batch_proportion[b]."""
        return self._E.cpu().numpy()

    @property
    def Phi(self):
        """Batch indicator matrix (N x B). One-hot encoding of batch membership."""
        return self._Phi.cpu().numpy().T

    @property
    def Phi_moe(self):
        """Batch indicator with intercept (N x (B+1)). First column is all ones."""
        return self._Phi_moe.cpu().numpy().T

    @property
    def Pr_b(self):
        """Batch proportions (B,). Pr_b[b] = cells in batch b / total cells."""
        return self._Pr_b.cpu().numpy()

    @property
    def theta(self):
        """Diversity penalty parameters (B,). Higher = more mixing encouraged."""
        return self._theta.cpu().numpy()

    @property
    def sigma(self):
        """Clustering bandwidth parameters (K,). Soft assignment kernel width."""
        return self._sigma.cpu().numpy()

    @property
    def lamb(self):
        """Ridge regression penalty ((B+1),). Regularization for batch correction."""
        return self._lamb.cpu().numpy()

    @property
    def objectives(self):
        """List of objective values for compatibility with evaluator."""
        return self.objective_harmony

    def result(self):
        """Return corrected data as NumPy array."""
        return self._Z_corr.cpu().numpy().T

    def allocate_buffers(self):
        self._scale_dist = torch.zeros((self.K, self.N), dtype=torch.float32, device=self.device)
        self._dist_mat = torch.zeros((self.K, self.N), dtype=torch.float32, device=self.device)
        self._O = torch.zeros((self.K, self.B), dtype=torch.float32, device=self.device)
        self._E = torch.zeros((self.K, self.B), dtype=torch.float32, device=self.device)
        self._W = torch.zeros((self.B + 1, self.d), dtype=torch.float32, device=self.device)
        self._R = torch.zeros((self.K, self.N), dtype=torch.float32, device=self.device)
        self._Y = torch.zeros((self.d, self.K), dtype=torch.float32, device=self.device)
        # Persistent correction buffer to avoid reallocations in moe_correct_ridge()
        self._Z_corr = torch.empty_like(self._Z_orig)

        # Previous assignments for momentum in update_R()
        self._R_prev = torch.zeros((self.K, self.N), dtype=torch.float32, device=self.device)

        # Reusable penalty buffer (allocated on-demand in update_R())
        self._log_penalty = None

        # Track cluster sizes for stable/cheap E recomputation in update_R()
        self._N_k = torch.zeros((self.K,), dtype=torch.float32, device=self.device)

        # Temp buffer for fast scatter_add updates of O in one-hot path (concatenated)
        self._O_tmp = torch.zeros((self.K, self.B), dtype=torch.float32, device=self.device)

        # Per-variable O for diversity penalty
        # NOTE: E is computed on-demand for only the categories present in a block.
        self._O_list = []
        self._O_tmp_list = []
        for Phi_v in self._Phi_list:
            Bv = Phi_v.shape[0]
            self._O_list.append(torch.zeros((self.K, Bv), dtype=torch.float32, device=self.device))
            self._O_tmp_list.append(torch.zeros((self.K, Bv), dtype=torch.float32, device=self.device))

        # Offsets into the concatenated Phi design for each variable
        # (so we can slice W_batch per variable and gather instead of dense Phi matmuls)
        self._phi_offsets = []
        off = 0
        for Phi_v in self._Phi_list:
            self._phi_offsets.append(off)
            off += int(Phi_v.shape[0])

        # Preallocate concatenated O across all variables/categories to avoid torch.cat each iteration
        # Shape: (K, B_total) where B_total == sum(Bv) across variables (== self.B)
        self._O = torch.zeros((self.K, self.B), dtype=torch.float32, device=self.device)

        # Small helper buffers/tensors reused across hot loops
        self._k_arange = torch.arange(self.K, device=self.device).view(-1, 1)

        # Precompute invariant per-dimension norm of original embedding (used for damping)
        eps = 1e-8
        self._z_norm_d = torch.linalg.norm(self._Z_orig, dim=1).clamp_min(eps)

        # Buffers for ridge correction (allocated lazily in moe_correct_ridge)
        self._cov_buf = None
        self._G_buf = None

    def init_cluster(self, random_state):
        logger.info("Computing initial centroids with sklearn.KMeans...")
        # KMeans needs CPU numpy array
        Z_cos_np = self._Z_cos.cpu().numpy()
        model = KMeans(n_clusters=self.K, init='k-means++',
                       n_init=1, max_iter=25, random_state=random_state)
        model.fit(Z_cos_np.T)
        self._Y = torch.tensor(model.cluster_centers_.T, dtype=torch.float32, device=self.device)
        logger.info("KMeans initialization complete.")

        # Normalize centroids
        self._Y = self._Y / torch.linalg.norm(self._Y, ord=2, dim=0)

        # Compute distance matrix: dist = 2 * (1 - Y.T @ Z_cos)
        self._dist_mat = 2 * (1 - self._Y.T @ self._Z_cos)

        # Compute R
        self._R = -self._dist_mat / self._sigma[:, None]
        self._R = torch.exp(self._R)
        self._R = self._R / self._R.sum(dim=0)

        # Batch diversity statistics (concatenated + per-variable)
        self._N_k = self._R.sum(dim=1)
        self._E = self._N_k[:, None] * self._Pr_b[None, :]
        self._O = self._R @ self._Phi.T

        # Also initialize per-variable O and keep concatenated O slices in sync
        for i, Phi_v in enumerate(self._Phi_list):
            Oi = self._R @ Phi_v.T
            self._O_list[i] = Oi
            off = self._phi_offsets[i]
            Bv = int(Phi_v.shape[0])
            self._O[:, off:off + Bv] = Oi

        self.compute_objective()
        self.objective_harmony.append(self.objective_kmeans[-1])

    def compute_objective(self):
        # Normalization constant
        norm_const = 2000.0 / self.N

        # Keep as tensors to avoid repeated device synchronization via .item()
        kmeans_error = torch.sum(self._R * self._dist_mat)

        entropy = torch.sum(safe_entropy_torch(self._R) * self._sigma[:, None])

        # Cross entropy (R package formula) with numerical stability
        R_sigma = self._R * self._sigma[:, None]
        # Clamp to avoid log(0) or division by zero
        O_clamped = torch.clamp(self._O, min=1e-8)
        E_clamped = torch.clamp(self._E, min=1e-8)
        ratio = (O_clamped + E_clamped) / E_clamped
        theta_log = self._theta.unsqueeze(0).expand(self.K, -1) * torch.log(ratio)  # (K, B)

        # Avoid (K¡ÁB)@(B¡ÁN) when Phi is one-hot: gather per-cell penalty instead
        if self._phi_idx is not None:
            theta_per_cell = theta_log[:, self._phi_idx]  # (K, N)
        else:
            theta_per_cell = theta_log @ self._Phi  # (K, N)

        cross_entropy = torch.sum(R_sigma * theta_per_cell)

        obj = (kmeans_error + entropy + cross_entropy) * norm_const

        # Store with a single .item() to avoid repeated syncs
        self.objective_kmeans.append(obj.item())
        self.objective_kmeans_dist.append((kmeans_error * norm_const).item())
        self.objective_kmeans_entropy.append((entropy * norm_const).item())
        self.objective_kmeans_cross.append((cross_entropy * norm_const).item())

    def harmonize(self, iter_harmony=10, verbose=True):
        converged = False
        with torch.no_grad():
            for i in range(1, iter_harmony + 1):
                # Track current Harmony iteration for annealed damping
                self._iter_harmony = i
                self._iter_harmony_max = iter_harmony

                if verbose:
                    logger.info(f"Iteration {i} of {iter_harmony}")

                self.cluster()
                self.moe_correct_ridge()

                converged = self.check_convergence(1)
                if converged:
                    if verbose:
                        logger.info(f"Converged after {i} iteration{'s' if i > 1 else ''}")
                    break

        if verbose and not converged:
            logger.info("Stopped before convergence")

    def cluster(self):
        rounds = 0

        # Reuse update order across kmeans inner iterations for lower overhead and more stability
        self._update_order = torch.randperm(self.N, device=self.device)

        # Ensure contiguous once per kmeans loop (avoid repeated contiguity conversions)
        Z_cos = self._Z_cos.contiguous()

        # Objective evaluation cadence (reduce overhead)
        obj_every = 2

        for i in range(self.max_iter_kmeans):
            # Track centroid shift for cheap early stopping
            Y_old = self._Y.clone()

            # Update Y
            self._Y = (Z_cos @ self._R.T).contiguous()
            self._Y = (self._Y / torch.linalg.norm(self._Y, ord=2, dim=0).clamp_min(1e-8)).contiguous()

            # Update distance matrix
            self._dist_mat = 2 * (1 - self._Y.T @ Z_cos)

            # Update R
            self.update_R()

            # Cheap assignment-drift early stop (use already-maintained _R_prev)
            if (i % obj_every == 0) or (i == self.max_iter_kmeans - 1):
                delta_R = torch.mean(torch.abs(self._R - self._R_prev))
                if delta_R.item() < (0.1 * self.epsilon_kmeans):
                    rounds = i + 1
                    break

            # Compute objective less frequently; always compute on last iteration
            if (i % obj_every == 0) or (i == self.max_iter_kmeans - 1):
                self.compute_objective()

            # Cheap centroid-shift early stop (works even when objective isn't computed this iter)
            # Compute less frequently to reduce norm overhead
            if (i % obj_every == 0) or (i == self.max_iter_kmeans - 1):
                deltaY = torch.linalg.norm(self._Y - Y_old) / (torch.linalg.norm(Y_old).clamp_min(1e-8))
                if deltaY.item() < self.epsilon_kmeans:
                    rounds = i + 1
                    break

            if i > self.window_size and len(self.objective_kmeans) > self.window_size:
                if self.check_convergence(0):
                    rounds = i + 1
                    break
            rounds = i + 1

        self.kmeans_rounds.append(rounds)
        # Ensure harmony objective has something to append
        if len(self.objective_kmeans) > 0:
            self.objective_harmony.append(self.objective_kmeans[-1])

    def update_R(self):
        # Compute scaled distances in log-space: log_scale = -dist/sigma
        # (we'll normalize per-cell with softmax later)
        self._scale_dist = -self._dist_mat / self._sigma[:, None]

        # Reuse shuffled update order if provided by cluster(); fall back if needed
        update_order = getattr(self, "_update_order", None)
        if update_order is None or update_order.numel() != self.N:
            update_order = torch.randperm(self.N, device=self.device)
            self._update_order = update_order

        # Anneal diversity pressure (mixing early, protect biology late)
        it = getattr(self, "_iter_harmony", 1)
        it_max = max(1, getattr(self, "_iter_harmony_max", 1))
        progress = float(it - 1) / float(max(1, it_max - 1))
        beta_start, beta_end = 1.2, 0.8
        pmax_start, pmax_end = 10.0, 6.0
        beta = beta_start + (beta_end - beta_start) * progress
        pmax = pmax_start + (pmax_end - pmax_start) * progress

        # Adaptive block sizing (avoid numpy in hot path)
        block_size_start = self.block_size if self.block_size >= 0.2 else 0.2
        block_size_eff = block_size_start + (self.block_size - block_size_start) * progress
        min_bs = 1.0 / float(max(1, self.N))
        if block_size_eff < min_bs:
            block_size_eff = min_bs
        if block_size_eff > 1.0:
            block_size_eff = 1.0

        # Process in blocks
        n_blocks = int((1.0 / block_size_eff) + (0 if (1.0 / block_size_eff).is_integer() else 1))
        cells_per_block = max(1, int(self.N * block_size_eff))

        K = self.K

        for blk in range(n_blocks):
            idx_min = blk * cells_per_block
            idx_max = self.N if blk == n_blocks - 1 else (blk + 1) * cells_per_block

            cols = update_order[idx_min:idx_max]

            R_block = self._R[:, cols]
            log_scale_block = self._scale_dist[:, cols]

            # ---- Remove cells from statistics (N_k and per-variable O/E)
            self._N_k.sub_(R_block.sum(dim=1))

            for vi in range(len(self._Phi_list)):
                phi_idx_v_block = self._phi_idx_list[vi][cols]
                O_tmp_v = self._O_tmp_list[vi]
                O_tmp_v.zero_()
                O_tmp_v.scatter_add_(1, phi_idx_v_block[None, :].expand(K, -1), R_block)
                self._O_list[vi].sub_(O_tmp_v)
                off = self._phi_offsets[vi]
                Bv = int(self._Phi_list[vi].shape[0])
                self._O[:, off:off + Bv].sub_(O_tmp_v)

            # ---- Recompute R for this block in log-space for stability
            # Diversity penalty computed with full-category gather (vectorized; smooth across blocks).
            nb = cols.numel()
            if (self._log_penalty is None) or (self._log_penalty.shape[1] < nb) or (self._log_penalty.shape[0] != K):
                self._log_penalty = torch.empty((K, nb), dtype=torch.float32, device=self.device)
            log_penalty = self._log_penalty[:, :nb]
            log_penalty.zero_()

            # Stabilization settings (protect against rare categories / tiny expected counts)
            eps = 1e-8
            pc = 0.1
            ratio_min, ratio_max = 1e-3, 1e3

            for vi in range(len(self._Phi_list)):
                O_v = self._O_list[vi]                 # (K, Bv)
                theta_v = self._theta_list[vi]         # (Bv,)
                phi_idx_v_block = self._phi_idx_list[vi][cols]  # (nb,)

                # Full expected counts for this variable: E = N_k * Pr
                E_v = self._N_k[:, None] * self._Pr_list[vi][None, :]           # (K, Bv)

                ratio = (O_v + pc) / (E_v + pc)
                ratio = torch.clamp(ratio, min=ratio_min, max=ratio_max)

                # penalty = -theta * log(ratio)
                pen_v = -torch.log(ratio) * theta_v.unsqueeze(0)                # (K, Bv)

                # Gather per-cell category penalties directly (no unique/searchsorted)
                log_penalty.add_(pen_v[:, phi_idx_v_block])                     # (K, nb)

            log_penalty.mul_(beta).clamp_(min=-pmax, max=pmax)

            log_R = log_scale_block + log_penalty
            R_block_new = torch.softmax(log_R, dim=0)

            # ---- Assignment momentum to reduce oscillations (annealed over Harmony iterations)
            mu_start, mu_end = 0.2, 0.0
            mu = mu_start + (mu_end - mu_start) * progress
            if mu > 0:
                R_block_new = (1.0 - mu) * R_block_new + mu * R_block
                R_block_new = R_block_new / R_block_new.sum(dim=0).clamp_min(eps)

            # ---- Put cells back
            self._N_k.add_(R_block_new.sum(dim=1))

            for vi in range(len(self._Phi_list)):
                phi_idx_v_block = self._phi_idx_list[vi][cols]
                O_tmp_v = self._O_tmp_list[vi]
                O_tmp_v.zero_()
                O_tmp_v.scatter_add_(1, phi_idx_v_block[None, :].expand(K, -1), R_block_new)
                self._O_list[vi].add_(O_tmp_v)
                off = self._phi_offsets[vi]
                Bv = int(self._Phi_list[vi].shape[0])
                self._O[:, off:off + Bv].add_(O_tmp_v)

            # Write back in-place (+ keep previous assignments for next iteration momentum)
            self._R_prev[:, cols] = self._R[:, cols]
            self._R[:, cols] = R_block_new

    def check_convergence(self, i_type):
        if i_type == 0:
            if len(self.objective_kmeans) <= self.window_size + 1:
                return False

            w = self.window_size
            obj_old = sum(self.objective_kmeans[-w-1:-1])
            obj_new = sum(self.objective_kmeans[-w:])
            return abs(obj_old - obj_new) / abs(obj_old) < self.epsilon_kmeans

        if i_type == 1:
            if len(self.objective_harmony) < 2:
                return False

            obj_old = self.objective_harmony[-2]
            obj_new = self.objective_harmony[-1]
            return (obj_old - obj_new) / abs(obj_old) < self.epsilon_harmony

        return True

    def moe_correct_ridge(self):
        """Ridge regression correction for batch effects.

        Rewritten to use sufficient statistics for one-hot batch design:
        - Avoids O(N) batched bmm for cov/G construction.
        - Builds cov analytically from N_k and O (cluster-by-batch soft counts).
        - Builds G from Z @ R^T and per-batch subset matmuls using self._batch_index.
        - Skips tiny clusters for stability/time.
        - Applies corrections in-place with per-PC damping.
        """
        # Reset corrected embedding in-place (avoid reallocation)
        self._Z_corr.copy_(self._Z_orig)

        Z_orig = self._Z_orig.contiguous()  # (d, N)
        Phi = self._Phi.contiguous()       # (B, N)
        R = self._R.contiguous()           # (K, N)

        Bp1 = self.B + 1
        K = self.K
        B = self.B

        # For lambda estimation use E from concatenated batch proportions
        self._E = self._N_k[:, None] * self._Pr_b[None, :]

        if self.lambda_estimation:
            lamb_all = torch.zeros((K, Bp1), dtype=torch.float32, device=self.device)
            lamb_all[:, 1:] = self.alpha * self._E
        else:
            lamb_all = self._lamb.view(1, -1).expand(K, -1)

        # Chunk clusters to limit peak memory from batched solve / accumulations
        chunk_k = 16

        # Correction application: block cells for cache friendliness
        cell_chunk = 4096

        # Annealed per-PC damping target
        it = getattr(self, "_iter_harmony", 1)
        it_max = max(1, getattr(self, "_iter_harmony_max", 1))
        progress = float(it - 1) / float(max(1, it_max - 1))
        target_start, target_end = 0.35, 0.12
        target = target_start + (target_end - target_start) * progress

        eps = 1e-8
        z_norm_d = getattr(self, "_z_norm_d", None)
        if z_norm_d is None:
            z_norm_d = torch.linalg.norm(Z_orig, dim=1).clamp_min(eps)  # (d,)
            self._z_norm_d = z_norm_d

        tiny = 1e-3

        for k0 in range(0, K, chunk_k):
            k1 = min(K, k0 + chunk_k)
            kc = k1 - k0

            N_k_chunk = self._N_k[k0:k1].contiguous()          # (kc,)
            O_chunk = self._O[k0:k1, :].contiguous()           # (kc, B)
            R_chunk = R[k0:k1, :].contiguous()                 # (kc, N)

            # Skip clusters with negligible mass
            active = N_k_chunk > tiny
            if not torch.any(active):
                continue

            # Build cov analytically: (kc, B+1, B+1) using reusable buffers
            if (getattr(self, "_cov_buf", None) is None) or (self._cov_buf.shape[0] < kc) or (self._cov_buf.shape[1] != Bp1):
                self._cov_buf = torch.empty((max(chunk_k, kc), Bp1, Bp1), dtype=torch.float32, device=self.device)
            cov = self._cov_buf[:kc]
            cov.zero_()
            cov[:, 0, 0] = N_k_chunk
            cov[:, 0, 1:] = O_chunk
            cov[:, 1:, 0] = O_chunk
            cov[:, 1:, 1:] = torch.diag_embed(O_chunk)
            cov = cov + torch.diag_embed(lamb_all[k0:k1, :])

            # Build G using sufficient stats with reusable buffers:
            # G0 = Z @ R^T, and per-batch Gb = Z[:,idx_b] @ R[:,idx_b]^T
            if (getattr(self, "_G_buf", None) is None) or (self._G_buf.shape[0] < kc) or (self._G_buf.shape[1] != Bp1) or (self._G_buf.shape[2] != self.d):
                self._G_buf = torch.empty((max(chunk_k, kc), Bp1, self.d), dtype=torch.float32, device=self.device)
            G = self._G_buf[:kc]
            G.zero_()

            G0 = (Z_orig @ R_chunk.T).T.contiguous()  # (kc, d)
            G[:, 0, :] = G0

            for b in range(B):
                idx_b = self._batch_index[b]
                if idx_b.numel() == 0:
                    continue
                Zb = self._Zb_list[b]                  # (d, nb)
                Rb = R_chunk[:, idx_b]                 # (kc, nb)
                Gb = (Rb @ Zb.T).contiguous()          # (kc, d)
                G[:, b + 1, :] = Gb

            # Solve cov @ W_T = G  (both in (kc, B+1, d))
            # Guard cholesky for numerical stability (ridge should ensure SPD)
            L = torch.linalg.cholesky(cov)
            W_T = torch.cholesky_solve(G, L)                  # (kc, B+1, d)
            W = W_T.transpose(1, 2).contiguous()              # (kc, d, B+1)
            W[:, :, 0] = 0  # protect intercept

            # Zero-out inactive clusters (avoid NaN propagation / wasted compute)
            if torch.any(~active):
                W[~active, :, :] = 0

            # Apply corrections in-place:
            # W_batch: (kc, d, B) where B is the concatenated category count across variables.
            # Instead of dense (W_batch @ Phi_blk), gather one category per variable per cell.
            W_batch = W[:, :, 1:].contiguous()

            # Reusable buffer for summing gathered coefficients across variables
            C_buf = torch.empty((kc, self.d, cell_chunk), dtype=torch.float32, device=self.device)

            for c0 in range(0, self.N, cell_chunk):
                c1 = min(self.N, c0 + cell_chunk)
                cols = slice(c0, c1)
                nb = c1 - c0

                R_blk = R_chunk[:, cols].contiguous()     # (kc, nb)

                # Build C by summing gathered coefficients from each variable in-place:
                # C: (kc, d, nb)
                C = C_buf[:, :, :nb]
                C.zero_()

                for vi, Phi_v in enumerate(self._Phi_list):
                    off = self._phi_offsets[vi]
                    Bv = int(Phi_v.shape[0])

                    Wv = W_batch[:, :, off:off + Bv]              # (kc, d, Bv)
                    phi_idx_v = self._phi_idx_list[vi][cols]      # (nb,)
                    gathered = Wv[:, :, phi_idx_v]                # (kc, d, nb)
                    C.add_(gathered)

                delta_blk = (C * R_blk.unsqueeze(1)).sum(dim=0)                    # (d, nb)

                delta_norm_d = torch.linalg.norm(delta_blk, dim=1).clamp_min(eps)  # (d,)
                eta_d = torch.minimum(torch.ones_like(delta_norm_d), (target * z_norm_d) / delta_norm_d)
                self._Z_corr[:, cols].sub_(eta_d.unsqueeze(1) * delta_blk)

        # Update Z_cos (safe normalization)
        _norm = torch.linalg.norm(self._Z_corr, ord=2, dim=0).clamp_min(1e-8)
        self._Z_cos = (self._Z_corr / _norm).contiguous()


def safe_entropy_torch(x):
    """Compute x * log(x), returning 0 where x is 0 or negative."""
    result = x * torch.log(x)
    result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
    return result


def harmony_pow_torch(A, T):
    """Element-wise power with different exponents per column (vectorized)."""
    return torch.pow(A, T.unsqueeze(0))


def find_lambda_torch(alpha, cluster_E, device):
    """Compute dynamic lambda based on cluster expected counts."""
    lamb = torch.zeros(len(cluster_E) + 1, dtype=torch.float32, device=device)
    lamb[1:] = cluster_E * alpha
    return lamb
