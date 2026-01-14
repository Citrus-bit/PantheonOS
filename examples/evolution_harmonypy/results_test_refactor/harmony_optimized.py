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

    # Create batch indicator matrix (one-hot encoded)
    phi = pd.get_dummies(meta_data[vars_use]).to_numpy().T.astype(np.float32)
    phi_n = meta_data[vars_use].describe().loc['unique'].to_numpy().astype(int)

    # Theta handling - default is 2 (matches R package)
    if theta is None:
        theta = np.repeat([2] * len(phi_n), phi_n).astype(np.float32)
    elif isinstance(theta, (float, int)):
        theta = np.repeat([theta] * len(phi_n), phi_n).astype(np.float32)
    elif len(theta) == len(phi_n):
        theta = np.repeat([theta], phi_n).astype(np.float32)
    else:
        theta = np.asarray(theta, dtype=np.float32)

    assert len(theta) == np.sum(phi_n), \
        "each batch variable must have a theta"

    # Lambda handling (matches R package)
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

    # Number of items in each category
    N_b = phi.sum(axis=1)
    Pr_b = (N_b / N).astype(np.float32)

    if tau > 0:
        theta = theta * (1 - np.exp(-(N_b / (nclust * tau)) ** 2))

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
        logger.info(f"  Data: {data_mat.shape[0]} PCs × {N} cells")
        logger.info(f"  Batch variables: {vars_use}")

    # Set random seeds
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # Ensure data_mat is a proper numpy array
    if hasattr(data_mat, 'values'):
        data_mat = data_mat.values
    data_mat = np.asarray(data_mat, dtype=np.float32)

    ho = Harmony(
        data_mat, phi, Pr_b, sigma.astype(np.float32),
        theta, lamb, alpha, lambda_estimation,
        max_iter_harmony, max_iter_kmeans,
        epsilon_cluster, epsilon_harmony, nclust, block_size, verbose,
        random_state, device_obj
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
            random_state, device
    ):
        self.device = device

        # ---------------------------------------------------------------------
        # Algorithmic knobs (no public API changes)
        # ---------------------------------------------------------------------
        # Full-batch damped responsibility update (mirror-descent style)
        self._r_update_eta = 0.7  # damping factor in (0,1]
        # Optional temperature annealing for affinity term; decays to 1.0
        self._temp_start = 1.5
        self._temp_end = 1.0
        self._temp = self._temp_start

        # Proposal A: KL/OT-style batch-mixing penalty via IPFP/Sinkhorn-like scaling
        # Number of IPFP steps per update_R() call (small fixed number for speed/stability)
        self._ipfp_iters = 4
        # Strength of batch mixing scaling (acts like a step-size on log scaling)
        self._mix_eta = 0.6

        # Proposal B: proximal/trust-region correction via adaptive lambda multiplier
        self._lambda_mult = 1.0
        self._lambda_mult_min = 0.25
        self._lambda_mult_max = 16.0
        # Target relative correction magnitude per outer iteration (||Δ||_F / ||Z||_F)
        self._target_corr = 0.12
        self._corr_band = 0.04
        self._lambda_mult_up = 1.25
        self._lambda_mult_down = 0.90
        # Mixing proxy gate: only relax lambda (allow more correction) if mixing is poor
        self._mix_poor_thresh = 0.02

        # Semi-hard correction after warm-up
        self._corr_warmup_iters = 2   # keep fully soft correction for first N Harmony iters
        self._corr_top_m = 2          # keep top-m clusters per cell for correction (m=1 or 2 typical)
        self._harmony_iter = 0        # current outer iteration counter (1-indexed in harmonize)

        # Convert to PyTorch tensors on device
        # Store with underscore prefix internally, expose as properties returning NumPy arrays
        self._Z_corr = torch.tensor(Z, dtype=torch.float32, device=device)
        self._Z_orig = torch.tensor(Z, dtype=torch.float32, device=device)

        # Simple L2 normalization (stable)
        self._Z_cos = self._Z_orig / (torch.linalg.norm(self._Z_orig, ord=2, dim=0) + 1e-8)

        # Batch indicators
        self._Phi = torch.tensor(Phi, dtype=torch.float32, device=device)
        self._Pr_b = torch.tensor(Pr_b, dtype=torch.float32, device=device)

        self.N = self._Z_corr.shape[1]
        self.B = Phi.shape[0]
        self.d = self._Z_corr.shape[0]

        # Cache batch id per cell (Phi is one-hot by construction)
        self._batch_id = torch.argmax(self._Phi, dim=0).to(torch.long)

        # Cache Phi^T for reuse in matmuls (contiguous helps GPU kernels)
        self._Phi_T = self._Phi.T.contiguous()

        # Create Phi_moe with intercept
        ones = torch.ones(1, self.N, dtype=torch.float32, device=device)
        self._Phi_moe = torch.cat([ones, self._Phi], dim=0)

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

        # Persistent IPFP/Sinkhorn scaling buffer for warm-starting update_R()
        self._log_s = torch.zeros((self.K, self.B), dtype=torch.float32, device=self.device)

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

        # Batch diversity statistics
        self._E = torch.outer(self._R.sum(dim=1), self._Pr_b)
        self._O = self._R @ self._Phi.T

        self.compute_objective()
        self.objective_harmony.append(self.objective_kmeans[-1])

    def compute_objective(self):
        # Normalization constant
        norm_const = 2000.0 / self.N

        # K-means error
        kmeans_error = torch.sum(self._R * self._dist_mat).item()

        # Entropy
        _entropy = torch.sum(safe_entropy_torch(self._R) * self._sigma[:, None]).item()

        # Proposal A: KL-style batch-mixing penalty (cluster batch distribution vs global Pr_b)
        # π_{k,b} = O_{k,b} / sum_b O_{k,b}
        eps = 1e-8
        O = torch.clamp(self._O, min=eps)
        O_row = torch.clamp(O.sum(dim=1, keepdim=True), min=eps)
        pi = O / O_row  # (K x B)
        Pr = torch.clamp(self._Pr_b.unsqueeze(0), min=eps)  # (1 x B)

        # KL(pi || Pr) = sum_b pi * log(pi/Pr); weighted by theta per batch (smoother control)
        kl = (pi * (torch.log(pi) - torch.log(Pr))).sum(dim=1)  # (K,)
        _mix_kl = (kl * torch.mean(self._theta)).sum().item()

        # Store with normalization constant
        self.objective_kmeans.append((kmeans_error + _entropy + _mix_kl) * norm_const)
        self.objective_kmeans_dist.append(kmeans_error * norm_const)
        self.objective_kmeans_entropy.append(_entropy * norm_const)
        self.objective_kmeans_cross.append(_mix_kl * norm_const)

    def harmonize(self, iter_harmony=10, verbose=True):
        converged = False
        for i in range(1, iter_harmony + 1):
            self._harmony_iter = i

            # Temperature annealing for affinity term in update_R
            if iter_harmony > 1:
                frac = (i - 1) / (iter_harmony - 1)
            else:
                frac = 1.0
            self._temp = self._temp_start + frac * (self._temp_end - self._temp_start)

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
        for i in range(self.max_iter_kmeans):
            # Update Y
            self._Y = self._Z_cos @ self._R.T
            self._Y = self._Y / torch.linalg.norm(self._Y, ord=2, dim=0)

            # Update distance matrix
            self._dist_mat = 2 * (1 - self._Y.T @ self._Z_cos)

            # Update R
            self.update_R()

            # Compute objective and check convergence
            self.compute_objective()

            if i > self.window_size:
                if self.check_convergence(0):
                    rounds = i + 1
                    break
            rounds = i + 1

        self.kmeans_rounds.append(rounds)
        self.objective_harmony.append(self.objective_kmeans[-1])

    def update_R(self):
        """KL/OT-style responsibility update with IPFP/Sinkhorn-like batch scaling.

        Replaces the heuristic diversity reweighting with a principled iterative
        proportional fitting (IPFP) style update:
          R_{k,i} ∝ A_{k,i} * s_{k,b(i)}
        where s_{k,b} is iteratively updated to push cluster batch marginals
        toward global batch proportions.

        Uses logits + softmax for stability, then a few IPFP iterations.
        """
        eps = 1e-8

        # Base affinity logits: logA_{k,i} = -dist/(T*sigma_k)
        temp = float(getattr(self, "_temp", 1.0))
        denom = torch.clamp(self._sigma[:, None] * temp, min=eps)
        logA = -self._dist_mat / denom  # (K x N)

        # Initialize scaling in log-space (K x B)
        log_s = torch.zeros((self.K, self.B), dtype=torch.float32, device=self.device)

        ipfp_iters = int(getattr(self, "_ipfp_iters", 4))
        mix_eta = float(getattr(self, "_mix_eta", 0.6))
        mix_eta = max(0.0, min(1.0, mix_eta))

        # IPFP loop
        # At each step, build responsibilities with current scaling and update scaling
        # based on discrepancy between observed O and desired E (K x B).
        for _ in range(max(1, ipfp_iters)):
            # Per-cell batch scaling term: (K x N) via (K x B) @ (B x N)
            logD = log_s @ self._Phi
            logits = logA + logD
            R_tmp = torch.softmax(logits, dim=0)  # (K x N)

            # Observed and desired cluster-batch counts
            O = R_tmp @ self._Phi.T  # (K x B)
            cluster_mass = torch.clamp(R_tmp.sum(dim=1, keepdim=True), min=eps)  # (K x 1)
            E = cluster_mass * self._Pr_b.unsqueeze(0)  # (K x B)

            # IPFP scaling update in log-space:
            # log_s <- log_s + eta * theta_b * log(E/O)
            # (theta acts as per-batch penalty strength)
            ratio = torch.log(torch.clamp(E, min=eps)) - torch.log(torch.clamp(O, min=eps))
            log_s = log_s + mix_eta * ratio * self._theta.unsqueeze(0)

        # Final responsibilities from the last scaling
        logits = logA + (log_s @ self._Phi)
        R_new = torch.softmax(logits, dim=0)

        # Damped update
        eta = float(getattr(self, "_r_update_eta", 1.0))
        eta = max(0.0, min(1.0, eta))
        self._R = (1.0 - eta) * self._R + eta * R_new

        # Update stats buffers
        self._O = self._R @ self._Phi_T
        self._E = torch.outer(self._R.sum(dim=1), self._Pr_b)

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
            denom = max(abs(obj_old), 1e-12)

            # Require small absolute relative change and avoid declaring convergence on increases
            rel_change = abs(obj_new - obj_old) / denom
            if obj_new > obj_old:
                return rel_change < (self.epsilon_harmony * 0.1)
            return rel_change < self.epsilon_harmony

        return True

    def moe_correct_ridge(self):
        """Ridge regression correction for batch effects.

        Two-phase strategy:
          - Warm-up: fully soft correction using R
          - After warm-up: semi-hard correction using top-m responsibilities per cell
            (R used for clustering remains unchanged)

        Optimized implementation:
          - Uses batch_id + scatter_add + gather to remove Python loops over batches
          - Uses structured solve via Schur complement (no cholesky/solve needed)
          - Reuses Z_corr buffer (no clone per iteration)
        """
        eps = 1e-8

        # Reuse buffer rather than allocating each iteration
        self._Z_corr.copy_(self._Z_orig)

        # Build correction-specific responsibilities R_corr
        if getattr(self, "_harmony_iter", 0) <= getattr(self, "_corr_warmup_iters", 0):
            R_corr = self._R
        else:
            m = int(getattr(self, "_corr_top_m", 1))
            m = max(1, min(m, self.K))
            # Top-m per cell along cluster dimension
            vals, idx = torch.topk(self._R, k=m, dim=0, largest=True, sorted=False)  # (m x N)
            R_corr = torch.zeros_like(self._R)
            R_corr.scatter_(0, idx, vals)
            R_corr = R_corr / torch.clamp(R_corr.sum(dim=0, keepdim=True), min=eps)

        batch_id = self._batch_id  # (N,)
        # Precompute expanded indices for RHS scatter: (d x N)
        batch_idx_exp = batch_id.unsqueeze(0).expand(self.d, self.N)

        # For each cluster, solve structured ridge regression and apply correction in one fused op
        for k in range(self.K):
            # Compute lambda if estimating
            if self.lambda_estimation:
                lamb_vec = find_lambda_torch(self.alpha, self._E[k, :], self.device)
            else:
                lamb_vec = self._lamb

            # Proposal B: adaptive regularization multiplier (trust-region style)
            lamb_vec = lamb_vec * float(getattr(self, "_lambda_mult", 1.0))

            r_k = R_corr[k, :]  # (N,)

            # Per-batch sums sb[b] via scatter_add
            sb = torch.zeros((self.B,), dtype=torch.float32, device=self.device)
            sb.scatter_add_(0, batch_id, r_k)

            # Per-batch RHS rhsb[:, b] via scatter_add over batch dimension
            rhsb = torch.zeros((self.d, self.B), dtype=torch.float32, device=self.device)
            X = self._Z_orig * r_k.unsqueeze(0)  # (d x N)
            rhsb.scatter_add_(1, batch_idx_exp, X)

            # Intercept RHS is sum over batches (since Phi is one-hot)
            rhs0 = rhsb.sum(dim=1)  # (d,)
            s0 = sb.sum()           # scalar

            # Structured system:
            # [ a    u^T ] [beta0] = [rhs0]
            # [ u     D  ] [betab]   [rhsb]
            # where D is diagonal with D_b = sb[b] + lamb[1+b]
            a = s0 + lamb_vec[0]
            u = sb  # (B,)

            D = sb + lamb_vec[1:]
            invD = 1.0 / torch.clamp(D, min=eps)

            # Schur complement solve for all d at once
            t = rhsb * invD.unsqueeze(0)  # (d x B)
            u_inv_rhs = (u.unsqueeze(0) * t).sum(dim=1)  # (d,)
            u_inv_u = (u * invD * u).sum()  # scalar

            den = torch.clamp(a - u_inv_u, min=eps)
            beta0 = (rhs0 - u_inv_rhs) / den  # (d,)
            betab = (rhsb - beta0.unsqueeze(1) * u.unsqueeze(0)) * invD.unsqueeze(0)  # (d x B)

            # Do not remove intercept
            beta0.zero_()

            # Apply correction in one fused operation:
            # For each cell i, use its batch b=batch_id[i] and subtract betab[:,b] * r_k[i]
            betab_cell = betab.index_select(1, batch_id)  # (d x N)
            self._Z_corr.sub_(betab_cell * r_k.unsqueeze(0))

        # Proposal B: adapt lambda multiplier based on correction magnitude and mixing proxy
        # Δ = Z_orig - Z_corr (what we subtracted)
        delta = self._Z_orig - self._Z_corr
        delta_norm = torch.linalg.norm(delta)
        orig_norm = torch.clamp(torch.linalg.norm(self._Z_orig), min=1e-8)
        corr_norm = (delta_norm / orig_norm).item()

        # Mixing proxy: normalized mismatch between observed and expected O/E
        O = self._O
        E = self._E
        mix_mismatch = (torch.linalg.norm(O - E) / torch.clamp(torch.linalg.norm(E), min=1e-8)).item()

        target = float(getattr(self, "_target_corr", 0.12))
        band = float(getattr(self, "_corr_band", 0.04))
        up = float(getattr(self, "_lambda_mult_up", 1.25))
        down = float(getattr(self, "_lambda_mult_down", 0.90))
        lam_min = float(getattr(self, "_lambda_mult_min", 0.25))
        lam_max = float(getattr(self, "_lambda_mult_max", 16.0))
        poor_thresh = float(getattr(self, "_mix_poor_thresh", 0.02))

        if corr_norm > target + band:
            self._lambda_mult = min(lam_max, self._lambda_mult * up)
        elif corr_norm < max(0.0, target - band) and mix_mismatch > poor_thresh:
            # Only relax regularization (allow more correction) when mixing is still poor
            self._lambda_mult = max(lam_min, self._lambda_mult * down)

        # Update Z_cos with rsqrt-based normalization (in-place-friendly)
        inv = torch.rsqrt((self._Z_corr * self._Z_corr).sum(dim=0) + 1e-8)
        self._Z_cos = self._Z_corr * inv.unsqueeze(0)


def safe_entropy_torch(x):
    """Compute x * log(x), returning 0 where x is 0 or negative."""
    result = x * torch.log(x)
    result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
    return result


def harmony_pow_torch(A, T):
    """Element-wise power with different exponents per column (vectorized)."""
    return torch.exp(torch.log(A) * T.unsqueeze(0))


def find_lambda_torch(alpha, cluster_E, device):
    """Compute dynamic lambda based on cluster expected counts."""
    lamb = torch.zeros(len(cluster_E) + 1, dtype=torch.float32, device=device)
    lamb[1:] = cluster_E * alpha
    return lamb
