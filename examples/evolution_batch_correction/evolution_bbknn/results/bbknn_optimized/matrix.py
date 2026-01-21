import warnings
import pandas as pd
import numpy as np
import scipy
import types
import sys
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from umap.umap_ import fuzzy_simplicial_set
from sklearn.metrics import DistanceMetric
from sklearn.neighbors import KDTree

# Optional imports
try:
	from annoy import AnnoyIndex
except ImportError:
	AnnoyIndex = None
try:
	import pynndescent
except ImportError:
	pynndescent = None
try:
	from scanpy import logging as logg
except ImportError:
	pass
try:
	import faiss
except ImportError:
	pass

def get_sparse_matrix_from_indices_distances_umap(knn_indices, knn_dists, n_obs, n_neighbors):
	'''
	Copied out of scanpy.neighbors

	Vectorized version (Change 6).
	'''
	rows = np.repeat(np.arange(n_obs, dtype=np.int64), n_neighbors)
	cols = knn_indices.reshape(-1).astype(np.int64, copy=False)
	vals = knn_dists.reshape(-1).astype(np.float64, copy=False)

	# mask out missing neighbors (-1)
	mask = cols != -1
	rows = rows[mask]
	vals = vals[mask]
	cols = cols[mask]

	# self-distance should be 0
	vals = vals.copy()
	vals[rows == cols] = 0.0

	result = coo_matrix((vals, (rows, cols)), shape=(n_obs, n_obs))
	result.eliminate_zeros()
	return result.tocsr()

def compute_connectivities_umap(
		knn_indices, knn_dists,
		n_obs, n_neighbors, set_op_mix_ratio=1.0,
		local_connectivity=1.0,
		knn_intra_indices=None, knn_intra_dists=None,
		knn_cross_indices=None, knn_cross_dists=None,
		edge_type_mix_lambda=0.5,
		adaptive_lambda=True,
		lam_min=0.03,
	):
	'''
	Copied out of scanpy.neighbors, extended with an edge-type-aware two-channel union.

	If knn_intra_* and knn_cross_* are provided, compute two fuzzy simplicial sets
	(intra-manifold vs cross-batch alignment) and combine them via a controlled union:
		cnts = (1-lambda_i)*cnts_intra + lambda_i*cnts_cross
	where lambda_i can be adaptive per cell based on intra-neighborhood dispersion.

	Falls back to the original single-channel behavior if split inputs are not provided.
	'''
	# original behavior
	if knn_intra_indices is None or knn_cross_indices is None:
		X = coo_matrix(([], ([], [])), shape=(n_obs, 1))
		connectivities = fuzzy_simplicial_set(
			X, n_neighbors, None, None,
			knn_indices=knn_indices, knn_dists=knn_dists,
			set_op_mix_ratio=set_op_mix_ratio,
			local_connectivity=local_connectivity
		)
		if isinstance(connectivities, tuple):
			# In umap-learn 0.4, this returns (result, sigmas, rhos)
			connectivities = connectivities[0]
		distances = get_sparse_matrix_from_indices_distances_umap(
			knn_indices, knn_dists, n_obs, n_neighbors
		)
		return distances, connectivities.tocsr()

	# two-channel fuzzy union
	k_intra = knn_intra_indices.shape[1]
	k_cross = knn_cross_indices.shape[1]

	X = coo_matrix(([], ([], [])), shape=(n_obs, 1))

	cnts_intra = fuzzy_simplicial_set(
		X, k_intra, None, None,
		knn_indices=knn_intra_indices, knn_dists=knn_intra_dists,
		set_op_mix_ratio=set_op_mix_ratio,
		local_connectivity=local_connectivity
	)
	if isinstance(cnts_intra, tuple):
		cnts_intra = cnts_intra[0]
	cnts_intra = cnts_intra.tocsr()

	cnts_cross = fuzzy_simplicial_set(
		X, k_cross, None, None,
		knn_indices=knn_cross_indices, knn_dists=knn_cross_dists,
		set_op_mix_ratio=set_op_mix_ratio,
		local_connectivity=local_connectivity
	)
	if isinstance(cnts_cross, tuple):
		cnts_cross = cnts_cross[0]
	cnts_cross = cnts_cross.tocsr()

	# compute merged distances for downstream compatibility (single sparse distance matrix)
	distances = get_sparse_matrix_from_indices_distances_umap(knn_indices, knn_dists, n_obs, n_neighbors)

	# lambda: either scalar or adaptive per cell using intra-neighborhood dispersion
	if adaptive_lambda:
		# dispersion proxy: median intra distance; map higher dispersion -> higher lambda
		intra_med = np.nanmedian(
			np.where(np.isfinite(knn_intra_dists), knn_intra_dists, np.nan),
			axis=1
		)
		# robust scale to [0,1]
		lo = np.nanpercentile(intra_med, 10)
		hi = np.nanpercentile(intra_med, 90)
		den = (hi - lo) if np.isfinite(hi - lo) and (hi - lo) > 0 else 1.0
		scaled_disp = (intra_med - lo) / den
		scaled_disp = np.clip(scaled_disp, 0.0, 1.0)

		# Fix bias: treat edge_type_mix_lambda as the *maximum* cross weight, scaled by dispersion.
		lam = float(edge_type_mix_lambda) * scaled_disp
		lam = np.clip(lam, float(lam_min), float(edge_type_mix_lambda))
	else:
		lam = float(edge_type_mix_lambda) * np.ones(n_obs, dtype=float)

	# Row-wise scaling (Change 5): avoid diagonal sparse matmuls.
	cnts_intra = cnts_intra.multiply((1.0 - lam)[:, None])
	cnts_cross = cnts_cross.multiply(lam[:, None])
	connectivities = cnts_intra + cnts_cross

	return distances, connectivities.tocsr()

def create_tree(data, params):
	'''
	Create a faiss/cKDTree/KDTree/annoy/pynndescent index for nearest neighbour lookup. 
	All undescribed input as in ``bbknn.bbknn()``. Returns the resulting index.

	Input
	-----
	data : ``numpy.array``
		PCA coordinates of a batch's cells to index.
	params : ``dict``
		A dictionary of arguments used to call ``bbknn.matrix.bbknn()``, plus ['computation']
		storing the knn algorithm to use.
	'''
	if params['computation'] == 'annoy':
		ckd = AnnoyIndex(data.shape[1],metric=params['metric'])
		for i in np.arange(data.shape[0]):
			ckd.add_item(i,data[i,:])
		ckd.build(params['annoy_n_trees'])
	elif params['computation'] == 'pynndescent':
		ckd = pynndescent.NNDescent(data, metric=params['metric'], n_jobs=-1,
									n_neighbors=params['pynndescent_n_neighbors'], 
									random_state=params['pynndescent_random_state'])
		ckd.prepare()
	elif params['computation'] == 'faiss':
		try: # uses GPU if faiss_gpu available.
			res = faiss.StandardGpuResources()
			ckd_ = faiss.IndexFlatL2(data.shape[1])
			ckd = faiss.index_cpu_to_gpu(res, 0, ckd_)
		except:
			ckd = faiss.IndexFlatL2(data.shape[1])
		ckd.add(data)
	elif params['computation'] == 'cKDTree':
		ckd = cKDTree(data)
	elif params['computation'] == 'KDTree':
		ckd = KDTree(data,metric=params['metric'])
	return ckd

def query_tree(data, ckd, params):
	'''
	Query the faiss/cKDTree/KDTree/annoy index with PCA coordinates from a batch. All undescribed input
	as in ``bbknn.bbknn()``. Returns a tuple of distances and indices of neighbours for each cell
	in the batch.

	Input
	-----
	data : ``numpy.array``
		PCA coordinates of a batch's cells to query.
	ckd : faiss/cKDTree/KDTree/annoy/pynndescent index
	params : ``dict``
		A dictionary of arguments used to call ``bbknn.matrix.bbknn()``, plus ['computation']
		storing the knn algorithm to use.
	'''
	if params['computation'] == 'annoy':
		ckdo_ind = []
		ckdo_dist = []
		for i in np.arange(data.shape[0]):
			holder = ckd.get_nns_by_vector(data[i,:],params['neighbors_within_batch'],include_distances=True)
			ckdo_ind.append(holder[0])
			ckdo_dist.append(holder[1])
		ckdout = (np.asarray(ckdo_dist),np.asarray(ckdo_ind))
	elif params['computation'] == 'pynndescent':
		ckdout = ckd.query(data, k=params['neighbors_within_batch'])
		ckdout = (ckdout[1], ckdout[0])
	elif params['computation'] == 'faiss':
		D, I = ckd.search(data, params['neighbors_within_batch'])
		#sometimes this turns up marginally negative values, just set those to zero
		D[D<0] = 0
		#the distance returned by faiss needs to be square rooted to be actual euclidean
		ckdout = (np.sqrt(D), I)
	elif params['computation'] == 'cKDTree':
		ckdout = ckd.query(x=data, k=params['neighbors_within_batch'], workers=-1)
	elif params['computation'] == 'KDTree':
		ckdout = ckd.query(data, k=params['neighbors_within_batch'])
	return ckdout

def get_graph(pca, batch_list, params):
	'''
	Identify the KNN structure to be used in graph construction. All input as in ``bbknn.bbknn()``
	and ``bbknn.matrix.bbknn()``. Returns a tuple of distances and indices of neighbours for
	each cell.

	New behavior (Iteration 9):
	Construct a degree-controlled neighbor list as the union of:
	1) within-batch kNN edges (preserve manifold)
	2) cross-batch OT-derived alignment edges (regularized optimal transport coupling)

	The output has exactly k_intra + k_cross neighbors per cell (padded with -1 where needed).

	Notes on OT implementation:
	- We use entropic Sinkhorn with uniform marginals per batch-pair.
	- To keep this viable, we compute OT on a *sparsified cost graph* obtained by querying
	  k_cross_candidate nearest neighbors across batches (instead of full cost matrices).
	- The OT plan is then converted into per-cell top-k_cross edges by mass.

	Input
	-----
	params : ``dict``
		A dictionary of arguments used to call ``bbknn.matrix.bbknn()``, plus ['computation']
		storing the knn algorithm to use.
	'''
	# get a list of all our batches
	batches = np.unique(batch_list)
	n_batches = len(batches)

	# in case we're gonna be faissing, turn the data to float32
	if params['computation'] == 'faiss':
		pca = pca.astype('float32')

	# Degree budget:
	k_intra = int(params['neighbors_within_batch'])
	# default cross edges slightly smaller than intra because OT edges are stronger constraints
	k_cross = int(params.get('k_cross', int(np.ceil(0.7 * k_intra))))
	k_cross = max(0, k_cross)

	# candidates used to sparsify the OT cost graph (per direction)
	k_cross_candidate = int(params.get('k_cross_candidate', max(30, 5 * max(1, k_cross))))
	# Sinkhorn regularization strength; higher -> smoother coupling
	ot_epsilon = float(params.get('ot_epsilon', 1.0))
	# Sinkhorn iterations/tolerance
	ot_max_iter = int(params.get('ot_max_iter', 200))
	ot_tol = float(params.get('ot_tol', 1e-3))

	# Change 6: distance-aware cross-edge ranking control
	cross_dist_alpha = float(params.get('cross_dist_alpha', 1.0))

	n_cells = pca.shape[0]
	total_k = k_intra + k_cross

	knn_indices = np.full((n_cells, total_k), -1, dtype=int)
	knn_distances = np.full((n_cells, total_k), np.inf, dtype=float)

	# Split outputs for edge-type-aware connectivity computation
	knn_intra_indices = np.full((n_cells, k_intra), -1, dtype=int)
	knn_intra_distances = np.full((n_cells, k_intra), np.inf, dtype=float)
	knn_cross_indices = np.full((n_cells, k_cross), -1, dtype=int) if k_cross > 0 else None
	knn_cross_distances = np.full((n_cells, k_cross), np.inf, dtype=float) if k_cross > 0 else None

	# Precompute indices for each batch (Change 3: ensure sorted for searchsorted mapping)
	batch_to_indices = {b: np.sort(np.where(batch_list == b)[0]) for b in batches}

	# Change 2: prebuild one tree per batch and reuse it
	batch_trees = {}
	for b in batches:
		ind = batch_to_indices[b]
		if ind.size == 0:
			continue
		batch_trees[b] = create_tree(data=pca[ind, :params['n_pcs']], params=params)

	# ---- 1) Within-batch kNN phase ----
	for b in batches:
		ind = batch_to_indices[b]
		if ind.size == 0:
			continue

		ckd = batch_trees[b]

		# Change 3: vectorize within-batch neighbor extraction by querying k_intra+1 and dropping self
		kq = min(k_intra + 1, ind.size)

		params_q = dict(params)
		params_q['neighbors_within_batch'] = kq
		dists, inds_rel = query_tree(data=pca[ind, :params['n_pcs']], ckd=ckd, params=params_q)

		# convert relative indices to global
		inds_global = ind[inds_rel]

		# If self appears first (typical for within-batch queries), drop it.
		# Otherwise, fall back to masking + vectorized selection.
		if kq >= 2 and np.all(inds_global[:, 0] == ind):
			inds_keep = inds_global[:, 1:k_intra + 1]
			dists_keep = dists[:, 1:k_intra + 1]
		else:
			# Change D: vectorized fallback self-removal
			selfmask = (inds_global == ind[:, None])
			dists_mod = np.array(dists, copy=True)
			dists_mod[selfmask] = np.inf

			if k_intra == 0:
				inds_keep = np.empty((ind.size, 0), dtype=int)
				dists_keep = np.empty((ind.size, 0), dtype=float)
			else:
				k_take = min(k_intra, dists_mod.shape[1])
				part = np.argpartition(dists_mod, kth=k_take - 1, axis=1)[:, :k_take]
				inds_keep = np.take_along_axis(inds_global, part, axis=1)
				dists_keep = np.take_along_axis(dists_mod, part, axis=1)

				# sort these k by distance for consistency
				ord2 = np.argsort(dists_keep, axis=1)
				inds_keep = np.take_along_axis(inds_keep, ord2, axis=1)
				dists_keep = np.take_along_axis(dists_keep, ord2, axis=1)

				# pad if k_take < k_intra (tiny batches)
				if k_take < k_intra:
					inds_pad = np.full((ind.size, k_intra), -1, dtype=int)
					dists_pad = np.full((ind.size, k_intra), np.inf, dtype=float)
					inds_pad[:, :k_take] = inds_keep
					dists_pad[:, :k_take] = dists_keep
					inds_keep, dists_keep = inds_pad, dists_pad

		take = inds_keep.shape[1]
		knn_indices[ind, :take] = inds_keep
		knn_distances[ind, :take] = dists_keep
		knn_intra_indices[ind, :take] = inds_keep
		knn_intra_distances[ind, :take] = dists_keep

	# Change C: precompute per-cell intra dispersion scale and alpha once
	tiny = 1e-12
	intra_scale = np.nanmedian(
		np.where(np.isfinite(knn_intra_distances), knn_intra_distances, np.nan),
		axis=1
	)
	intra_scale = np.where(np.isfinite(intra_scale) & (intra_scale > 0), intra_scale, 1.0)
	alpha = cross_dist_alpha / (intra_scale + tiny)

	# Change B: global best cross-edge buffers (avoid overwrite across batch pairs)
	if k_cross > 0:
		cross_best_idx = np.full((n_cells, k_cross), -1, dtype=int)
		cross_best_dist = np.full((n_cells, k_cross), np.inf, dtype=float)
		cross_best_score = np.full((n_cells, k_cross), -np.inf, dtype=float)

	# ---- 2) Cross-batch OT coupling phase (pairwise batches, sparsified costs) ----
	if k_cross > 0 and n_batches > 1:
		def _sinkhorn_sparse_matvec(rows, cols, costs, n_rows, n_cols, epsilon, a, b, max_iter, tol):
			'''
			Change 1: Sparse Sinkhorn-Knopp scaling via vectorized sparse matvecs.
			Build sparse kernel K once, then iterate:
				u = a / (K @ v)
				v = b / (K.T @ u)
			Returns transport mass pi on the provided edges (aligned with rows/cols/costs).

			Change 4: Stabilize kernel computation by shifting costs per-row to prevent underflow.
			'''
			tiny = 1e-12
			eps = max(float(epsilon), tiny)

			# Rows are assumed already grouped (sorted by row) by the caller (Change E).
			# Per-row minima via reduceat directly.
			starts = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]]) if rows.size > 0 else np.array([], dtype=int)
			row_min = np.zeros(n_rows, dtype=float)
			if rows.size > 0:
				row_min_s = np.minimum.reduceat(costs, starts)
				row_min[rows[starts]] = row_min_s

			costs_shifted = costs - row_min[rows]

			# avoid exp underflow/overflow: clamp exponent
			exponent = -costs_shifted / eps
			exponent = np.clip(exponent, -700.0, 700.0)

			K_data = np.exp(exponent).astype(float, copy=False)
			K = scipy.sparse.csr_matrix((K_data, (rows, cols)), shape=(n_rows, n_cols))
			K.sum_duplicates()

			# Change E: use CSC for transpose matvec without explicit transpose-to-CSR materialization
			Kc = K.tocsc()

			u = np.ones(n_rows, dtype=float)
			v = np.ones(n_cols, dtype=float)
			u_old = np.empty_like(u)

			for _ in range(max_iter):
				u_old[:] = u
				Kv = K @ v
				u = a / (Kv + tiny)
				KTu = Kc.T @ u
				v = b / (KTu + tiny)

				du = np.nanmax(np.abs(u - u_old)) if u.size > 0 else 0.0
				if np.isfinite(du) and du < tol:
					break

			pi = u[rows] * K_data * v[cols]
			return pi

		def _select_topk_by_group(src_local, tgt_local, costs_local, scores_local, k):
			'''
			Change A: fully vectorized top-k per source cell via lexsort + rank mask.

			Inputs are 1D edge-aligned arrays:
				src_local: int, source cell index in [0, n_src)
				tgt_local: int, target cell index in [0, n_tgt)
				costs_local: float
				scores_local: float (higher is better)

			Returns:
				sel_src_local, sel_tgt_local, sel_costs, sel_scores
			'''
			if k <= 0 or src_local.size == 0:
				return (np.array([], dtype=int), np.array([], dtype=int),
						np.array([], dtype=float), np.array([], dtype=float))

			# sort by (src, -score, cost)
			order = np.lexsort((costs_local, -scores_local, src_local))
			src_s = src_local[order]
			tgt_s = tgt_local[order]
			cost_s = costs_local[order]
			score_s = scores_local[order]

			starts = np.flatnonzero(np.r_[True, src_s[1:] != src_s[:-1]])
			counts = np.diff(np.r_[starts, src_s.size])

			rank = np.arange(src_s.size) - np.repeat(starts, counts)
			keep = rank < k

			return src_s[keep], tgt_s[keep], cost_s[keep], score_s[keep]

		def _merge_cross_best(best_idx, best_dist, best_score, prop_idx, prop_dist, prop_score, k):
			'''
			Change B: merge current best buffers with per-pair proposals without Python loops.
			Tie-break: primary by score (desc), secondary by distance (asc).
			'''
			if k <= 0:
				return best_idx, best_dist, best_score

			idx_cat = np.concatenate([best_idx, prop_idx], axis=1)
			dist_cat = np.concatenate([best_dist, prop_dist], axis=1)
			score_cat = np.concatenate([best_score, prop_score], axis=1)

			valid = (idx_cat != -1) & np.isfinite(dist_cat) & np.isfinite(score_cat)
			# combined key: score dominates; tiny dist penalty breaks ties deterministically
			key = np.where(valid, score_cat - 1e-6 * dist_cat, -np.inf)

			# choose top-k by key
			part = np.argpartition(-key, kth=min(k - 1, key.shape[1] - 1), axis=1)[:, :k]
			key_sel = np.take_along_axis(key, part, axis=1)
			dist_sel = np.take_along_axis(dist_cat, part, axis=1)

			# finalize ordering on this small set (desc key, asc dist)
			ord2 = np.lexsort((dist_sel, -key_sel), axis=1)
			part2 = np.take_along_axis(part, ord2, axis=1)

			new_idx = np.take_along_axis(idx_cat, part2, axis=1)
			new_dist = np.take_along_axis(dist_cat, part2, axis=1)
			new_score = np.take_along_axis(score_cat, part2, axis=1)

			# ensure invalid positions are reset (in case fewer than k valid)
			new_valid = (new_idx != -1) & np.isfinite(new_dist) & np.isfinite(new_score)
			new_idx = np.where(new_valid, new_idx, -1)
			new_dist = np.where(new_valid, new_dist, np.inf)
			new_score = np.where(new_valid, new_score, -np.inf)

			return new_idx, new_dist, new_score

		for a_i in range(n_batches):
			b1 = batches[a_i]
			ind1 = batch_to_indices[b1]
			if ind1.size == 0:
				continue
			for a_j in range(a_i + 1, n_batches):
				b2 = batches[a_j]
				ind2 = batch_to_indices[b2]
				if ind2.size == 0:
					continue

				ckd2 = batch_trees[b2]
				ckd1 = batch_trees[b1]

				k12 = min(k_cross_candidate, ind2.size)
				k21 = min(k_cross_candidate, ind1.size)

				params_12 = dict(params)
				params_12['neighbors_within_batch'] = k12
				d12, i12_rel = query_tree(data=pca[ind1, :params['n_pcs']], ckd=ckd2, params=params_12)
				i12 = ind2[i12_rel]  # global indices in b2

				params_21 = dict(params)
				params_21['neighbors_within_batch'] = k21
				d21, i21_rel = query_tree(data=pca[ind2, :params['n_pcs']], ckd=ckd1, params=params_21)
				i21 = ind1[i21_rel]  # global indices in b1

				# b1 -> b2 candidate edges
				rows12 = np.repeat(np.arange(ind1.size, dtype=int), i12.shape[1])
				cand12 = i12.reshape(-1)

				pos12 = np.searchsorted(ind2, cand12)
				ok12 = (pos12 >= 0) & (pos12 < ind2.size) & (ind2[pos12] == cand12)
				cols12 = pos12
				costs12 = d12.reshape(-1).astype(float, copy=False)

				# b2 -> b1 candidate edges (store as b1->b2 edges by swapping)
				cand21 = i21.reshape(-1)
				pos21 = np.searchsorted(ind1, cand21)
				ok21 = (pos21 >= 0) & (pos21 < ind1.size) & (ind1[pos21] == cand21)

				rows21 = pos21
				cols21 = np.repeat(np.arange(ind2.size, dtype=int), i21.shape[1])
				costs21 = d21.reshape(-1).astype(float, copy=False)

				# filter invalid mappings and non-finite costs
				m12 = ok12 & np.isfinite(costs12)
				m21 = ok21 & np.isfinite(costs21)

				rows_all = np.concatenate([rows12[m12], rows21[m21]]).astype(int, copy=False)
				cols_all = np.concatenate([cols12[m12], cols21[m21]]).astype(int, copy=False)
				costs_all = np.concatenate([costs12[m12], costs21[m21]]).astype(float, copy=False)

				if rows_all.size == 0:
					continue

				# min-reduce duplicates by sorting on linearized (row, col) keys
				key = rows_all.astype(np.int64) * np.int64(ind2.size) + cols_all.astype(np.int64)
				order = np.argsort(key, kind='mergesort')
				key_s = key[order]
				rows_s = rows_all[order]
				cols_s = cols_all[order]
				costs_s = costs_all[order]

				starts = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1]])
				min_costs = np.minimum.reduceat(costs_s, starts) if costs_s.size > 0 else np.array([], dtype=float)

				rows = rows_s[starts]
				cols = cols_s[starts]
				costs = min_costs

				# Change E: ensure grouped by rows for Sinkhorn row-min optimization
				order_r = np.argsort(rows, kind='mergesort')
				rows = rows[order_r]
				cols = cols[order_r]
				costs = costs[order_r]

				# Uniform marginals per batch (balanced OT)
				a = np.ones(ind1.size, dtype=float) / float(ind1.size)
				b = np.ones(ind2.size, dtype=float) / float(ind2.size)

				pi = _sinkhorn_sparse_matvec(
					rows=rows, cols=cols, costs=costs,
					n_rows=ind1.size, n_cols=ind2.size,
					epsilon=ot_epsilon, a=a, b=b,
					max_iter=ot_max_iter, tol=ot_tol
				)

				# Change C: use precomputed alpha (slice once)
				alpha1 = alpha[ind1]
				alpha2 = alpha[ind2]

				# scores for both directions (same edge set)
				logpi = np.log(pi + 1e-30)
				scores1 = logpi - alpha1[rows] * costs
				scores2 = logpi - alpha2[cols] * costs

				# per-pair proposals (filled only for participating cells)
				pair_idx = np.full((n_cells, k_cross), -1, dtype=int)
				pair_dist = np.full((n_cells, k_cross), np.inf, dtype=float)
				pair_score = np.full((n_cells, k_cross), -np.inf, dtype=float)

				# Fill proposals for ind1 (sources = rows, targets = cols)
				s_src, s_tgt, s_cost, s_score = _select_topk_by_group(
					src_local=rows.astype(int, copy=False),
					tgt_local=cols.astype(int, copy=False),
					costs_local=costs.astype(float, copy=False),
					scores_local=scores1.astype(float, copy=False),
					k=k_cross
				)
				if s_src.size > 0:
					# scatter into [global_cell, rank] positions
					starts_s = np.flatnonzero(np.r_[True, s_src[1:] != s_src[:-1]])
					counts_s = np.diff(np.r_[starts_s, s_src.size])
					rank_s = np.arange(s_src.size) - np.repeat(starts_s, counts_s)
					cell_g = ind1[s_src]
					pair_idx[cell_g, rank_s] = ind2[s_tgt]
					pair_dist[cell_g, rank_s] = s_cost
					pair_score[cell_g, rank_s] = s_score

				# Fill proposals for ind2 (sources = cols, targets = rows)
				t_src, t_tgt, t_cost, t_score = _select_topk_by_group(
					src_local=cols.astype(int, copy=False),
					tgt_local=rows.astype(int, copy=False),
					costs_local=costs.astype(float, copy=False),
					scores_local=scores2.astype(float, copy=False),
					k=k_cross
				)
				if t_src.size > 0:
					starts_t = np.flatnonzero(np.r_[True, t_src[1:] != t_src[:-1]])
					counts_t = np.diff(np.r_[starts_t, t_src.size])
					rank_t = np.arange(t_src.size) - np.repeat(starts_t, counts_t)
					cell_g2 = ind2[t_src]
					pair_idx[cell_g2, rank_t] = ind1[t_tgt]
					pair_dist[cell_g2, rank_t] = t_cost
					pair_score[cell_g2, rank_t] = t_score

				# Change B: merge into global best buffers (no overwriting)
				cross_best_idx, cross_best_dist, cross_best_score = _merge_cross_best(
					cross_best_idx, cross_best_dist, cross_best_score,
					pair_idx, pair_dist, pair_score, k_cross
				)

		# After all batch pairs, write cross buffers into knn_* (one final fill)
		knn_cross_indices = cross_best_idx
		knn_cross_distances = cross_best_dist
		knn_indices[:, k_intra:k_intra + k_cross] = cross_best_idx
		knn_distances[:, k_intra:k_intra + k_cross] = cross_best_dist

	return knn_distances, knn_indices, knn_intra_distances, knn_intra_indices, knn_cross_distances, knn_cross_indices

def print_warning(message, scanpy_logging):
	'''
	Print a specified warning to the screen, using either warnings or scanpy.
	
	Input
	-----
	message : ``str``
		The warning message to print
	scanpy_logging : ``bool``
		Whether to use scanpy logging to print updates rather than ``warnings.warn()``
	'''
	if scanpy_logging:
		logg.warning(message)
	else:
		warnings.warn(message)

def legacy_computation_selection(params, scanpy_logging):
	'''
	Do pre-1.6.0 computation algorithm selection based on possible legacy arguments. Looks 
	at the following (default None) parameters: approx, use_annoy, use_faiss
	
	Input
	-----
	params : ``dict``
		A dictionary of arguments used to call ``bbknn.matrix.bbknn()``
	scanpy_logging : ``bool``
		Whether to use scanpy logging to print updates rather than ``warnings.warn()``
	'''
	#if these are not None then they were set at the time of the call
	if any([params[i] is not None for i in ['approx','use_annoy','use_faiss']]):
		#encourage upgrading the call
		print_warning('consider updating your call to make use of `computation`', scanpy_logging)
		#fill in any missing defaults
		if params['approx'] is None:
			params['approx'] = True
		if params['use_annoy'] is None:
			params['use_annoy'] = True
		if params['use_faiss'] is None:
			params['use_faiss'] = True
		#now that we have all the old defaults restored, use them to pick a computation
		if params['approx']:
			#pick between these two packages based on another param
			if params['use_annoy']:
				params['computation'] = 'annoy'
			else:
				params['computation'] = 'pynndescent'
		else:
			#if the metric is euclidean, then pick between these two packages
			if params['metric'] == 'euclidean':
				if params['use_faiss'] and 'faiss' in sys.modules:
					params['computation'] = 'faiss'
				else:
					params['computation'] = 'cKDTree'
			else:
				params['computation'] = 'KDTree'
	return params

def check_knn_metric(params, counts, scanpy_logging):
	'''
	Checks if the provided metric can be used with the implied KNN algorithm. Returns parameters
	with the metric validated.
	
	Input
	-----
	params : ``dict``
		A dictionary of arguments used to call ``bbknn.matrix.bbknn()``
	counts : ``np.array``
		The number of cells in each batch
	scanpy_logging : ``bool``
		Whether to use scanpy logging to print updates rather than ``warnings.warn()``
	'''
	#take note if we end up going back to Euclidean
	swapped = False
	if params['computation'] == 'annoy':
		if params['metric'] not in ['angular', 'euclidean', 'manhattan', 'hamming']:
			swapped = True
	elif params['computation'] == 'pynndescent':
		if np.min(counts) < 11:
			raise ValueError("Not all batches have at least 11 cells in them - required by pynndescent.")
		#metric needs to be a function or in the named list
		if not (params['metric'] in pynndescent.distances.named_distances or
				isinstance(params['metric'], types.FunctionType)):
			swapped = True
	elif params['computation'] == 'faiss':
		if params['metric'] != 'euclidean':
			swapped = True
	elif params['computation'] == 'cKDTree':
		if params['metric'] != 'euclidean':
			swapped = True
	elif params['computation'] == 'KDTree':
		if not (isinstance(params['metric'], DistanceMetric) or 
				params['metric'] in KDTree.valid_metrics):
			swapped = True
			#and now for a next level swap - this can be done more efficiently via cKDTree
			params['computation'] = 'cKDTree'
	else:
		raise ValueError("Incorrect value of `computation`.")
	if swapped:
		#go back to euclidean
		params['metric'] = 'euclidean'
		#need to let the user know we swapped the metric
		print_warning('unrecognised metric for type of neighbor calculation, switching to euclidean', scanpy_logging)
	return params

def trimming(cnts, trim, mutual_mask=None):
	'''
	Trims the graph to the top connectivities for each cell, while optionally preserving
	mutual edges. All undescribed input as in ``bbknn.bbknn()``.

	Input
	-----
	cnts : ``CSR``
		Sparse matrix of processed connectivities to trim.
	trim : ``int``
		Maximum number of outgoing edges to retain per node.
	mutual_mask : ``CSR`` or ``None``
		Optional boolean sparse matrix where True entries indicate mutual edges to
		preferentially retain during trimming.
	'''
	# fast path: original behavior if no mutual mask provided
	if mutual_mask is None:
		vals = np.zeros(cnts.shape[0])
		for i in range(cnts.shape[0]):
			#Get the row slice, not a copy, only the non zero elements
			row_array = cnts.data[cnts.indptr[i]: cnts.indptr[i+1]]
			if row_array.shape[0] <= trim:
				continue
			#fish out the threshold value
			vals[i] = row_array[np.argsort(row_array)[-1*trim]]
		for iter in range(2):
			#filter rows, flip, filter columns using the same thresholds
			for i in range(cnts.shape[0]):
				#Get the row slice, not a copy, only the non zero elements
				row_array = cnts.data[cnts.indptr[i]: cnts.indptr[i+1]]
				#apply cutoff
				row_array[row_array<vals[i]] = 0
			cnts.eliminate_zeros()
			cnts = cnts.T.tocsr()
		return cnts

	# mutual-first trimming: always keep mutual edges, fill remaining slots by weight
	cnts = cnts.tocsr()
	mutual_mask = mutual_mask.tocsr()

	out = cnts.copy()
	out.data = out.data.copy()

	# operate row-wise; avoid transpose-based thresholding that can drop key cross-batch anchors
	for i in range(out.shape[0]):
		start = out.indptr[i]
		end = out.indptr[i+1]
		row_nnz = end - start
		if row_nnz <= trim:
			continue

		cols = out.indices[start:end]
		data = out.data[start:end]

		mstart = mutual_mask.indptr[i]
		mend = mutual_mask.indptr[i+1]
		mcols = mutual_mask.indices[mstart:mend]

		# Change 5: vectorized membership rather than Python set() per row
		if mcols.size == 0:
			is_mutual = np.zeros(cols.shape[0], dtype=bool)
		else:
			is_mutual = np.in1d(cols, mcols, assume_unique=False)

		mutual_count = int(is_mutual.sum())

		# if too many mutual edges, keep strongest mutual only
		if mutual_count >= trim:
			mutual_idx = np.where(is_mutual)[0]
			keep_rel = mutual_idx[np.argsort(data[mutual_idx])[-trim:]]
		else:
			non_mutual_idx = np.where(~is_mutual)[0]
			need = trim - mutual_count
			if need > 0 and non_mutual_idx.size > 0:
				keep_non = non_mutual_idx[np.argsort(data[non_mutual_idx])[-need:]]
				keep_rel = np.concatenate([np.where(is_mutual)[0], keep_non])
			else:
				keep_rel = np.where(is_mutual)[0]

		keep_mask = np.zeros(len(cols), dtype=bool)
		keep_mask[keep_rel] = True
		# zero out dropped edges; eliminate_zeros later
		data[~keep_mask] = 0.0
		out.data[start:end] = data

	out.eliminate_zeros()
	return out

def bbknn(pca, batch_list, neighbors_within_batch=3, n_pcs=50, trim=None,
		  computation='annoy', annoy_n_trees=10, pynndescent_n_neighbors=30, 
		  pynndescent_random_state=0, metric='euclidean', set_op_mix_ratio=1, 
		  local_connectivity=1, approx=None, use_annoy=None, use_faiss=None,
		  scanpy_logging=False):
	'''
	Scanpy-independent BBKNN variant that runs on a PCA matrix and list of per-cell batch assignments instead of
	an AnnData object. Non-data-entry arguments behave the same way as ``bbknn.bbknn()``.
	Returns a ``(distances, connectivities, parameters)`` tuple, like what would have been stored in the AnnData object.
	The connectivities are the actual neighbourhood graph.

	Input
	-----
	pca : ``numpy.array``
		PCA (or other dimensionality reduction) coordinates for each cell, with cells as rows.
	batch_list : ``numpy.array`` or ``list``
		A list of batch assignments for each cell.
	scanpy_logging : ``bool``, optional (default: ``False``)
		Whether to use scanpy logging to print updates rather than ``warnings.warn()``
	'''
	#catch all arguments for easy passing to subsequent functions
	params = locals()
	del params['pca']
	del params['batch_list']
	#more basic sanity checks/processing
	#do we have the same number of cells in pca and batch_list?
	if pca.shape[0] != len(batch_list):
		raise ValueError("Different cell counts indicated by `pca.shape[0]` and `len(batch_list)`.")
	#make sure n_pcs is not larger than the actual dimensions of the data
	#as that can introduce some uninformative knock-on errors in UMAP
	params['n_pcs'] = np.min([params['n_pcs'], pca.shape[1]])
	#convert batch_list to np.array of strings for ease of mask making later
	batch_list = np.asarray([str(i) for i in batch_list])
	#assert that all batches have at least neighbors_within_batch cells in there
	unique, counts = np.unique(batch_list, return_counts=True)
	if np.min(counts) < params['neighbors_within_batch']:
		raise ValueError("Not all batches have at least `neighbors_within_batch` cells in them.")
	#if any of the old legacy computation selection arguments are detected
	#use them to select the knn computation algorithm
	params = legacy_computation_selection(params, scanpy_logging)
	#sanity check the metric
	params = check_knn_metric(params, counts, scanpy_logging)

	# obtain degree-controlled graph: within-batch kNN + cross-batch OT alignment edges
	(knn_distances, knn_indices,
	 knn_intra_distances, knn_intra_indices,
	 knn_cross_distances, knn_cross_indices) = get_graph(pca=pca, batch_list=batch_list, params=params)

	# sort the merged neighbours so that they're actually in order from closest to furthest
	newidx = np.argsort(knn_distances, axis=1)
	knn_indices = knn_indices[np.arange(np.shape(knn_indices)[0])[:, np.newaxis], newidx]
	knn_distances = knn_distances[np.arange(np.shape(knn_distances)[0])[:, np.newaxis], newidx]

	# Edge-type-aware UMAP connectivities: two-channel fuzzy union (intra vs cross)
	dist, cnts = compute_connectivities_umap(
		knn_indices, knn_distances, knn_indices.shape[0], knn_indices.shape[1],
		set_op_mix_ratio=set_op_mix_ratio,
		local_connectivity=local_connectivity,
		knn_intra_indices=knn_intra_indices,
		knn_intra_dists=knn_intra_distances,
		knn_cross_indices=knn_cross_indices,
		knn_cross_dists=knn_cross_distances,
		edge_type_mix_lambda=float(params.get('edge_type_mix_lambda', 0.5)),
		adaptive_lambda=bool(params.get('adaptive_lambda', True)),
		lam_min=float(params.get('lam_min', 0.03)),
	)

	# Degree is controlled at construction time; trimming is generally unnecessary.
	# Preserve API: if user explicitly requested trimming (>0), apply legacy trimming.
	if params['trim'] is None:
		# default: disable trimming under the new degree-controlled construction
		params['trim'] = 0
	if params['trim'] > 0:
		cnts = trimming(cnts=cnts, trim=params['trim'])

	#create a collated parameters dictionary, formatted like scanpy's neighbours one
	p_dict = {'n_neighbors': knn_distances.shape[1], 'method': 'umap', 
			  'metric': params['metric'], 'n_pcs': params['n_pcs'], 
			  'bbknn': {'trim': params['trim'], 'computation': params['computation']}}
	return (dist, cnts, p_dict)
