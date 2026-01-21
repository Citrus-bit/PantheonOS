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
	'''
	rows = np.zeros((n_obs * n_neighbors), dtype=np.int64)
	cols = np.zeros((n_obs * n_neighbors), dtype=np.int64)
	vals = np.zeros((n_obs * n_neighbors), dtype=np.float64)

	for i in range(knn_indices.shape[0]):
		for j in range(n_neighbors):
			if knn_indices[i, j] == -1:
				continue  # We didn't get the full knn for i
			if knn_indices[i, j] == i:
				val = 0.0
			else:
				val = knn_dists[i, j]

			rows[i * n_neighbors + j] = i
			cols[i * n_neighbors + j] = knn_indices[i, j]
			vals[i * n_neighbors + j] = val

	result = coo_matrix((vals, (rows, cols)),
									  shape=(n_obs, n_obs))
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
	):
	'''
	Copied out of scanpy.neighbors.

	Iteration 10 behavior change:
	- Primary control of cross-batch influence is moved from per-cell mixing (lambda_i)
	  to per-edge confidence weighting encoded directly in knn_cross_dists (effective distances).
	- If split inputs are provided, we compute a two-channel union but *without* adaptive
	  row-wise scaling; instead, we use a fixed scalar mix as a mild prior.

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

	# two-channel fuzzy union (confidence-weighted via effective cross distances)
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

	# mild fixed prior mix (do not adapt per-cell; rely on per-edge confidence already encoded)
	alpha = float(edge_type_mix_lambda)
	alpha = 0.0 if not np.isfinite(alpha) else float(np.clip(alpha, 0.0, 1.0))
	connectivities = (1.0 - alpha) * cnts_intra + alpha * cnts_cross

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

	New behavior (Iteration 10):
	Construct a degree-controlled neighbor list as the union of:
	1) within-batch kNN edges (preserve manifold)
	2) cross-batch MNN/CSLS anchor edges (reciprocal, local alignment)

	Cross edges are selected via Mutual Nearest Neighbors (MNN) per batch-pair, and ranked
	by a simple CSLS-like corrected score computed from the same neighbor queries:
		score(i,j) = -d(i,j) + 0.5*(mean_k d(i, N_b2(i)) + mean_k d(j, N_b1(j)))

	Per-edge confidence is encoded as an *effective distance scaling* for cross edges:
		d_eff = d / (w + eps)
	where w in (0,1] is derived from the corrected score margin for each cell.

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
	# default cross edges comparable to intra for strong mixing, but still local/reciprocal
	k_cross = int(params.get('k_cross', int(np.ceil(0.7 * k_intra))))
	k_cross = max(0, k_cross)

	# candidates used for MNN/CSLS (per direction)
	k_cross_candidate = int(params.get('k_cross_candidate', max(30, 5 * max(1, k_cross))))
	k_cross_candidate = max(1, k_cross_candidate)

	# confidence controls
	confidence_eps = float(params.get('cross_confidence_eps', 1e-8))
	confidence_power = float(params.get('cross_confidence_power', 1.0))

	n_cells = pca.shape[0]
	total_k = k_intra + k_cross

	knn_indices = np.full((n_cells, total_k), -1, dtype=int)
	knn_distances = np.full((n_cells, total_k), np.inf, dtype=float)

	# Split outputs for edge-type-aware connectivity computation
	knn_intra_indices = np.full((n_cells, k_intra), -1, dtype=int)
	knn_intra_distances = np.full((n_cells, k_intra), np.inf, dtype=float)
	knn_cross_indices = np.full((n_cells, k_cross), -1, dtype=int) if k_cross > 0 else None
	knn_cross_distances = np.full((n_cells, k_cross), np.inf, dtype=float) if k_cross > 0 else None

	# Precompute indices for each batch
	batch_to_indices = {b: np.where(batch_list == b)[0] for b in batches}

	# ---- 1) Within-batch kNN phase ----
	for b in batches:
		ind = batch_to_indices[b]
		if ind.size == 0:
			continue
		# build index within batch
		ckd = create_tree(data=pca[ind, :params['n_pcs']], params=params)

		# query only points in this batch against the within-batch pool
		params_q = dict(params)
		params_q['neighbors_within_batch'] = min(k_intra, ind.size)
		dists, inds_rel = query_tree(data=pca[ind, :params['n_pcs']], ckd=ckd, params=params_q)

		# convert relative indices to global
		inds_global = ind[inds_rel]

		# remove self if present; keep closest k_intra non-self
		for ii, gi in enumerate(ind):
			row_inds = inds_global[ii]
			row_d = dists[ii]
			mask = row_inds != gi
			row_inds = row_inds[mask]
			row_d = row_d[mask]
			take = min(k_intra, row_inds.shape[0])
			if take > 0:
				order = np.argsort(row_d)[:take]
				knn_indices[gi, :take] = row_inds[order]
				knn_distances[gi, :take] = row_d[order]
				knn_intra_indices[gi, :take] = row_inds[order]
				knn_intra_distances[gi, :take] = row_d[order]

	# ---- 2) Cross-batch MNN/CSLS anchor phase ----
	if k_cross > 0 and n_batches > 1:
		# For each cell: accumulate candidates as (score, d_raw, d_eff, j)
		cross_lists = [[] for _ in range(n_cells)]

		def _row_means(d):
			# mean over finite values; fall back to nanmean semantics
			d = np.asarray(d, dtype=float)
			with np.errstate(invalid='ignore'):
				m = np.nanmean(np.where(np.isfinite(d), d, np.nan), axis=1)
			# replace NaNs with global median if needed
			if np.any(~np.isfinite(m)):
				fallback = np.nanmedian(np.where(np.isfinite(d), d, np.nan))
				if not np.isfinite(fallback):
					fallback = 0.0
				m[~np.isfinite(m)] = fallback
			return m

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

				# Build cross-batch candidate graph by kNN queries in both directions.
				ckd2 = create_tree(data=pca[ind2, :params['n_pcs']], params=params)
				ckd1 = create_tree(data=pca[ind1, :params['n_pcs']], params=params)

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

				# CSLS density correction terms: mean neighbor distance across batches
				r1 = _row_means(d12)  # for cells in ind1
				r2 = _row_means(d21)  # for cells in ind2

				# Build fast reverse lookup: for each cell in ind2, map neighbor->rank in b2->b1 query
				# and store distance for that neighbor.
				reverse_rank = {}
				reverse_dist = {}
				for r in range(ind2.size):
					gj = int(ind2[r])
					neis = i21[r]
					dd = d21[r]
					mr = {}
					md = {}
					for t in range(neis.shape[0]):
						gi = int(neis[t])
						# first occurrence is best due to knn order
						if gi not in mr:
							mr[gi] = t
							md[gi] = float(dd[t])
					reverse_rank[gj] = mr
					reverse_dist[gj] = md

				# Collect mutual edges and score them
				for r in range(ind1.size):
					gi = int(ind1[r])
					cands = i12[r]
					dd = d12[r]
					for t in range(cands.shape[0]):
						gj = int(cands[t])
						if gj not in reverse_rank:
							continue
						mr = reverse_rank[gj]
						# mutual check: gi appears in gj's top-k21
						if gi not in mr:
							continue

						# ranks (0 is best)
						rank_ij = int(t)
						rank_ji = int(mr[gi])

						d_ij = float(dd[t])
						# use the exact reverse distance if available (should be)
						d_ji = float(reverse_dist[gj].get(gi, d_ij))
						d_raw = 0.5 * (d_ij + d_ji)

						# CSLS-like corrected score (higher is better)
						score = (-d_raw) + 0.5 * (float(r1[r]) + float(r2[rank_ji if False else np.where(ind2 == gj)[0][0] if False else 0]))
						# The r2 term should be per gj. Get its local index:
						j_local = np.where(ind2 == gj)[0]
						if j_local.size > 0:
							score = (-d_raw) + 0.5 * (float(r1[r]) + float(r2[int(j_local[0])]))
						else:
							score = (-d_raw) + 0.5 * (float(r1[r]) + float(np.nanmedian(r2)))

						# rank agreement confidence (smaller ranks => higher confidence)
						# map to (0,1]
						rconf = 1.0 / (1.0 + 0.5 * (rank_ij + rank_ji))

						# we will compute margin-based confidence later per-cell; store rconf as a prior
						cross_lists[gi].append((score, d_raw, rconf, gj))
						cross_lists[gj].append((score, d_raw, rconf, gi))

		# finalize per-cell top-k_cross by corrected score; compute confidence by score margin
		for i in range(n_cells):
			if len(cross_lists[i]) == 0:
				continue

			# unique by neighbor id: keep best score (tie-break by smaller raw distance)
			best = {}
			for score, d_raw, rconf, j in cross_lists[i]:
				if j not in best or (score > best[j][0]) or (score == best[j][0] and d_raw < best[j][1]):
					best[j] = (float(score), float(d_raw), float(rconf))

			items = [(s, d, rc, j) for j, (s, d, rc) in best.items()]
			items.sort(key=lambda x: (-x[0], x[1]))  # max score, then min dist

			take = min(k_cross, len(items))
			if take <= 0:
				continue

			# score margin confidence for top edges
			best_score = items[0][0]
			next_score = items[take][0] if len(items) > take else (items[-1][0] if len(items) > 0 else best_score)
			margin = best_score - next_score
			if not np.isfinite(margin) or margin <= 0:
				margin = 0.0

			js = np.asarray([items[t][3] for t in range(take)], dtype=int)
			ds_raw = np.asarray([items[t][1] for t in range(take)], dtype=float)
			rcs = np.asarray([items[t][2] for t in range(take)], dtype=float)
			scores = np.asarray([items[t][0] for t in range(take)], dtype=float)

			# per-edge confidence: combine rank-confidence with normalized score (relative to best)
			# map score deltas into (0,1] via logistic-ish transform
			delta = scores - scores.max()
			w_score = np.exp(delta)  # <= 1
			w = (0.5 * rcs + 0.5 * w_score)
			w = np.clip(w, confidence_eps, 1.0)
			w = np.power(w, confidence_power)

			# effective distances: high confidence => smaller effective distance => stronger membership
			ds_eff = ds_raw / (w + confidence_eps)

			start = k_intra
			end = k_intra + take
			knn_indices[i, start:end] = js
			knn_distances[i, start:end] = ds_eff
			knn_cross_indices[i, :take] = js
			knn_cross_distances[i, :take] = ds_eff

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
		mset = set(mcols.tolist()) if (mend - mstart) > 0 else set()

		is_mutual = np.fromiter((c in mset for c in cols), dtype=bool, count=len(cols))
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

