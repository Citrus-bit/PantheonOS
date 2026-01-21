from annoy import AnnoyIndex
from intervaltree import IntervalTree
from itertools import cycle, islice
import numpy as np
import operator
import random
import scipy
from scipy.sparse import csc_matrix, csr_matrix, vstack
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import sys
import warnings

from .utils import plt, dispersion, reduce_dimensionality
from .utils import visualize_cluster, visualize_expr, visualize_dropout
from .utils import handle_zeros_in_scale

# Default parameters.
ALPHA = 0.10
APPROX = True
BATCH_SIZE = 5000
DIMRED = 100
HVG = None
KNN = 20
N_ITER = 500
PERPLEXITY = 1200
SIGMA = 15
VERBOSE = 2

# Do batch correction on a list of data sets.
def correct(datasets_full, genes_list, return_dimred=False,
            batch_size=BATCH_SIZE, verbose=VERBOSE, ds_names=None,
            dimred=DIMRED, approx=APPROX, sigma=SIGMA, alpha=ALPHA, knn=KNN,
            return_dense=False, hvg=None, union=False, seed=0):
    """Integrate and batch correct a list of data sets.

    Parameters
    ----------
    datasets_full : `list` of `scipy.sparse.csr_matrix` or of `numpy.ndarray`
        Data sets to integrate and correct.
    genes_list: `list` of `list` of `string`
        List of genes for each data set.
    return_dimred: `bool`, optional (default: `False`)
        In addition to returning batch corrected matrices, also returns
        integrated low-dimesional embeddings.
    batch_size: `int`, optional (default: `5000`)
        The batch size used in the alignment vector computation. Useful when
        correcting very large (>100k samples) data sets. Set to large value
        that runs within available memory.
    verbose: `bool` or `int`, optional (default: 2)
        When `True` or not equal to 0, prints logging output.
    ds_names: `list` of `string`, optional
        When `verbose=True`, reports data set names in logging output.
    dimred: `int`, optional (default: 100)
        Dimensionality of integrated embedding.
    approx: `bool`, optional (default: `True`)
        Use approximate nearest neighbors, greatly speeds up matching runtime.
    sigma: `float`, optional (default: 15)
        Correction smoothing parameter on Gaussian kernel.
    alpha: `float`, optional (default: 0.10)
        Alignment score minimum cutoff.
    knn: `int`, optional (default: 20)
        Number of nearest neighbors to use for matching.
    return_dense: `bool`, optional (default: `False`)
        Return `numpy.ndarray` matrices instead of `scipy.sparse.csr_matrix`.
    hvg: `int`, optional (default: None)
        Use this number of top highly variable genes based on dispersion.
    seed: `int`, optional (default: 0)
        Random seed to use.

    Returns
    -------
    corrected, genes
        By default (`return_dimred=False`), returns a two-tuple containing a
        list of `scipy.sparse.csr_matrix` each with batch corrected values,
        and a single list of genes containing the intersection of inputted
        genes.

    integrated, corrected, genes
        When `return_dimred=True`, returns a three-tuple containing a list
        of `numpy.ndarray` with integrated low dimensional embeddings, a list
        of `scipy.sparse.csr_matrix` each with batch corrected values, and a
        a single list of genes containing the intersection of inputted genes.
    """
    np.random.seed(seed)
    random.seed(seed)

    datasets_full = check_datasets(datasets_full)

    datasets, genes = merge_datasets(datasets_full, genes_list,
                                     ds_names=ds_names, union=union)
    datasets_dimred, genes = process_data(datasets, genes, hvg=hvg,
                                          dimred=dimred)

    datasets_dimred = assemble(
        datasets_dimred, # Assemble in low dimensional space.
        expr_datasets=datasets, # Modified in place.
        verbose=verbose, knn=knn, sigma=sigma, approx=approx,
        alpha=alpha, ds_names=ds_names, batch_size=batch_size,
    )

    if return_dense:
        datasets = [ ds.toarray() for ds in datasets ]

    if return_dimred:
        return datasets_dimred, datasets, genes

    return datasets, genes

# Integrate a list of data sets.
def integrate(datasets_full, genes_list, batch_size=BATCH_SIZE,
              verbose=VERBOSE, ds_names=None, dimred=DIMRED, approx=APPROX,
              sigma=SIGMA, alpha=ALPHA, knn=KNN, union=False, hvg=None, seed=0,
              sketch=False, sketch_method='geosketch', sketch_max=10000,):
    """Integrate a list of data sets.

    Parameters
    ----------
    datasets_full : `list` of `scipy.sparse.csr_matrix` or of `numpy.ndarray`
        Data sets to integrate and correct.
    genes_list: `list` of `list` of `string`
        List of genes for each data set.
    batch_size: `int`, optional (default: `5000`)
        The batch size used in the alignment vector computation. Useful when
        correcting very large (>100k samples) data sets. Set to large value
        that runs within available memory.
    verbose: `bool` or `int`, optional (default: 2)
        When `True` or not equal to 0, prints logging output.
    ds_names: `list` of `string`, optional
        When `verbose=True`, reports data set names in logging output.
    dimred: `int`, optional (default: 100)
        Dimensionality of integrated embedding.
    approx: `bool`, optional (default: `True`)
        Use approximate nearest neighbors, greatly speeds up matching runtime.
    sigma: `float`, optional (default: 15)
        Correction smoothing parameter on Gaussian kernel.
    alpha: `float`, optional (default: 0.10)
        Alignment score minimum cutoff.
    knn: `int`, optional (default: 20)
        Number of nearest neighbors to use for matching.
    hvg: `int`, optional (default: None)
        Use this number of top highly variable genes based on dispersion.
    seed: `int`, optional (default: 0)
        Random seed to use.
    sketch: `bool`, optional (default: False)
        Apply sketching-based acceleration by first downsampling the datasets.
        See Hie et al., Cell Systems (2019).
    sketch_method: {'geosketch', 'uniform'}, optional (default: `geosketch`)
        Apply the given sketching method to the data. Only used if
        `sketch=True`.
    sketch_max: `int`, optional (default: 10000)
        If a dataset has more cells than `sketch_max`, downsample to
        `sketch_max` using the method provided in `sketch_method`. Only used
        if `sketch=True`.

    Returns
    -------
    integrated, genes
        Returns a two-tuple containing a list of `numpy.ndarray` with
        integrated low dimensional embeddings and a single list of genes
        containing the intersection of inputted genes.
    """
    np.random.seed(seed)
    random.seed(seed)

    datasets_full = check_datasets(datasets_full)

    datasets, genes = merge_datasets(datasets_full, genes_list,
                                     ds_names=ds_names, union=union)
    datasets_dimred, genes = process_data(datasets, genes, hvg=hvg,
                                          dimred=dimred)

    if sketch:
        datasets_dimred = integrate_sketch(
            datasets_dimred, sketch_method=sketch_method, N=sketch_max,
            integration_fn=assemble, integration_fn_args={
                'verbose': verbose, 'knn': knn, 'sigma': sigma,
                'approx': approx, 'alpha': alpha, 'ds_names': ds_names,
                'batch_size': batch_size,
            },
        )

    else:
        datasets_dimred = assemble(
            datasets_dimred, # Assemble in low dimensional space.
            verbose=verbose, knn=knn, sigma=sigma, approx=approx,
            alpha=alpha, ds_names=ds_names, batch_size=batch_size,
        )

    return datasets_dimred, genes

# Batch correction with scanpy's AnnData object.
def correct_scanpy(adatas, **kwargs):
    """Batch correct a list of `scanpy.api.AnnData`.

    Parameters
    ----------
    adatas : `list` of `scanpy.api.AnnData`
        Data sets to integrate and/or correct.
        `adata.var_names` must be set to the list of genes.
    return_dimred : `bool`, optional (default=`False`)
        When `True`, the returned `adatas` are each modified to
        also have the integrated low-dimensional embeddings in
        `adata.obsm['X_scanorama']`.
    kwargs : `dict`
        See documentation for the `correct()` method for a full list of
        parameters to use for batch correction.

    Returns
    -------
    corrected
        By default (`return_dimred=False`), returns a list of new
        `scanpy.api.AnnData`.
        When `return_dimred=True`, `corrected` also includes the
        integrated low-dimensional embeddings in
        `adata.obsm['X_scanorama']`.
    """
    if 'return_dimred' in kwargs and kwargs['return_dimred']:
        datasets_dimred, datasets, genes = correct(
            [adata.X for adata in adatas],
            [adata.var_names.values for adata in adatas],
            **kwargs
        )
    else:
        datasets, genes = correct(
            [adata.X for adata in adatas],
            [adata.var_names.values for adata in adatas],
            **kwargs
        )

    from anndata import AnnData

    new_adatas = []
    for i in range(len((adatas))):
        adata = AnnData(datasets[i])
        adata.obs = adatas[i].obs
        adata.obsm = adatas[i].obsm

        # Ensure that variables are in the right order,
        # as Scanorama rearranges genes to be in alphabetical
        # order and as the intersection (or union) of the
        # original gene sets.
        adata.var_names = genes
        gene2idx = { gene: idx for idx, gene in
                     zip(adatas[i].var.index,
                         adatas[i].var_names.values) }
        var_idx = [ gene2idx[gene] for gene in genes ]
        adata.var = adatas[i].var.loc[var_idx]

        adata.uns = adatas[i].uns
        new_adatas.append(adata)

    if 'return_dimred' in kwargs and kwargs['return_dimred']:
        for adata, X_dimred in zip(new_adatas, datasets_dimred):
            adata.obsm['X_scanorama'] = X_dimred

    return new_adatas

# Integration with scanpy's AnnData object.
def integrate_scanpy(adatas, **kwargs):
    """Integrate a list of `scanpy.api.AnnData`.

    Parameters
    ----------
    adatas : `list` of `scanpy.api.AnnData`
        Data sets to integrate.
    kwargs : `dict`
        See documentation for the `integrate()` method for a full list of
        parameters to use for batch correction.

    Returns
    -------
    None
    """
    datasets_dimred, genes = integrate(
        [adata.X for adata in adatas],
        [adata.var_names.values for adata in adatas],
        **kwargs
    )

    for adata, X_dimred in zip(adatas, datasets_dimred):
        adata.obsm['X_scanorama'] = X_dimred

# Visualize a scatter plot with cluster labels in the
# `cluster' variable.
def plot_clusters(coords, clusters, s=1, colors=None):
    if coords.shape[0] != clusters.shape[0]:
        sys.stderr.write(
            'Error: mismatch, {} cells, {} labels\n'
            .format(coords.shape[0], clusters.shape[0])
        )
        sys.exit(1)

    if colors is None:
        colors = np.array(
            list(islice(cycle([
                '#377eb8', '#ff7f00', '#4daf4a',
                '#f781bf', '#a65628', '#984ea3',
                '#999999', '#e41a1c', '#dede00',
                '#ffe119', '#e6194b', '#ffbea3',
                '#911eb4', '#46f0f0', '#f032e6',
                '#d2f53c', '#008080', '#e6beff',
                '#aa6e28', '#800000', '#aaffc3',
                '#808000', '#ffd8b1', '#000080',
                '#808080', '#fabebe', '#a3f4ff'
            ]), int(max(clusters) + 1)))
        )

    plt.figure()
    plt.scatter(coords[:, 0], coords[:, 1],
                c=colors[clusters], s=s)

# Put datasets into a single matrix with the intersection of all genes.
def merge_datasets(datasets, genes, ds_names=None, verbose=True,
                   union=False):
    if union:
        sys.stderr.write(
            'WARNING: Integrating based on the union of genes is '
            'highly discouraged, consider taking the intersection '
            'or requantifying gene expression.\n'
        )

    # Find genes in common.
    keep_genes = set()
    for idx, gene_list in enumerate(genes):
        if len(keep_genes) == 0:
            keep_genes = set(gene_list)
        elif union:
            keep_genes |= set(gene_list)
        else:
            keep_genes &= set(gene_list)
        if not union and not ds_names is None and verbose:
            print('After {}: {} genes'.format(ds_names[idx], len(keep_genes)))
        if len(keep_genes) == 0:
            print('Error: No genes found in all datasets, exiting...')
            sys.exit(1)
    if verbose:
        print('Found {} genes among all datasets'
              .format(len(keep_genes)))

    if union:
        union_genes = sorted(keep_genes)
        for i in range(len(datasets)):
            if verbose:
                print('Processing data set {}'.format(i))
            X_new = np.zeros((datasets[i].shape[0], len(union_genes)))
            X_old = csc_matrix(datasets[i])
            gene_to_idx = { gene: idx for idx, gene in enumerate(genes[i]) }
            for j, gene in enumerate(union_genes):
                if gene in gene_to_idx:
                    X_new[:, j] = X_old[:, gene_to_idx[gene]].toarray().flatten()
            datasets[i] = csr_matrix(X_new)
        ret_genes = np.array(union_genes)
    else:
        # Only keep genes in common.
        ret_genes = np.array(sorted(keep_genes))
        for i in range(len(datasets)):
            # Remove duplicate genes.
            uniq_genes, uniq_idx = np.unique(genes[i], return_index=True)
            datasets[i] = datasets[i][:, uniq_idx]

            # Do gene filtering.
            gene_sort_idx = np.argsort(uniq_genes)
            gene_idx = [ idx for idx in gene_sort_idx
                         if uniq_genes[idx] in keep_genes ]
            datasets[i] = datasets[i][:, gene_idx]
            assert(np.array_equal(uniq_genes[gene_idx], ret_genes))

    return datasets, ret_genes

def check_datasets(datasets_full):
    datasets_new = []
    for i, ds in enumerate(datasets_full):
        if issubclass(type(ds), np.ndarray):
            datasets_new.append(csr_matrix(ds))
        elif issubclass(type(ds), csr_matrix):
            datasets_new.append(ds)
        else:
            sys.stderr.write('ERROR: Data sets must be numpy array or '
                             'scipy.sparse.csr_matrix, received type '
                             '{}.\n'.format(type(ds)))
            sys.exit(1)
    return datasets_new

# Randomized SVD.
def dimensionality_reduce(datasets, dimred=DIMRED):
    X = vstack(datasets)
    X = reduce_dimensionality(X, dim_red_k=dimred)
    datasets_dimred = []
    base = 0
    for ds in datasets:
        datasets_dimred.append(X[base:(base + ds.shape[0]), :])
        base += ds.shape[0]
    return datasets_dimred

# Normalize and reduce dimensionality.
def process_data(datasets, genes, hvg=HVG, dimred=DIMRED, verbose=False):
    # Only keep highly variable genes
    if not hvg is None and hvg > 0 and hvg < len(genes):
        if verbose:
            print('Highly variable filter...')
        X = vstack(datasets)
        disp = dispersion(X)
        highest_disp_idx = np.argsort(disp[0])[::-1]
        top_genes = set(genes[highest_disp_idx[range(hvg)]])
        for i in range(len(datasets)):
            gene_idx = [ idx for idx, g_i in enumerate(genes)
                         if g_i in top_genes ]
            datasets[i] = datasets[i][:, gene_idx]
        genes = np.array(sorted(top_genes))

    # Normalize.
    if verbose:
        print('Normalizing...')
    for i, ds in enumerate(datasets):
        datasets[i] = normalize(ds, axis=1)

    # Compute compressed embedding.
    if dimred > 0:
        if verbose:
            print('Reducing dimension...')
        datasets_dimred = dimensionality_reduce(datasets, dimred=dimred)
        if verbose:
            print('Done processing.')
        return datasets_dimred, genes

    if verbose:
        print('Done processing.')

    return datasets, genes

# Plot t-SNE visualization.
def visualize(assembled, labels, namespace, data_names,
              gene_names=None, gene_expr=None, genes=None,
              n_iter=N_ITER, perplexity=PERPLEXITY, verbose=VERBOSE,
              learn_rate=200., early_exag=12., embedding=None,
              shuffle_ds=False, size=1, multicore_tsne=True,
              image_suffix='.svg', viz_cluster=False, colors=None,
              random_state=None,):
    # Fit t-SNE.
    if embedding is None:
        try:
            from MulticoreTSNE import MulticoreTSNE
            tsne = MulticoreTSNE(
                n_iter=n_iter, perplexity=perplexity,
                verbose=verbose, random_state=random_state,
                learning_rate=learn_rate,
                early_exaggeration=early_exag,
                n_jobs=40
            )
        except ImportError:
            multicore_tsne = False

        if not multicore_tsne:
            tsne = TSNE(
                n_iter=n_iter, perplexity=perplexity,
                verbose=verbose, random_state=random_state,
                learning_rate=learn_rate,
                early_exaggeration=early_exag
            )

        tsne.fit(np.concatenate(assembled))
        embedding = tsne.embedding_

    if shuffle_ds:
        rand_idx = range(embedding.shape[0])
        random.shuffle(list(rand_idx))
        embedding = embedding[rand_idx, :]
        labels = labels[rand_idx]

    # Plot clusters together.
    plot_clusters(embedding, labels, s=size, colors=colors)
    plt.title(('Panorama ({} iter, perplexity: {}, sigma: {}, ' +
               'knn: {}, hvg: {}, dimred: {}, approx: {})')
              .format(n_iter, perplexity, SIGMA, KNN, HVG,
                      DIMRED, APPROX))
    plt.savefig(namespace + image_suffix, dpi=500)

    # Plot clusters individually.
    if viz_cluster and not shuffle_ds:
        for i in range(len(data_names)):
            visualize_cluster(embedding, i, labels,
                              cluster_name=data_names[i], size=size,
                              viz_prefix=namespace,
                              image_suffix=image_suffix)

    # Plot gene expression levels.
    if (not gene_names is None) and \
       (not gene_expr is None) and \
       (not genes is None):
        if shuffle_ds:
            gene_expr = gene_expr[rand_idx, :]
        for gene_name in gene_names:
            visualize_expr(gene_expr, embedding,
                           genes, gene_name, size=size,
                           viz_prefix=namespace,
                           image_suffix=image_suffix)

    return embedding

# Exact nearest neighbors search.
def nn(ds1, ds2, knn=KNN, metric_p=2):
    # Find nearest neighbors of first dataset.
    nn_ = NearestNeighbors(n_neighbors=knn, p=metric_p)
    nn_.fit(ds2)
    ind = nn_.kneighbors(ds1, return_distance=False)

    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((a, b_i))

    return match

# Approximate nearest neighbors using locality sensitive hashing.
def nn_approx(ds1, ds2, knn=KNN, metric='manhattan', n_trees=10):
    # Build index.
    a = AnnoyIndex(ds2.shape[1], metric=metric)
    for i in range(ds2.shape[0]):
        a.add_item(i, ds2[i, :])
    a.build(n_trees)

    # Search index.
    ind = []
    for i in range(ds1.shape[0]):
        ind.append(a.get_nns_by_vector(ds1[i, :], knn, search_k=-1))
    ind = np.array(ind)

    # Match.
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((a, b_i))

    return match

# Find mutual nearest neighbors.
def mnn(ds1, ds2, knn=KNN, approx=APPROX):
    # Find nearest neighbors in first direction.
    if approx:
        match1 = nn_approx(ds1, ds2, knn=knn)
    else:
        match1 = nn(ds1, ds2, knn=knn)

    # Find nearest neighbors in second direction.
    if approx:
        match2 = nn_approx(ds2, ds1, knn=knn)
    else:
        match2 = nn(ds2, ds1, knn=knn)

    # Compute mutual nearest neighbors.
    mutual = match1 & set([ (b, a) for a, b in match2 ])

    return mutual

# Visualize alignment between two datasets.
def plot_mapping(curr_ds, curr_ref, ds_ind, ref_ind):
    tsne = TSNE(n_iter=400, verbose=VERBOSE, random_state=69)

    tsne.fit(curr_ds)
    plt.figure()
    coords_ds = tsne.embedding_[:, :]
    coords_ds[:, 1] += 100
    plt.scatter(coords_ds[:, 0], coords_ds[:, 1])

    tsne.fit(curr_ref)
    coords_ref = tsne.embedding_[:, :]
    plt.scatter(coords_ref[:, 0], coords_ref[:, 1])

    x_list, y_list = [], []
    for r_i, c_i in zip(ds_ind, ref_ind):
        x_list.append(coords_ds[r_i, 0])
        x_list.append(coords_ref[c_i, 0])
        x_list.append(None)
        y_list.append(coords_ds[r_i, 1])
        y_list.append(coords_ref[c_i, 1])
        y_list.append(None)
    plt.plot(x_list, y_list, 'b-', alpha=0.3)
    plt.show()


# Populate a table (in place) that stores mutual nearest neighbors
# between datasets.
def fill_table(table, i, curr_ds, datasets, base_ds=0,
               knn=KNN, approx=APPROX):
    curr_ref = np.concatenate(datasets)
    if approx:
        match = nn_approx(curr_ds, curr_ref, knn=knn)
    else:
        match = nn(curr_ds, curr_ref, knn=knn, metric_p=1)

    # Build interval tree.
    itree_ds_idx = IntervalTree()
    itree_pos_base = IntervalTree()
    pos = 0
    for j in range(len(datasets)):
        n_cells = datasets[j].shape[0]
        itree_ds_idx[pos:(pos + n_cells)] = base_ds + j
        itree_pos_base[pos:(pos + n_cells)] = pos
        pos += n_cells

    # Store all mutual nearest neighbors between datasets.
    for d, r in match:
        interval = itree_ds_idx[r]
        assert(len(interval) == 1)
        j = interval.pop().data
        interval = itree_pos_base[r]
        assert(len(interval) == 1)
        base = interval.pop().data
        if not (i, j) in table:
            table[(i, j)] = set()
        table[(i, j)].add((d, r - base))
        assert(r - base >= 0)

gs_idxs = None

# Fill table of alignment scores.
def find_alignments_table(datasets, knn=KNN, approx=APPROX, verbose=VERBOSE,
                          prenormalized=False):
    if not prenormalized:
        datasets = [ normalize(ds, axis=1) for ds in datasets ]

    table = {}
    for i in range(len(datasets)):
        if len(datasets[:i]) > 0:
            fill_table(table, i, datasets[i], datasets[:i], knn=knn,
                       approx=approx)
        if len(datasets[i+1:]) > 0:
            fill_table(table, i, datasets[i], datasets[i+1:],
                       knn=knn, base_ds=i+1, approx=approx)
    # Count all mutual nearest neighbors between datasets.
    matches = {}
    table1 = {}
    if verbose > 1:
        table_print = np.zeros((len(datasets), len(datasets)))
    for i in range(len(datasets)):
        for j in range(len(datasets)):
            if i >= j:
                continue
            if not (i, j) in table or not (j, i) in table:
                continue
            match_ij = table[(i, j)]
            match_ji = set([ (b, a) for a, b in table[(j, i)] ])
            matches[(i, j)] = match_ij & match_ji

            table1[(i, j)] = (max(
                float(len(set([ idx for idx, _ in matches[(i, j)] ]))) /
                datasets[i].shape[0],
                float(len(set([ idx for _, idx in matches[(i, j)] ]))) /
                datasets[j].shape[0]
            ))
            if verbose > 1:
                table_print[i, j] += table1[(i, j)]

    if verbose > 1:
        print(table_print)
        return table1, table_print, matches
    else:
        return table1, None, matches

# Find the matching pairs of cells between datasets.
def find_alignments(datasets, knn=KNN, approx=APPROX, verbose=VERBOSE,
                    alpha=ALPHA, prenormalized=False,):
    table1, _, matches = find_alignments_table(
        datasets, knn=knn, approx=approx, verbose=verbose,
        prenormalized=prenormalized,
    )

    alignments = [ (i, j) for (i, j), val in reversed(
        sorted(table1.items(), key=operator.itemgetter(1))
    ) if val > alpha ]

    return alignments, matches

# Find connections between datasets to identify panoramas.
def connect(datasets, knn=KNN, approx=APPROX, alpha=ALPHA,
            verbose=VERBOSE):
    # Find alignments.
    alignments, _ = find_alignments(
        datasets, knn=knn, approx=approx, alpha=alpha,
        verbose=verbose
    )
    if verbose:
        print(alignments)

    panoramas = []
    connected = set()
    for i, j in alignments:
        # See if datasets are involved in any current panoramas.
        panoramas_i = [ panoramas[p] for p in range(len(panoramas))
                        if i in panoramas[p] ]
        assert(len(panoramas_i) <= 1)
        panoramas_j = [ panoramas[p] for p in range(len(panoramas))
                        if j in panoramas[p] ]
        assert(len(panoramas_j) <= 1)

        if len(panoramas_i) == 0 and len(panoramas_j) == 0:
            panoramas.append([ i ])
            panoramas_i = [ panoramas[-1] ]

        if len(panoramas_i) == 0:
            panoramas_j[0].append(i)
        elif len(panoramas_j) == 0:
            panoramas_i[0].append(j)
        elif panoramas_i[0] != panoramas_j[0]:
            panoramas_i[0] += panoramas_j[0]
            panoramas.remove(panoramas_j[0])

        connected.add(i)
        connected.add(j)

    for i in range(len(datasets)):
        if not i in connected:
            panoramas.append([ i ])

    return panoramas

# To reduce memory usage, split bias computation into batches.
# Robust, density-adaptive, ridge-shrunk kernel smoother for bias field.
def batch_bias(curr_ds, match_ds, bias, batch_size=None, sigma=SIGMA,
               adaptive=True, k_sigma=30, lambda_shrink=1.0,
               robust=True, huber_c=2.5, knn_anchor=200):
    """Estimate per-cell bias using a conservative kernel smoother.

    This replaces the previous normalized RBF averaging with a ridge-shrunk
    smoother:
        b_hat(x) = sum_i w_i(x) * b_i / (lambda + sum_i w_i(x))

    Additionally supports:
      - density-adaptive bandwidth sigma(x) via k_sigma-th NN distance to anchors
      - robust anchor weights (Huber) based on ||b_i||
      - optional anchor subsampling via kNN anchors per query (reduces smoothing)
    """
    # Ensure dense arrays for distance computations.
    curr = curr_ds.toarray() if scipy.sparse.issparse(curr_ds) else np.asarray(curr_ds)
    anch = match_ds.toarray() if scipy.sparse.issparse(match_ds) else np.asarray(match_ds)

    # Bias may be sparse if provided from sparse subtraction.
    B = bias.toarray() if scipy.sparse.issparse(bias) else np.asarray(bias)

    n_query = curr.shape[0]
    n_anchor = anch.shape[0]

    # Robust reweighting of anchors based on bias magnitude (Huber).
    anchor_w = np.ones((n_anchor,), dtype=float)
    if robust and n_anchor > 0:
        bnorm = np.linalg.norm(B, axis=1)
        med = np.median(bnorm)
        mad = np.median(np.abs(bnorm - med)) + 1e-12
        s = 1.4826 * mad + 1e-12
        r = bnorm / (huber_c * s)
        # Huber weights: 1 for inliers, 1/r for outliers.
        anchor_w = np.ones_like(r)
        out = r > 1.0
        anchor_w[out] = 1.0 / r[out]

    # Optional: restrict anchors per query to kNN for speed and reduced oversmoothing.
    use_knn = knn_anchor is not None and knn_anchor > 0 and knn_anchor < n_anchor
    if use_knn:
        nn_ = NearestNeighbors(n_neighbors=min(knn_anchor, n_anchor), p=2)
        nn_.fit(anch)
        neigh_ind = nn_.kneighbors(curr, return_distance=False)
    else:
        neigh_ind = None

    # Local adaptive bandwidth per query via k_sigma-th neighbor distance to anchors.
    if adaptive and n_anchor > 1:
        k_eff = min(max(2, k_sigma), n_anchor)
        # If we already have kNN anchors, use them to estimate local scale.
        if neigh_ind is not None and neigh_ind.shape[1] >= k_eff:
            neigh_for_sigma = neigh_ind[:, :k_eff]
            d2 = np.sum((curr[:, None, :] - anch[neigh_for_sigma, :]) ** 2, axis=2)
            # distance to k_eff-th neighbor
            sig = np.sqrt(np.partition(d2, k_eff - 1, axis=1)[:, k_eff - 1]) + 1e-12
        else:
            d2_full = euclidean_distances(curr, anch, squared=True)
            sig = np.sqrt(np.partition(d2_full, k_eff - 1, axis=1)[:, k_eff - 1]) + 1e-12
    else:
        # Interpret sigma as a global length-scale in the exp(-d^2/(2*sigma^2)) kernel.
        # If sigma passed in is <= 0, fall back to a small positive value.
        sig_val = float(sigma)
        if not np.isfinite(sig_val) or sig_val <= 0.0:
            sig_val = 1.0
        sig = np.full((n_query,), sig_val, dtype=float)

    avg_bias = np.zeros((n_query, B.shape[1]), dtype=float)

    if batch_size is None:
        if neigh_ind is None:
            d2 = euclidean_distances(curr, anch, squared=True)
            W = np.exp(-d2 / (2.0 * (sig[:, None] ** 2)))
            W *= anchor_w[None, :]
            numer = W.dot(B)
            denom = lambda_shrink + np.sum(W, axis=1)
            denom = handle_zeros_in_scale(denom, copy=False)
            avg_bias = numer / denom[:, None]
            return avg_bias

        # kNN anchor path.
        denom = np.zeros((n_query,), dtype=float)
        for q in range(n_query):
            idx = neigh_ind[q]
            d2 = np.sum((curr[q, None, :] - anch[idx, :]) ** 2, axis=1)
            w = np.exp(-d2 / (2.0 * (sig[q] ** 2)))
            w *= anchor_w[idx]
            avg_bias[q, :] = (w[:, None] * B[idx, :]).sum(axis=0)
            denom[q] = w.sum()
        denom = lambda_shrink + denom
        denom = handle_zeros_in_scale(denom, copy=False)
        avg_bias /= denom[:, None]
        return avg_bias

    # Batched path (only for full-anchor mode).
    if neigh_ind is not None:
        # For simplicity, ignore batch_size when using kNN anchors (already sparse).
        denom = np.zeros((n_query,), dtype=float)
        for q in range(n_query):
            idx = neigh_ind[q]
            d2 = np.sum((curr[q, None, :] - anch[idx, :]) ** 2, axis=1)
            w = np.exp(-d2 / (2.0 * (sig[q] ** 2)))
            w *= anchor_w[idx]
            avg_bias[q, :] = (w[:, None] * B[idx, :]).sum(axis=0)
            denom[q] = w.sum()
        denom = lambda_shrink + denom
        denom = handle_zeros_in_scale(denom, copy=False)
        avg_bias /= denom[:, None]
        return avg_bias

    base = 0
    denom = np.zeros((n_query,), dtype=float)
    while base < n_anchor:
        batch_idx = range(base, min(base + batch_size, n_anchor))
        anch_b = anch[batch_idx, :]
        d2 = euclidean_distances(curr, anch_b, squared=True)
        W = np.exp(-d2 / (2.0 * (sig[:, None] ** 2)))
        W *= anchor_w[np.array(list(batch_idx))][None, :]
        avg_bias += W.dot(B[np.array(list(batch_idx)), :])
        denom += np.sum(W, axis=1)
        base += batch_size

    denom = lambda_shrink + denom
    denom = handle_zeros_in_scale(denom, copy=False)
    avg_bias /= denom[:, np.newaxis]

    return avg_bias

# Compute nonlinear translation vectors between dataset
# and a reference.
def transform(curr_ds, curr_ref, ds_ind, ref_ind, sigma=SIGMA, cn=False,
              batch_size=None):
    # Compute the matching.
    match_ds = curr_ds[ds_ind, :]
    match_ref = curr_ref[ref_ind, :]
    bias = match_ref - match_ds
    if cn:
        match_ds = match_ds.toarray()
        curr_ds = curr_ds.toarray()
        bias = bias.toarray()

    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=RuntimeWarning)
        try:
            # Density-adaptive, ridge-shrunk, robust bias field.
            avg_bias = batch_bias(
                curr_ds, match_ds, bias,
                sigma=sigma, batch_size=batch_size,
                adaptive=True, k_sigma=max(10, min(30, len(ds_ind) if len(ds_ind) > 0 else 30)),
                lambda_shrink=1.0, robust=True, huber_c=2.5,
                knn_anchor=min(200, match_ds.shape[0]) if match_ds.shape[0] > 0 else None,
            )
        except RuntimeWarning:
            sys.stderr.write('WARNING: Oversmoothing detected, refusing to batch '
                             'correct, consider lowering sigma value.\n')
            return csr_matrix(curr_ds.shape, dtype=float)
        except MemoryError:
            if batch_size is None:
                sys.stderr.write('WARNING: Out of memory, consider turning on '
                                 'batched computation with batch_size parameter.\n')
            else:
                sys.stderr.write('WARNING: Out of memory, consider lowering '
                                 'the batch_size parameter.\n')
            return csr_matrix(curr_ds.shape, dtype=float)

    if cn:
        avg_bias = csr_matrix(avg_bias)

    return avg_bias

# Finds alignments between datasets and uses them to construct
# panoramas. "Merges" datasets by correcting gene expression
# values.
def assemble(datasets, verbose=VERBOSE, view_match=False, knn=KNN,
             sigma=SIGMA, approx=APPROX, alpha=ALPHA, expr_datasets=None,
             ds_names=None, batch_size=None,
             alignments=None, matches=None):
    if len(datasets) == 1:
        return datasets

    if alignments is None and matches is None:
        alignments, matches = find_alignments(
            datasets, knn=knn, approx=approx, alpha=alpha, verbose=verbose,
        )

    # --- Global, graph-based alignment (order-independent) + refinement ---
    # Build connected components over the alignment graph; integrate each component.
    n_ds = len(datasets)
    adj = [set() for _ in range(n_ds)]
    for i, j in alignments:
        adj[i].add(j)
        adj[j].add(i)

    visited = set()
    components = []
    for i in range(n_ds):
        if i in visited:
            continue
        stack = [i]
        comp = []
        visited.add(i)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        components.append(sorted(comp))

    # Helper: solve global translations t_i via least squares on all MNN pairs
    # within a component, then apply them to datasets in-place.
    def _global_translation_init(comp, ridge=1e-3):
        if len(comp) <= 1:
            return

        # Map dataset index -> local variable index
        idx_map = {ds_idx: k for k, ds_idx in enumerate(comp)}
        d = datasets[comp[0]].shape[1]
        m = len(comp)

        # Accumulate linear system: (A^T A + ridge I) t = A^T y
        # Each match contributes (t_i - t_j) = (x_b^j - x_a^i)
        L = np.zeros((m, m), dtype=float)
        B = np.zeros((m, d), dtype=float)

        edges_used = 0
        for (i, j) in alignments:
            if i not in idx_map or j not in idx_map:
                continue
            # Use matches in consistent orientation.
            if (i, j) in matches:
                pairs = list(matches[(i, j)])
                Xi = datasets[i]
                Xj = datasets[j]
                # rhs is x_j - x_i
                rhs = Xj[[b for _, b in pairs], :] - Xi[[a for a, _ in pairs], :]
            elif (j, i) in matches:
                pairs = list(matches[(j, i)])
                Xi = datasets[i]
                Xj = datasets[j]
                # matches[(j,i)] is (a in j, b in i); need x_j - x_i
                rhs = Xj[[a for a, _ in pairs], :] - Xi[[b for _, b in pairs], :]
            else:
                continue

            if rhs.shape[0] == 0:
                continue

            rhs_mean = np.mean(rhs, axis=0)
            ii = idx_map[i]
            jj = idx_map[j]
            L[ii, ii] += 1.0
            L[jj, jj] += 1.0
            L[ii, jj] -= 1.0
            L[jj, ii] -= 1.0
            B[ii, :] += rhs_mean
            B[jj, :] -= rhs_mean
            edges_used += 1

        if edges_used == 0:
            return

        # Anchor one dataset (largest) to remove gauge freedom.
        root = max(comp, key=lambda k: datasets[k].shape[0])
        root_i = idx_map[root]

        # Add ridge and strong anchor constraint for root: t_root = 0
        M = L + ridge * np.eye(m)
        M[root_i, :] = 0.0
        M[:, root_i] = 0.0
        M[root_i, root_i] = 1.0
        B[root_i, :] = 0.0

        try:
            T = np.linalg.solve(M, B)  # (m, d)
        except np.linalg.LinAlgError:
            return

        for ds_idx in comp:
            t = T[idx_map[ds_idx], :]
            datasets[ds_idx] = np.asarray(datasets[ds_idx] + t)

    # Apply global translation initialization per component.
    for comp in components:
        _global_translation_init(comp, ridge=1e-3)

    # Iterative refinement: recompute MNNs after initial alignment and apply
    # conservative nonlinear corrections using the robust, adaptive transform().
    # A small number of rounds reduces order effects without heavy cost.
    n_rounds = 2
    for r in range(n_rounds):
        if verbose:
            print('Refinement round {}'.format(r + 1))

        # Recompute alignments/matches on current (partially corrected) coordinates.
        alignments_r, matches_r = find_alignments(
            datasets, knn=knn, approx=approx, alpha=alpha, verbose=0,
            prenormalized=True,
        )

        for (i, j) in alignments_r:
            # Apply one-sided correction toward the larger dataset to reduce drift.
            if datasets[i].shape[0] >= datasets[j].shape[0]:
                src, ref = j, i
            else:
                src, ref = i, j

            # Build indices for src->ref matches.
            if (src, ref) in matches_r:
                pairs = list(matches_r[(src, ref)])
                ds_ind = [a for a, _ in pairs]
                ref_ind = [b for _, b in pairs]
            elif (ref, src) in matches_r:
                pairs = list(matches_r[(ref, src)])
                ds_ind = [b for _, b in pairs]
                ref_ind = [a for a, _ in pairs]
            else:
                continue

            if len(ds_ind) == 0:
                continue

            if verbose > 1:
                if ds_names is None:
                    print('Refining {} -> {}'.format(src, ref))
                else:
                    print('Refining {} -> {}'.format(ds_names[src], ds_names[ref]))

            bias = transform(
                datasets[src], datasets[ref],
                ds_ind, ref_ind, sigma=sigma, batch_size=batch_size
            )
            datasets[src] = np.asarray(datasets[src] + bias)

            if expr_datasets:
                # Apply correction to expression space using same anchors.
                bias_expr = transform(
                    expr_datasets[src], expr_datasets[ref],
                    ds_ind, ref_ind, sigma=sigma, cn=True, batch_size=batch_size
                )
                expr_datasets[src] = expr_datasets[src] + bias_expr

            if view_match:
                plot_mapping(datasets[src], datasets[ref], ds_ind, ref_ind)

    return datasets

# Sketch-based acceleration of integration.
def integrate_sketch(datasets_dimred, sketch_method='geosketch', N=10000,
                     integration_fn=assemble, integration_fn_args={}):

    from geosketch import gs, uniform

    if sketch_method.lower() == 'geosketch' or sketch_method.lower() == 'gs':
        sampling_fn = gs
    else:
        sampling_fn = uniform

    # Sketch each dataset.
    sketch_idxs = [
        sorted(set(sampling_fn(X, N, replace=False)))
        if X.shape[0] > N else list(range(X.shape[0]))
        for X in datasets_dimred
    ]
    datasets_sketch = [ X[idx] for X, idx in zip(datasets_dimred, sketch_idxs) ]

    # Integrate the dataset sketches.
    datasets_int = integration_fn(datasets_sketch[:], **integration_fn_args)

    # Apply integrated coordinates back to full data.
    for i, (X_dimred, X_sketch) in enumerate(zip(datasets_dimred, datasets_sketch)):
        X_int = datasets_int[i]

        neigh = NearestNeighbors(n_neighbors=3).fit(X_dimred)
        _, neigh_idx = neigh.kneighbors(X_sketch)

        ds_idxs, ref_idxs = [], []
        for ref_idx in range(neigh_idx.shape[0]):
            for k_idx in range(neigh_idx.shape[1]):
                ds_idxs.append(neigh_idx[ref_idx, k_idx])
                ref_idxs.append(ref_idx)

        bias = transform(X_dimred, X_int, ds_idxs, ref_idxs, 15, batch_size=1000)

        datasets_int[i] = X_dimred + bias

    return datasets_int

# Non-optimal dataset assembly. Simply accumulate datasets into a
# reference.
def assemble_accum(datasets, verbose=VERBOSE, knn=KNN, sigma=SIGMA,
                   approx=APPROX, batch_size=None):
    if len(datasets) == 1:
        return datasets

    for i in range(len(datasets) - 1):
        j = i + 1

        if verbose:
            print('Processing datasets {}'.format((i, j)))

        ds1 = datasets[j]
        ds2 = np.concatenate(datasets[:i+1])
        match = mnn(ds1, ds2, knn=knn, approx=approx)

        ds_ind = [ a for a, _ in match ]
        ref_ind = [ b for _, b in match ]

        bias = transform(ds1, ds2, ds_ind, ref_ind, sigma=sigma,
                         batch_size=batch_size)
        datasets[j] = np.asarray(ds1 + bias)

    return datasets
