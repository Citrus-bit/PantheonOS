---
id: vitessce_live_view
name: Visualize Data with Vitessce LiveView
description: |
  Open and drive an interactive Vitessce browser (spatial transcriptomics,
  single-cell, microscopy imaging) in the Pantheon sidebar via the
  live_view tools. Covers building a valid view config, the coordination
  model used to drive the view, and how to avoid the common failure of
  invented data URLs.
tags: [vitessce, spatial, single-cell, visium, xenium, ome-zarr, anndata, live-view]
---

# Visualizing Data with Vitessce

Vitessce is an interactive browser for spatial transcriptomics, single-cell,
and microscopy-imaging data. You open it as a LiveView and then drive it.

## How Vitessce works (important)

Vitessce is a **frontend-only** library — it has **no backend server**. It
reads data **directly over HTTP** from the URLs in its *view config*. Data
must be in a chunked format (**Zarr**: AnnData-Zarr, OME-Zarr; or OME-TIFF,
CSV) hosted somewhere the browser can fetch with CORS.

⚠️ **The #1 failure mode: invented data URLs.** Do NOT guess or hand-type
data file URLs — fabricated URLs 404 and the view shows nothing. Only ever
use URLs that are either (a) a published Vitessce example config you
actually retrieved, or (b) produced by the `vitessce` Python package from
data you actually converted.

## The workflow

```
1. Get/build a valid Vitessce view config (see below)
2. open_live_view("vitessce", title, state=config)   → view_id
3. drive it:  live_view_update(view_id, patch)
4. observe:   live_view_get_state(view_id)            (incl. the user's edits)
```

For Vitessce the LiveView **state IS the Vitessce view config**. A
`live_view_update` patch is **deep-merged** into the config — almost always
into `coordinationSpace` (see "Driving the view" below).

## Building a config — use the `vitessce` Python package

The reliable way to produce a valid config is the `vitessce` Python package
(run it in `python_interpreter`). It knows the schema and will not typo URLs.

```python
# pip install vitessce  (if missing)
from vitessce import VitessceConfig, AnnDataWrapper, Component as cm

vc = VitessceConfig(schema_version="1.0.16", name="My dataset")
dataset = vc.add_dataset(name="data").add_object(
    AnnDataWrapper(
        # for remote data: adata_url=...   for local: adata_path=... (needs a server)
        adata_url="https://<host>/data.h5ad.zarr",
        obs_embedding_paths=["obsm/X_umap"],
        obs_embedding_names=["UMAP"],
        obs_set_paths=["obs/cell_type"],
        obs_set_names=["Cell Type"],
        obs_feature_matrix_path="X",
    )
)
scatter = vc.add_view(cm.SCATTERPLOT, dataset=dataset, mapping="UMAP")
sets    = vc.add_view(cm.OBS_SETS, dataset=dataset)
genes   = vc.add_view(cm.FEATURE_LIST, dataset=dataset)
heatmap = vc.add_view(cm.HEATMAP, dataset=dataset)
vc.layout((scatter | sets) / (genes | heatmap))
config = vc.to_dict(base_url="https://<host>")   # -> pass this to open_live_view
```

`vc.to_dict()` returns a fully valid config dict. Pass it straight to
`open_live_view("vitessce", title, state=config)`.

## View config structure (for understanding / hand-editing)

```jsonc
{
  "version": "1.0.16",
  "name": "...",
  "initStrategy": "auto",          // "auto" fills in obvious coordinations
  "datasets": [{
    "uid": "ds",
    "files": [{
      "fileType": "anndata.zarr",   // see fileTypes below
      "url": "https://.../data.zarr",
      "options": { /* which obsm/obs/X paths to read */ },
      "coordinationValues": { "obsType": "cell" }
    }]
  }],
  "coordinationSpace": { /* the live state — see Driving the view */ },
  "layout": [{
    "component": "spatial",         // spatial | scatterplot | heatmap |
                                    // obsSets | featureList | layerController |
                                    // description | status
    "x": 0, "y": 0, "w": 6, "h": 12, // 12-column grid
    "coordinationScopes": { "dataset": "ds" }
  }]
}
```

Common `fileType`s: `anndata.zarr`, `obsEmbedding.csv`, `image.ome-zarr`,
`obsSegmentations.json`, `obsFeatureMatrix.anndata.zarr`. (The `vitessce`
package picks these for you.)

## Driving the view — the coordination model

`coordinationSpace` is the live state. Every view is linked to it; changing
a value updates all linked views. Drive it with `live_view_update`, patching
into `coordinationSpace`. Each coordination type is scoped — the default
scope is `"A"`.

```python
# zoom a spatial view
live_view_update(view_id, {"coordinationSpace": {"spatialZoom": {"A": 4}}})

# pan
live_view_update(view_id, {"coordinationSpace": {
    "spatialTargetX": {"A": 1200}, "spatialTargetY": {"A": 900}}})

# color cells by a gene's expression
live_view_update(view_id, {"coordinationSpace": {
    "obsColorEncoding": {"A": "geneSelection"},
    "featureSelection": {"A": ["CD3D"]}}})

# select a cell set
live_view_update(view_id, {"coordinationSpace": {
    "obsSetSelection": {"A": [["Cell Type", "T cell"]]}}})
```

Useful coordination types: `spatialZoom`, `spatialTargetX`, `spatialTargetY`,
`spatialRotation`, `embeddingZoom`, `embeddingTargetX`, `embeddingTargetY`,
`obsColorEncoding` (`"cellSetSelection"` | `"geneSelection"`),
`featureSelection` (list of gene names), `obsSetSelection`, `obsHighlight`.

Always `live_view_get_state(view_id)` before the next move — it reflects
changes the **user** made by interacting with the view directly.

## Visualizing the user's own data

Vitessce needs the data as Zarr served over HTTP+CORS:

1. Convert with `python_interpreter`: AnnData `adata.write_zarr(path)`;
   images → OME-Zarr (`bioformats2raw` / `ome-zarr-py`).
2. Serve it with CORS so the browser can fetch it, and get a base URL.
   *(If a `serve_local_data` / data-server tool is available, use it; it
   turns a workspace path into a fetchable URL. Until then, only remote /
   public datasets work.)*
3. Build the config with the `vitessce` package, `base_url` = the served URL.

## Quick public-data demo

To demo without the user's own data, retrieve a **published** Vitessce
example config (e.g. from the Vitessce examples) and pass it to
`open_live_view`. Do not assemble data URLs yourself.

## Checklist before open_live_view

- [ ] config came from the `vitessce` package OR a retrieved published config
- [ ] every `files[].url` is real (converted-and-served, or verified public)
- [ ] `version` and `initStrategy` set; `layout` has at least one component
