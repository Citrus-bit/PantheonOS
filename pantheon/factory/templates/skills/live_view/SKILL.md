---
id: live_view_index
name: LiveView Skills Index
description: |
  Skills for opening and driving agent-controllable visualization
  components in the Pantheon UI sidebar — interactive viewers the agent
  can open, control, and read back. Viewers: Vitessce (spatial / single-
  cell omics), Viv (bioimage / microscopy), plus agent-generated apps.
tags: [live-view, visualization, vitessce, viv, bioimage, spatial, single-cell, interactive]
---

# LiveView — Agent-Controllable UI Components

A **LiveView** is an interactive component the agent opens in the Pantheon
UI's right sidebar, then drives and observes through the `live_view` tools.
Unlike a static plot image, a LiveView is live: the agent changes its state
and the user sees it update; the user interacts with it and the agent reads
the result back.

Load the relevant skill file before building a visualization.

**Architecture** — only the LiveView SDK runtime is built in. Every viewer
is a **plugin**: a folder `skills/live_view/<name>/` holding `<name>.md`
(this guide), `adapter.js` (a `setup(lv, root)` module), and an optional
`demo.json`. `open_live_view` either resolves a named viewer plugin
(`view_type="vitessce"`) or loads an agent-generated component
(`view_type="custom"` + `module_url`). Adding a viewer = dropping a new
`<name>/` folder here; no app code changes.

## Available skills

### Vitessce — spatial / single-cell / imaging data

Open a Vitessce browser to explore spatial transcriptomics, single-cell, and
microscopy-imaging datasets: spatial scatterplots, gene-expression coloring,
heatmaps, cell-set selection, image layers.

**Skill file**: [vitessce/vitessce.md](./vitessce/vitessce.md)

**When to use**:
- Visualizing spatial transcriptomics (10x Visium, Xenium, MERFISH, …)
- Single-cell data with embeddings (UMAP/t-SNE) the user should explore
- Cell segmentations as **interactive** objects — click/hover a cell,
  colour by type or gene. (Just *viewing* boundaries on an image → use Viv;
  Vitessce does not render clean boundaries.)
- Spatial omics where cells/sets/embeddings matter, not just the image

### Viv — bioimage / microscopy viewer

Open a Viv viewer for high-resolution, multiplexed bioimaging — OME-TIFF
and OME-Zarr (OME-NGFF): multichannel fluorescence, microscopy, IF/IMC/
CODEX, whole-slide images. Channel colors, contrast, pan/zoom, overview.

**Skill file**: [viv/viv.md](./viv/viv.md)

**When to use**:
- The data *is an image* — OME-TIFF / OME-Zarr microscopy
- Multichannel fluorescence the user wants to recolour / adjust
- Cloud-hosted or local bioimages (served via `serve_local_data`)
- **Overlaying a cell segmentation / showing cell boundaries** on an image
  (boundaries as an extra channel — the preferred way to *view* a mask)

### Mol* — 3D molecular structures

Open a Mol* viewer for 3D macromolecular structures — proteins, nucleic
acids, complexes — from the RCSB PDB, the AlphaFold DB, or a local
`.pdb` / `.cif` file. Rotate, zoom, inspect; AlphaFold models colour by
pLDDT.

**Skill file**: [molstar/molstar.md](./molstar/molstar.md)

**When to use**:
- Showing a protein / nucleic-acid 3D structure (experimental or predicted)
- Visualising an AlphaFold prediction
- Any `.pdb` / `.cif` structure file (see also the `structural_biology` skill
  for obtaining / predicting structures)

### Generate a custom LiveView app

Write your own interactive component with the LiveView SDK when no existing
viewer fits — a bespoke dashboard, custom plot, or tailored data view that
the agent can still open, drive, and observe.

**Skill file**: [live-view-app.md](./live-view-app.md)

**When to use**:
- The data / interaction doesn't match a ready-made viewer
- You need a tailored view of analysis output
- You want a custom interactive control surface for the user

## The `live_view` tools

| Tool | Purpose |
|------|---------|
| `open_live_view(view_type, title, state, module_url?)` | Open a viewer plugin (e.g. `view_type="vitessce"`) or a custom component (`view_type="custom"` + `module_url`); returns `view_id` |
| `serve_local_data(path)` | Expose a workspace file/dir over HTTP+CORS; returns a fetchable URL |
| `live_view_update(view_id, patch)` | Deep-merge a partial-state patch (drive it) |
| `live_view_set_state(view_id, state)` | Replace the whole state |
| `live_view_get_state(view_id)` | Read state, `status`, and `diagnostics` — incl. the user's own edits |
| `live_view_call(view_id, action, args)` | Invoke a component-defined action |
| `live_view_screenshot(view_id)` | Render the view to an image — `observe_images` it to see it |
| `list_live_views()` / `close_live_view(view_id)` | List / close |

Workflow: `open_live_view` → **verify** (`live_view_get_state` for
`diagnostics`, `live_view_screenshot` to see it — `status: ready` does NOT
mean it rendered correctly) → drive with `live_view_update` →
`live_view_get_state` before the next move. Never treat reading back your
own `live_view_update` value as verification.
