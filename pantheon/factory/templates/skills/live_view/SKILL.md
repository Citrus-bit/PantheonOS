---
id: live_view_index
name: LiveView Skills Index
description: |
  Skills for opening and driving agent-controllable visualization
  components in the Pantheon UI sidebar — interactive viewers the agent
  can open, control, and read back. Currently: Vitessce (spatial / single-
  cell / imaging data).
tags: [live-view, visualization, vitessce, spatial, single-cell, interactive]
---

# LiveView — Agent-Controllable UI Components

A **LiveView** is an interactive component the agent opens in the Pantheon
UI's right sidebar, then drives and observes through the `live_view` tools.
Unlike a static plot image, a LiveView is live: the agent changes its state
and the user sees it update; the user interacts with it and the agent reads
the result back.

Load the relevant skill file before building a visualization.

## Available skills

### Vitessce — spatial / single-cell / imaging data

Open a Vitessce browser to explore spatial transcriptomics, single-cell, and
microscopy-imaging datasets: spatial scatterplots, gene-expression coloring,
heatmaps, cell-set selection, image layers.

**Skill file**: [vitessce.md](./vitessce.md)

**When to use**:
- Visualizing spatial transcriptomics (10x Visium, Xenium, MERFISH, …)
- Single-cell data with embeddings (UMAP/t-SNE) the user should explore
- Microscopy images, optionally with segmentation overlays
- A ready-made domain viewer fits the data

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
| `open_live_view(view_type, title, state)` | Open a component; returns `view_id` |
| `live_view_update(view_id, patch)` | Deep-merge a partial-state patch (drive it) |
| `live_view_set_state(view_id, state)` | Replace the whole state |
| `live_view_get_state(view_id)` | Read current state — incl. the user's own edits |
| `live_view_call(view_id, action, args)` | Invoke a component-defined action |
| `list_live_views()` / `close_live_view(view_id)` | List / close |

Workflow: `open_live_view` → drive with `live_view_update` → before deciding
the next move, `live_view_get_state` to see where the view (and the user)
currently is.
