"""
Tabula Adapter

Upstream: https://github.com/aristoteleo/tabula

This adapter focuses on Tabula PERTURB (gene perturbation prediction) using a
fine-tuned checkpoint produced by Tabula's GEARS perturbation training.

Checkpoint layout expected (directory pointed to by SCFM_CHECKPOINT_DIR_TABULA
or passed as checkpoint_dir):
  - best_model.pth
  - genes.json
  - gene_ids.npy
  - meta.json (optional)

Notes:
  - Upstream Tabula uses FlashAttention weights (Wqkv/out_proj). If the
    flash-attn package is not installed, upstream Tabula's attention module
    must provide a compatible fallback (we patch vendor tabula to do so).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..registry import TaskType, get_registry
from .base import BaseAdapter


def _check_tabula_importable() -> tuple[bool, Optional[str]]:
    try:
        import tabula  # noqa: F401
        import tabula as _t
        return True, str(Path(_t.__file__).parent)
    except Exception:
        return False, None


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _split_genes(s: str) -> list[str]:
    # Accept "GATA4" or "GATA4, TBX5" or "GATA4 TBX5".
    parts = re.split(r"[,\s]+", s.strip())
    return [p for p in (x.strip() for x in parts) if p]


def _safe_gene_token(g: str) -> str:
    # Stable column suffix for adata.obs.
    return re.sub(r"[^0-9A-Za-z]+", "_", g.upper()).strip("_")[:64]


@dataclass(frozen=True)
class _PerturbAssets:
    ckpt_dir: Path
    model_path: Path
    genes: list[str]
    gene_ids: np.ndarray  # (n_genes,)
    meta: dict[str, Any]


class TabulaAdapter(BaseAdapter):
    """
    Tabula adapter.

    Currently implemented:
    - perturb: run a fine-tuned GEARS perturbation model on a new AnnData.

    Embedding/integration/annotation are not implemented in this adapter version.
    """

    def __init__(self, checkpoint_dir: Optional[str] = None):
        spec = get_registry().get("tabula")
        if spec is None:
            raise ValueError("Tabula model not found in registry")
        super().__init__(spec, checkpoint_dir)
        self._tabula_ok, self._tabula_path = _check_tabula_importable()
        self._assets: Optional[_PerturbAssets] = None
        self._model = None

    def run(
        self,
        task: TaskType,
        adata_path: str,
        output_path: str,
        batch_key: Optional[str] = None,  # unused
        label_key: Optional[str] = None,  # optional for stratified summaries
        device: str = "auto",
        batch_size: int = 8,
    ) -> dict[str, Any]:
        if task != TaskType.PERTURB:
            return {
                "error": f"Tabula adapter currently only implements task='{TaskType.PERTURB.value}'",
                "requested_task": task.value,
                "supported_tasks": [TaskType.PERTURB.value],
                "note": "This adapter is intentionally minimal for perturbation benchmarking.",
            }

        device = self._resolve_device(device)
        if device == "cpu":
            return {
                "error": "Tabula perturbation requires GPU for practical runtime (CUDA).",
                "suggestion": "Run on a CUDA GPU node or use a different model.",
            }

        if not self._tabula_ok:
            return {
                "error": "Tabula python package/source is not importable (import tabula failed).",
                "hint": "Ensure vendor tabula is on PYTHONPATH (e.g., ~/scfm_eval/vendor/tabula).",
            }

        try:
            import scanpy as sc
        except Exception as e:
            return {"error": f"scanpy is required: {e}"}

        adata = sc.read_h5ad(adata_path)

        # Resolve perturb genes from env. Keep default aligned with scfm_eval query builder.
        pert_genes = _split_genes(os.environ.get("SCFM_PERTURB_GENES", "GATA4"))
        pert_genes = [g.upper() for g in pert_genes]

        try:
            assets = self._load_assets()
            model = self._load_model_with_assets(device=device, assets=assets)
        except Exception as e:
            return {"error": f"Failed to load Tabula perturbation checkpoint: {e}"}

        try:
            pred, summaries = self._predict_perturb(
                adata=adata,
                assets=assets,
                model=model,
                pert_genes=pert_genes,
                device=device,
                batch_size=batch_size,
                label_key=label_key,
            )
        except Exception as e:
            return {"error": f"Tabula perturbation inference failed: {e}"}

        # Optionally store the full prediction matrix (can be large).
        save_matrix = _env_bool("SCFM_PERTURB_SAVE_MATRIX", default=False)
        output_keys: list[str] = []
        if save_matrix:
            layer_key = "X_tabula_perturb_pred"
            adata.layers[layer_key] = pred
            output_keys.append(f"layers['{layer_key}']")

        # Always store per-gene summary vectors in obs.
        for g in pert_genes:
            tok = _safe_gene_token(g)
            if tok in summaries:
                for k, v in summaries[tok]["vectors"].items():
                    col = f"tabula_perturb_{k}_{tok}"
                    adata.obs[col] = v
                    output_keys.append(f"obs['{col}']")

        # Minimal metadata for downstream metrics aggregation.
        adata.uns["scfm_perturb"] = {
            "model": "tabula",
            "checkpoint_dir": str(assets.ckpt_dir),
            "pert_genes": pert_genes,
            "gene_list_len": int(len(assets.genes)),
            "save_matrix": bool(save_matrix),
            "normalization": "normalize_total(1e4)+log1p on counts/raw",
            "stats": {k: v["stats"] for k, v in summaries.items()},
        }
        output_keys.append("uns['scfm_perturb']")

        self._add_provenance(adata, task, output_keys)
        adata.write(output_path)

        return {
            "status": "success",
            "output_path": output_path,
            "output_keys": output_keys,
            "stats": {
                "n_cells": int(adata.n_obs),
                "device": device,
                "perturbation": {
                    "pert_genes": pert_genes,
                    "summaries": {k: v["stats"] for k, v in summaries.items()},
                    "saved_matrix": bool(save_matrix),
                    "n_model_genes": int(len(assets.genes)),
                },
            },
        }

    def _load_model_with_assets(self, device: str, assets: _PerturbAssets):
        if self._model is not None:
            return self._model

        import torch

        from tabula.model.transfomer.transformer import TabulaTransformer

        sd = torch.load(str(assets.model_path), map_location=device)
        if not isinstance(sd, dict):
            raise ValueError("Checkpoint file did not contain a state_dict dict")

        # Infer key dimensions from state_dict for robustness.
        try:
            d_token = int(sd["cls.weight"].shape[1])
        except Exception:
            d_token = 192

        try:
            vocab_size = int(sd["feature_tokenizer.gene_encoder.embedding.weight"].shape[0])
        except Exception:
            vocab_size = int(assets.gene_ids.max()) + 1

        in_feature = int(len(assets.genes))

        model = TabulaTransformer(
            in_feature=in_feature,
            embedding_in_feature=vocab_size,
            contrastive_out_feature=128,
            supervised_out_feature=0,
            d_token=d_token,
            n_blocks=3,
            residual_dropout=0.0,
            additive_attention=False,
            flash_attention=True,
            attention_n_heads=8,
            attention_dropout=0.2,
            ffn_d_hidden=d_token,
            ffn_dropout=0.1,
            cls=True,
            pre_normalization=True,
            global_token=True,
            pretrain_objective="both",
            enable_batch=False,
            explicit_zero_prob=False,
            do_mgm=True,
            do_cmgm=False,
            cmgm_decoder_style="inner product",
            do_dab=False,
            n_batch=1,
            embed_style="cls",
        ).to(device)

        missing, unexpected = model.load_state_dict(sd, strict=False)
        # We expect strict-ish matches for a fine-tuned checkpoint; surface obvious issues early.
        if unexpected:
            raise ValueError(f"Unexpected checkpoint keys (sample): {unexpected[:10]}")
        if any("mgm_decoder" in k for k in missing):
            raise ValueError(
                "Checkpoint is missing mgm_decoder weights; this does not look like a perturbation fine-tune. "
                "Expected a Tabula GEARS fine-tuned best_model.pth."
            )

        model.eval()
        self._model = model
        return model

    # --- BaseAdapter abstract method implementations ---

    def _load_model(self, device: str):
        # Lazy-load the perturbation checkpoint when invoked by BaseAdapter workflows.
        assets = self._load_assets()
        return self._load_model_with_assets(device=device, assets=assets)

    def _preprocess(self, adata, task: TaskType):
        # Perturbation preprocessing is handled inside _predict_perturb.
        return adata

    def _postprocess(self, adata, embeddings, task: TaskType) -> list[str]:
        # Perturbation output writing is handled inside run().
        return []

    def _load_assets(self) -> _PerturbAssets:
        if self._assets is not None:
            return self._assets

        ckpt_dir = self._resolve_checkpoint_dir(require=True)
        assert ckpt_dir is not None

        model_path = ckpt_dir / "best_model.pth"
        if not model_path.exists():
            # Fall back to any .pth in the folder.
            model_path = self._find_checkpoint(ckpt_dir, extensions=[".pth", ".pt", ".ckpt"])

        genes_path = ckpt_dir / "genes.json"
        if not genes_path.exists():
            raise FileNotFoundError(f"Missing required file: {genes_path}")
        genes = json.loads(genes_path.read_text())
        if not isinstance(genes, list) or not all(isinstance(x, str) for x in genes):
            raise ValueError("genes.json must be a JSON list[str]")

        gene_ids_path = ckpt_dir / "gene_ids.npy"
        if not gene_ids_path.exists():
            raise FileNotFoundError(f"Missing required file: {gene_ids_path}")
        gene_ids = np.load(gene_ids_path)
        gene_ids = np.asarray(gene_ids, dtype=np.int64)
        if gene_ids.ndim != 1 or gene_ids.shape[0] != len(genes):
            raise ValueError("gene_ids.npy must be a 1D array aligned to genes.json")

        meta_path = ckpt_dir / "meta.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta_obj = json.loads(meta_path.read_text())
                if isinstance(meta_obj, dict):
                    meta = meta_obj
            except Exception:
                meta = {}

        assets = _PerturbAssets(
            ckpt_dir=ckpt_dir,
            model_path=model_path,
            genes=list(genes),
            gene_ids=gene_ids,
            meta=meta,
        )
        self._assets = assets
        return assets

    def _predict_perturb(
        self,
        adata,
        assets: _PerturbAssets,
        model,
        pert_genes: list[str],
        device: str,
        batch_size: int,
        label_key: Optional[str],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        import torch

        # Use counts if provided, else raw, else X.
        X = None
        if hasattr(adata, "layers") and isinstance(getattr(adata, "layers"), dict) and "counts" in adata.layers:
            X = adata.layers["counts"]
        elif getattr(adata, "raw", None) is not None:
            try:
                X = adata.raw.X
            except Exception:
                X = adata.X
        else:
            X = adata.X

        # Map adata genes to model gene order.
        model_genes_upper = [g.upper() for g in assets.genes]
        adata_genes_upper = [g.upper() for g in adata.var_names.tolist()]
        gene_to_col = {g: i for i, g in enumerate(adata_genes_upper)}
        cols = [gene_to_col.get(g) for g in model_genes_upper]

        present_pos = [i for i, c in enumerate(cols) if c is not None]
        present_cols = [cols[i] for i in present_pos]

        n_cells = int(adata.n_obs)
        n_genes = int(len(model_genes_upper))
        values = np.zeros((n_cells, n_genes), dtype=np.float32)

        if present_cols:
            # Extract in one shot and place into the aligned matrix.
            Xp = X[:, present_cols]
            if hasattr(Xp, "toarray"):
                Xp = Xp.toarray()
            Xp = np.asarray(Xp, dtype=np.float32)
            values[:, present_pos] = Xp

        # Normalize per cell to 1e4 and log1p (matches common scRNA conventions).
        cell_sums = X.sum(axis=1)
        if hasattr(cell_sums, "A1"):
            cell_sums = cell_sums.A1
        cell_sums = np.asarray(cell_sums, dtype=np.float32).reshape(-1)
        denom = np.maximum(cell_sums, 1.0)
        values = values / denom[:, None] * 1e4
        values = np.log1p(values)

        # Preserve baseline (unperturbed) expression for the perturbed genes so we can
        # report delta relative to the original input, not the KO-zeroed value.
        gene_to_pos = {g: i for i, g in enumerate(model_genes_upper)}
        baseline_by_gene: dict[str, np.ndarray] = {}
        for g in pert_genes:
            p = gene_to_pos.get(g)
            if p is not None:
                baseline_by_gene[g] = values[:, p].copy()

        # Apply knockout by zeroing the input value and setting pert_flags.
        pert_pos = []
        for g in pert_genes:
            if g in gene_to_pos:
                pert_pos.append(gene_to_pos[g])

        pert_flags = np.zeros((n_cells, n_genes), dtype=np.int64)
        for p in pert_pos:
            values[:, p] = 0.0
            pert_flags[:, p] = 1

        # Prepare constant gene token ids (already aligned to assets.genes).
        gene_ids = torch.tensor(assets.gene_ids, dtype=torch.long, device=device)
        gene_ids = gene_ids.unsqueeze(0)  # (1, n_genes)

        pred = np.zeros((n_cells, n_genes), dtype=np.float32)
        per_gene_vectors: dict[str, dict[str, np.ndarray]] = {}
        for g in pert_genes:
            tok = _safe_gene_token(g)
            per_gene_vectors[tok] = {
                "baseline": np.zeros((n_cells,), dtype=np.float32),
                "pred": np.zeros((n_cells,), dtype=np.float32),
                "delta": np.zeros((n_cells,), dtype=np.float32),
            }

        model.eval()
        with torch.no_grad():
            for start in range(0, n_cells, batch_size):
                end = min(start + batch_size, n_cells)
                b = end - start

                batch_gene_ids = gene_ids.expand(b, -1)
                batch_values = torch.tensor(values[start:end], dtype=torch.float32, device=device)
                batch_flags = torch.tensor(pert_flags[start:end], dtype=torch.long, device=device)

                with torch.cuda.amp.autocast(enabled=True):
                    out = model(
                        genes=batch_gene_ids,
                        values=batch_values,
                        pert_flags=batch_flags,
                        head=None,
                        do_mgm=True,
                    )
                    mgm = out.get("mgm_pred")
                    if mgm is None:
                        raise ValueError("Model output missing mgm_pred (is do_mgm enabled in the checkpoint?)")
                    mgm = mgm.float().detach().cpu().numpy().astype(np.float32)

                pred[start:end] = mgm

        # Build lightweight per-gene vectors + summary stats.
        summaries: dict[str, Any] = {}
        for g in pert_genes:
            tok = _safe_gene_token(g)
            if g not in gene_to_pos:
                summaries[tok] = {
                    "stats": {"error": f"gene_not_in_model_gene_list gene={g}"},
                    "vectors": {},
                }
                continue

            p = gene_to_pos[g]
            v_base = baseline_by_gene.get(g)
            if v_base is None:
                v_base = values[:, p]
            v_pred = pred[:, p]
            v_delta = v_pred - v_base
            per_gene_vectors[tok]["baseline"] = v_base.astype(np.float32)
            per_gene_vectors[tok]["pred"] = v_pred.astype(np.float32)
            per_gene_vectors[tok]["delta"] = v_delta.astype(np.float32)

            stats = {
                "gene": g,
                "n_cells": int(n_cells),
                "mean_delta": float(np.mean(v_delta)),
                "median_delta": float(np.median(v_delta)),
                "frac_negative_delta": float(np.mean(v_delta < 0)),
                "mean_baseline": float(np.mean(v_base)),
                "mean_pred": float(np.mean(v_pred)),
            }

            # Optional stratified stats if label_key exists.
            if label_key and label_key in adata.obs.columns:
                grp = adata.obs[label_key].astype(str).to_numpy()
                by = {}
                for k in sorted(set(grp)):
                    m = grp == k
                    if int(m.sum()) < 5:
                        continue
                    by[k] = {
                        "n": int(m.sum()),
                        "mean_delta": float(np.mean(v_delta[m])),
                        "frac_negative_delta": float(np.mean(v_delta[m] < 0)),
                    }
                stats["by_label"] = by

            summaries[tok] = {"stats": stats, "vectors": per_gene_vectors[tok]}

        return pred, summaries
