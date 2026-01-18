"""
UCE (Universal Cell Embeddings) Adapter

UCE is a zero-shot, cross-species foundation model that uses ESM-2 protein embeddings
for gene representations. It's ideal for embedding tasks without fine-tuning.

Reference: https://github.com/snap-stanford/UCE
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..registry import ModelSpec, TaskType, get_registry
from .base import BaseAdapter


def _check_uce_installed() -> tuple[bool, Optional[str]]:
    """Check if UCE package is installed and return its location."""
    try:
        import uce
        return True, str(Path(uce.__file__).parent)
    except ImportError:
        return False, None


class UCEAdapter(BaseAdapter):
    """
    Adapter for Universal Cell Embeddings (UCE) model.

    Supports:
    - embed: Zero-shot cell embeddings
    - integrate: Batch integration via shared embedding space

    Requirements:
    - GPU with 16+ GB VRAM
    - UCE package installed (pip install git+https://github.com/snap-stanford/UCE.git)
    - ESM-2 weights (auto-downloaded)
    """

    def __init__(self, checkpoint_dir: Optional[str] = None):
        spec = get_registry().get("uce")
        if spec is None:
            raise ValueError("UCE model not found in registry")
        super().__init__(spec, checkpoint_dir)

        self._uce_model = None
        self._uce_installed, self._uce_path = _check_uce_installed()
        # Mapping from short names to scientific names (for display)
        self._species_to_name = {
            "human": "Homo sapiens",
            "mouse": "Mus musculus",
            "zebrafish": "Danio rerio",
            "frog": "Xenopus tropicalis",
            "pig": "Sus scrofa",
            "mouse_lemur": "Microcebus murinus",
            "macaca_mulatta": "Macaca mulatta",
            "macaca_fascicularis": "Macaca fascicularis",
        }
        # UCE expects short species names for the eval script
        self._uce_species_names = {
            "human": "human",
            "mouse": "mouse",
            "zebrafish": "zebrafish",
            "frog": "frog",
            "pig": "pig",
            "mouse_lemur": "mouse_lemur",
            "macaca_mulatta": "macaca_mulatta",
            "macaca_fascicularis": "macaca_fascicularis",
        }

    def run(
        self,
        task: TaskType,
        adata_path: str,
        output_path: str,
        batch_key: Optional[str] = None,
        label_key: Optional[str] = None,
        device: str = "auto",
        batch_size: int = 100,
    ) -> dict[str, Any]:
        """
        Run UCE model for embedding or integration task.

        Args:
            task: TaskType.EMBED or TaskType.INTEGRATE
            adata_path: Path to input .h5ad file
            output_path: Path for output .h5ad file
            batch_key: Not used for UCE (integration is implicit)
            label_key: Not used for UCE
            device: Device to use ('auto', 'cuda', 'cpu')
            batch_size: Batch size for inference (default: 100)

        Returns:
            Dictionary with output_path, output_keys, and statistics
        """
        if task not in [TaskType.EMBED, TaskType.INTEGRATE]:
            return {
                "error": f"UCE does not support task '{task.value}'",
                "supported_tasks": ["embed", "integrate"],
            }

        device = self._resolve_device(device)
        if device == "cpu":
            return {
                "error": "UCE requires GPU. CPU fallback not supported.",
                "suggestion": "Use a GPU-enabled environment or try Geneformer (has CPU fallback)",
            }

        # Check for UCE package
        if not self._uce_installed:
            return {
                "error": "UCE package not installed",
                "install": "pip install git+https://github.com/snap-stanford/UCE.git",
                "documentation": "https://github.com/snap-stanford/UCE",
            }

        # Load data
        try:
            import scanpy as sc
            adata = sc.read_h5ad(adata_path)
        except ImportError:
            return {"error": "scanpy not installed. Install with: pip install scanpy"}
        except Exception as e:
            return {"error": f"Failed to read AnnData: {str(e)}"}

        # Detect species
        species = self._detect_species(adata)
        if species not in self._species_to_name:
            return {
                "error": f"Species '{species}' not supported by UCE",
                "supported": list(self._species_to_name.keys()),
            }

        # Get UCE-compatible species name (short form)
        uce_species = self._uce_species_names.get(species, species)

        # Load model
        try:
            self._load_model(device)
        except Exception as e:
            return {"error": f"Failed to load UCE model: {str(e)}"}

        # Preprocess
        try:
            processed_adata = self._preprocess(adata, task)
        except Exception as e:
            return {"error": f"Preprocessing failed: {str(e)}"}

        # Run inference
        try:
            embeddings, output_adata = self._run_inference(
                processed_adata,
                species=uce_species,  # Use short species name for UCE
                device=device,
                batch_size=batch_size,
            )
        except Exception as e:
            return {"error": f"Inference failed: {str(e)}"}

        # UCE may filter cells without valid gene mappings
        # Use the output adata from UCE which has properly aligned cells
        n_filtered = adata.n_obs - output_adata.n_obs

        # Add provenance to the output adata
        output_keys = ["X_uce"]

        # Add simplified provenance (h5ad cannot serialize lists of dicts)
        from datetime import datetime
        if "scfm" not in output_adata.uns:
            output_adata.uns["scfm"] = {}
        output_adata.uns["scfm"]["model_name"] = self.spec.name
        output_adata.uns["scfm"]["version"] = str(self.spec.version) if self.spec.version else "unknown"
        output_adata.uns["scfm"]["task"] = task.value
        output_adata.uns["scfm"]["output_keys"] = ",".join(output_keys)  # Store as comma-separated string
        output_adata.uns["scfm"]["timestamp"] = datetime.now().isoformat()

        # Save the UCE output (with potentially fewer cells)
        output_adata.write(output_path)

        result = {
            "status": "success",
            "output_path": output_path,
            "output_keys": output_keys,
            "stats": {
                "n_cells_input": adata.n_obs,
                "n_cells_output": output_adata.n_obs,
                "n_cells_filtered": n_filtered,
                "embedding_dim": embeddings.shape[1],
                "species": species,
                "device": device,
            },
        }

        if n_filtered > 0:
            result["warning"] = (
                f"UCE filtered {n_filtered} cells without valid gene mappings. "
                f"Output contains {output_adata.n_obs} cells."
            )

        return result

    def _load_model(self, device: str):
        """
        Load UCE model.

        UCE model loading is handled by the evaluation script.
        This method validates the environment and caches model location.
        """
        if self._uce_model is not None:
            return

        # UCE uses its own model loading via eval_single_anndata.py
        # We just validate that the package is available
        if not self._uce_installed:
            raise ImportError("UCE package not installed")

        # Mark as "loaded" - actual loading happens in _run_inference
        self._uce_model = "ready"

    def _preprocess(self, adata, task: TaskType):
        """
        Preprocess AnnData for UCE.

        UCE requires:
        - Log-normalized counts
        - Gene symbols (not Ensembl IDs)
        """
        import scanpy as sc

        # Work on a copy
        adata = adata.copy()

        # Ensure we have raw counts or normalized data
        if adata.raw is not None:
            # Use raw counts if available
            adata = adata.raw.to_adata()

        # Log-normalize if not already done
        if "log1p" not in adata.uns:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

        return adata

    def _run_inference(
        self,
        adata,
        species: str,
        device: str,
        batch_size: int,
    ):
        """
        Run UCE inference to generate embeddings.

        Uses UCE's eval_single_anndata.py script via subprocess for robust execution.
        This ensures compatibility with UCE's complex preprocessing pipeline.

        Args:
            adata: Preprocessed AnnData object
            species: Species name (e.g., "Homo sapiens")
            device: Device string (e.g., "cuda")
            batch_size: Batch size for inference

        Returns:
            tuple: (embeddings array, output adata with aligned cells)
                   UCE may filter cells without valid gene mappings
        """
        import scanpy as sc

        # Create temporary directory for UCE intermediate files
        with tempfile.TemporaryDirectory(prefix="uce_") as temp_dir:
            temp_path = Path(temp_dir)

            # Save preprocessed data to temp file
            input_path = temp_path / "input.h5ad"
            adata.write(str(input_path))

            # Run UCE evaluation script - returns (embeddings, output_adata)
            embeddings, output_adata = self._run_uce_script(
                input_path=str(input_path),
                output_dir=str(temp_path),
                species=species,
                batch_size=batch_size,
            )

            return embeddings, output_adata

    def _download_model_from_huggingface(self, nlayers: int = 33) -> str:
        """
        Download UCE model weights from HuggingFace Hub.

        Args:
            nlayers: Number of layers (4 for 100M model, 33 for 650M model)

        Returns:
            Path to downloaded model file
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub not installed. Install with: pip install huggingface_hub"
            )

        # Select model based on layers
        if nlayers == 33:
            repo_id = "minwoosun/uce-650m"
        else:
            repo_id = "minwoosun/uce-100m"

        # Download model weights (cached after first download)
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="pytorch_model.bin",
        )

        return model_path

    def _setup_protein_embeddings(self) -> str:
        """
        Download and setup protein embeddings from HuggingFace.

        UCE requires protein embeddings for gene-to-protein mapping.
        These are downloaded from minwoosun/uce-misc repository.

        Returns:
            Path to protein embeddings directory
        """
        import tarfile

        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface_hub not installed. Install with: pip install huggingface_hub"
            )

        # Download protein embeddings archive
        embeddings_tar = hf_hub_download(
            repo_id="minwoosun/uce-misc",
            filename="protein_embeddings.tar.gz",
        )

        # Download species files
        species_chrom = hf_hub_download(
            repo_id="minwoosun/uce-misc",
            filename="species_chrom.csv",
        )
        species_offsets = hf_hub_download(
            repo_id="minwoosun/uce-misc",
            filename="species_offsets.pkl",
        )
        all_tokens = hf_hub_download(
            repo_id="minwoosun/uce-misc",
            filename="all_tokens.torch",
        )

        # Extract protein embeddings if not already done
        cache_dir = Path(embeddings_tar).parent
        embeddings_dir = cache_dir / "protein_embeddings"

        if not embeddings_dir.exists():
            print(f"Extracting protein embeddings to {embeddings_dir}...")
            with tarfile.open(embeddings_tar, "r:gz") as tar:
                tar.extractall(path=cache_dir)

        # Create model_files directory structure expected by UCE
        model_files_dir = cache_dir / "model_files"
        model_files_dir.mkdir(exist_ok=True)

        # Create symlinks or copies to match UCE expected paths
        import shutil

        # Copy species files (resolve HuggingFace symlinks to get actual file content)
        for src, dst_name in [
            (species_chrom, "species_chrom.csv"),
            (species_offsets, "species_offsets.pkl"),
            (all_tokens, "all_tokens.torch"),
        ]:
            dst = model_files_dir / dst_name
            # Re-copy if file doesn't exist or is 0-bytes (failed copy)
            needs_copy = not dst.exists() or dst.stat().st_size == 0
            if needs_copy:
                if dst.exists():
                    dst.unlink()  # Remove 0-byte file
                # HuggingFace cache files are symlinks - resolve to get actual file
                src_resolved = Path(src).resolve()
                shutil.copy2(src_resolved, dst)

        # Link protein embeddings
        pe_link = model_files_dir / "protein_embeddings"
        if not pe_link.exists():
            if embeddings_dir.exists():
                pe_link.symlink_to(embeddings_dir)

        # Create new_species_protein_embeddings.csv mapping species to embedding files
        species_csv = model_files_dir / "new_species_protein_embeddings.csv"
        if not species_csv.exists():
            # Map species names to their protein embedding files
            species_mapping = {
                "Homo sapiens": "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt",
                "Mus musculus": "Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt",
                "Danio rerio": "Danio_rerio.GRCz11.gene_symbol_to_embedding_ESM2.pt",
                "Xenopus tropicalis": "Xenopus_tropicalis.Xenopus_tropicalis_v9.1.gene_symbol_to_embedding_ESM2.pt",
                "Sus scrofa": "Sus_scrofa.Sscrofa11.1.gene_symbol_to_embedding_ESM2.pt",
                "Macaca mulatta": "Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM2.pt",
                "Macaca fascicularis": "Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt",
                "Microcebus murinus": "Microcebus_murinus.Mmur_3.0.gene_symbol_to_embedding_ESM2.pt",
            }

            # Write CSV with species -> path mapping
            import csv
            with open(species_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["species", "path"])
                for species_name, filename in species_mapping.items():
                    # Path relative to model_files/
                    path = f"./model_files/protein_embeddings/{filename}"
                    writer.writerow([species_name, path])

        return str(model_files_dir)

    def _run_uce_script(
        self,
        input_path: str,
        output_dir: str,
        species: str,
        batch_size: int,
        nlayers: int = 33,
    ) -> np.ndarray:
        """
        Execute UCE evaluation script and return embeddings.

        Args:
            input_path: Path to input .h5ad file
            output_dir: Directory for UCE output files
            species: Species name
            batch_size: Batch size for inference
            nlayers: Number of model layers (4 or 33, default 33)

        Returns:
            np.ndarray: Cell embeddings
        """
        import scanpy as sc

        # Find the UCE eval script
        if self._uce_path is None:
            raise ImportError("UCE package not found")

        # Download model from HuggingFace (bypasses figshare which may be blocked)
        try:
            model_path = self._download_model_from_huggingface(nlayers=nlayers)
        except Exception as e:
            raise RuntimeError(f"Failed to download UCE model from HuggingFace: {e}")

        # Setup protein embeddings from HuggingFace
        try:
            model_files_dir = self._setup_protein_embeddings()
        except Exception as e:
            raise RuntimeError(f"Failed to setup protein embeddings: {e}")

        uce_script = Path(self._uce_path) / "eval_single_anndata.py"

        # Build command with model_loc pointing to HuggingFace downloaded model
        # Also set pe_dir for protein embeddings location
        base_args = [
            "--adata_path", input_path,
            "--dir", output_dir,
            "--species", species,
            "--batch_size", str(batch_size),
            "--model_loc", model_path,
            "--nlayers", str(nlayers),
        ]

        # UCE eval_single_anndata.py uses relative imports (from evaluate import ...)
        # So we need to run from the UCE package directory
        uce_package_dir = Path(self._uce_path)

        # Use the script directly from the package
        if not uce_script.exists():
            raise RuntimeError(f"UCE eval script not found at {uce_script}")

        # Setup model_files in UCE package directory
        # UCE looks for ./model_files/ relative to its package directory
        uce_model_files = uce_package_dir / "model_files"

        if uce_model_files.is_symlink():
            # Remove old symlink if it points to wrong location
            if str(uce_model_files.resolve()) != str(Path(model_files_dir).resolve()):
                uce_model_files.unlink()
                uce_model_files.symlink_to(model_files_dir)
        elif uce_model_files.is_dir():
            # Directory exists - check if it has our files, if not copy them
            # Copy all files from HuggingFace model_files to UCE model_files
            import shutil
            for item in Path(model_files_dir).iterdir():
                dst = uce_model_files / item.name
                # Check if we need to copy: doesn't exist, or is 0-bytes (failed copy)
                needs_copy = not dst.exists()
                if dst.exists() and dst.is_file() and dst.stat().st_size == 0:
                    needs_copy = True
                    dst.unlink()  # Remove corrupted 0-byte file

                if needs_copy:
                    # Resolve symlinks to get actual content
                    src_resolved = item.resolve() if item.is_symlink() else item
                    if src_resolved.is_dir():
                        shutil.copytree(src_resolved, dst)
                    else:
                        shutil.copy2(src_resolved, dst)
        elif not uce_model_files.exists():
            uce_model_files.symlink_to(model_files_dir)

        # Ensure output_dir ends with / (UCE has a bug that concatenates without separator)
        if not output_dir.endswith("/"):
            output_dir = output_dir + "/"
            # Update base_args with corrected output_dir
            dir_idx = base_args.index("--dir") + 1
            base_args[dir_idx] = output_dir

        cmd = ["python", str(uce_script)] + base_args

        # Execute UCE from UCE package directory (required for relative imports)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout for large datasets
                check=True,
                cwd=str(uce_package_dir),  # Run from UCE package dir for relative imports
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"UCE execution failed: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("UCE execution timed out (>1 hour)")

        # Load results from output file
        # UCE outputs to {output_dir}input_uce_adata.h5ad (input is the dataset name)
        output_dir_path = Path(output_dir.rstrip("/"))

        # Look for *_uce_adata.h5ad files
        output_files = list(output_dir_path.glob("*_uce_adata.h5ad"))

        # Also check parent directory (in case UCE path bug)
        if not output_files:
            output_files = list(output_dir_path.parent.glob("*_uce_adata.h5ad"))

        # Filter to find the UCE output file
        if not output_files:
            # Fallback: look for any .h5ad that's not input
            output_files = [f for f in output_dir_path.glob("*.h5ad") if f.name != "input.h5ad"]

        if not output_files:
            # If no separate output, UCE may have modified input in place
            output_file = Path(input_path)
        else:
            output_file = output_files[0]

        # Read embeddings from output
        result_adata = sc.read_h5ad(str(output_file))

        if "X_uce" not in result_adata.obsm:
            raise RuntimeError(
                f"UCE output missing X_uce embeddings. "
                f"Available keys: {list(result_adata.obsm.keys())}"
            )

        # Return both embeddings and the output adata
        # UCE may have filtered some cells, so return the aligned adata
        return result_adata.obsm["X_uce"], result_adata

    def _run_inference_direct(
        self,
        adata,
        species: str,
        device: str,
        batch_size: int,
    ) -> np.ndarray:
        """
        Alternative: Run UCE inference using direct Python imports.

        This method attempts to use UCE's internal APIs directly.
        Falls back to subprocess method if imports fail.

        Note: This is more fragile as it depends on UCE's internal structure.
        """
        try:
            # Try importing UCE's evaluation module
            from uce.evaluate import AnndataProcessor
            from accelerate import Accelerator

            # Set up accelerator
            accelerator = Accelerator()

            # Create args namespace for UCE
            class Args:
                pass

            args = Args()
            args.adata_path = None  # We'll pass adata directly
            args.dir = tempfile.mkdtemp(prefix="uce_")
            args.species = species
            args.batch_size = batch_size
            args.nlayers = 4  # Default 4-layer model
            args.output_dim = self.spec.embedding_dim
            args.model_loc = None  # Use default model location
            args.multi_gpu = False
            args.filter = False
            args.skip = False

            # Process with UCE
            processor = AnndataProcessor(args, accelerator)
            processor.adata = adata
            processor.preprocess_anndata()
            processor.generate_idxs()
            processor.run_evaluation()

            # Get embeddings
            embeddings = processor.adata.obsm["X_uce"]

            # Clean up
            shutil.rmtree(args.dir, ignore_errors=True)

            return embeddings

        except ImportError:
            # Fall back to subprocess method
            return self._run_uce_script(
                input_path=str(Path(tempfile.mkdtemp()) / "input.h5ad"),
                output_dir=tempfile.mkdtemp(),
                species=species,
                batch_size=batch_size,
            )

    def _postprocess(self, adata, embeddings: np.ndarray, task: TaskType) -> list[str]:
        """
        Write embeddings to AnnData.

        Args:
            adata: AnnData object to update
            embeddings: Cell embeddings array (n_cells, embedding_dim)
            task: Task type

        Returns:
            List of output keys written
        """
        output_keys = []

        # Write embeddings
        key = self.spec.output_keys.embedding_key  # "X_uce"
        adata.obsm[key] = embeddings
        output_keys.append(f"obsm['{key}']")

        return output_keys

    def _detect_species(self, adata) -> str:
        """Detect species from AnnData"""
        # Check uns first
        if "species" in adata.uns:
            species = adata.uns["species"].lower()
            if "human" in species or "sapiens" in species:
                return "human"
            elif "mouse" in species or "musculus" in species:
                return "mouse"

        # Infer from gene naming
        gene_names = adata.var_names[:100].tolist()
        uppercase_count = sum(1 for g in gene_names if g.isupper())

        if uppercase_count > 50:
            return "human"
        return "mouse"
