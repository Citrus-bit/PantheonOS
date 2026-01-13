#!/usr/bin/env python
"""
Compare official Harmony vs evolved Harmony on TMA data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import importlib.util
import sys
from pathlib import Path
from umap import UMAP
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

# Setup paths
example_dir = Path(__file__).parent
data_dir = example_dir / "data"


def load_module(name: str, path: Path):
    """Load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_tma_data(split: str = "train"):
    """Load TMA data with real cell type labels."""
    df = pd.read_csv(data_dir / f"tma_8000_{split}.csv")
    X = df.iloc[:, :30].values  # PC1-PC30
    batch_labels = df["donor"].values
    cell_types = df["celltype"].values
    return X, batch_labels, cell_types


def compute_batch_mixing_score(X: np.ndarray, batch_labels: np.ndarray, k: int = 50) -> float:
    """Compute batch mixing score using k-nearest neighbors."""
    n_cells = X.shape[0]
    unique_batches = np.unique(batch_labels)
    expected_props = np.array([np.sum(batch_labels == b) / n_cells for b in unique_batches])

    nn = NearestNeighbors(n_neighbors=min(k + 1, n_cells), algorithm="auto")
    nn.fit(X)
    _, indices = nn.kneighbors(X)

    mixing_scores = []
    for i in range(n_cells):
        neighbor_batches = batch_labels[indices[i, 1:]]
        observed_props = np.array([np.sum(neighbor_batches == b) / k for b in unique_batches])
        score = 1 - np.sqrt(np.mean((observed_props - expected_props) ** 2))
        mixing_scores.append(max(0, score))

    return np.mean(mixing_scores)


def compute_bio_conservation_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute biological structure conservation using silhouette score."""
    try:
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(X, labels)
            return (silhouette + 1) / 2
        return 0.5
    except Exception:
        return 0.5


def run_comparison():
    """Run the comparison and create plots."""
    print("Loading TMA data...")
    X, batch_labels, cell_types = load_tma_data("train")
    print(f"  Data shape: {X.shape}")
    print(f"  Batches: {np.unique(batch_labels)}")
    print(f"  Cell types: {len(np.unique(cell_types))}")

    # Create meta_data DataFrame for official API
    meta_data = pd.DataFrame({"batch": batch_labels})

    # Load both harmony implementations
    print("\nLoading Harmony implementations...")
    harmony_official = load_module("harmony_official", example_dir / "harmony.py")
    harmony_evolved = load_module("harmony_evolved", example_dir / "results_official" / "harmony_optimized.py")

    # Run official harmony
    print("\nRunning official Harmony (PyTorch)...")
    start_time = time.time()
    hm_official = harmony_official.run_harmony(
        X, meta_data, vars_use="batch", nclust=50, max_iter_harmony=10, random_state=42, verbose=False
    )
    time_official = time.time() - start_time
    X_corrected_official = hm_official.Z_corr
    print(f"  Time: {time_official:.2f}s")
    print(f"  Iterations: {len(hm_official.objectives)}")

    # Run evolved harmony
    print("\nRunning evolved Harmony...")
    start_time = time.time()
    hm_evolved = harmony_evolved.run_harmony(
        X, meta_data, vars_use="batch", nclust=50, max_iter_harmony=10, random_state=42, verbose=False
    )
    time_evolved = time.time() - start_time
    X_corrected_evolved = hm_evolved.Z_corr
    print(f"  Time: {time_evolved:.2f}s")
    print(f"  Iterations: {len(hm_evolved.objectives)}")

    # Compute metrics
    print("\nComputing metrics...")
    metrics = {
        "Original": {
            "mixing_score": compute_batch_mixing_score(X, batch_labels),
            "bio_score": compute_bio_conservation_score(X, cell_types),
            "time": 0,
        },
        "Official Harmony": {
            "mixing_score": compute_batch_mixing_score(X_corrected_official, batch_labels),
            "bio_score": compute_bio_conservation_score(X_corrected_official, cell_types),
            "time": time_official,
        },
        "Evolved Harmony": {
            "mixing_score": compute_batch_mixing_score(X_corrected_evolved, batch_labels),
            "bio_score": compute_bio_conservation_score(X_corrected_evolved, cell_types),
            "time": time_evolved,
        },
    }

    for name, m in metrics.items():
        print(f"  {name}: mixing={m['mixing_score']:.4f}, bio={m['bio_score']:.4f}, time={m['time']:.2f}s")

    # Compute UMAP embeddings
    print("\nComputing UMAP embeddings...")
    umap = UMAP(n_neighbors=30, min_dist=0.3, random_state=42)

    print("  Original...")
    umap_original = umap.fit_transform(X)

    print("  Official Harmony corrected...")
    umap_official = umap.fit_transform(X_corrected_official)

    print("  Evolved Harmony corrected...")
    umap_evolved = umap.fit_transform(X_corrected_evolved)

    # Create output directory
    output_dir = example_dir / "results_official"

    # Create figure 1: UMAP comparisons
    print("\nCreating UMAP comparison figure...")
    fig1, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Color maps
    batch_cmap = plt.cm.Set1
    cluster_cmap = plt.cm.tab10

    unique_batches = np.unique(batch_labels)
    unique_clusters = np.unique(cell_types)
    batch_colors = {b: batch_cmap(i / len(unique_batches)) for i, b in enumerate(unique_batches)}
    cluster_colors = {c: cluster_cmap(i % 10) for i, c in enumerate(unique_clusters)}

    # Row 1: Color by batch
    datasets = [
        (umap_original, "Original (Uncorrected)", metrics["Original"]),
        (umap_official, "Official Harmony", metrics["Official Harmony"]),
        (umap_evolved, "Evolved Harmony", metrics["Evolved Harmony"]),
    ]

    for idx, (umap_emb, title, m) in enumerate(datasets):
        ax = axes[0, idx]
        for batch in unique_batches:
            mask = batch_labels == batch
            ax.scatter(umap_emb[mask, 0], umap_emb[mask, 1],
                      c=[batch_colors[batch]], label=batch, s=5, alpha=0.6)
        ax.set_title(f"{title}\nMixing: {m['mixing_score']:.3f}", fontsize=11)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        if idx == 2:
            ax.legend(title="Batch", markerscale=3, loc='upper right', fontsize=9)

    # Row 2: Color by cell type
    for idx, (umap_emb, title, m) in enumerate(datasets):
        ax = axes[1, idx]
        for ct in unique_clusters:
            mask = cell_types == ct
            ax.scatter(umap_emb[mask, 0], umap_emb[mask, 1],
                      c=[cluster_colors[ct]], label=ct, s=5, alpha=0.6)
        ax.set_title(f"{title}\nBio Conservation: {m['bio_score']:.3f}", fontsize=11)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        if idx == 2:
            ax.legend(title="Cell Type", markerscale=3, loc='upper right', fontsize=7, ncol=2)

    axes[0, 0].set_ylabel("Color by Batch\nUMAP2")
    axes[1, 0].set_ylabel("Color by Cell Type\nUMAP2")

    plt.suptitle("TMA Data: Official vs Evolved Harmony Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig1.savefig(output_dir / "tma_umap_comparison_official.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: results_official/tma_umap_comparison_official.png")

    # Create figure 2: Performance comparison bar chart
    print("\nCreating performance comparison figure...")
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))

    methods = ["Original", "Official\nHarmony", "Evolved\nHarmony"]
    colors = ["#8b949e", "#58a6ff", "#238636"]

    # Mixing score
    ax = axes2[0]
    mixing_scores = [metrics["Original"]["mixing_score"],
                     metrics["Official Harmony"]["mixing_score"],
                     metrics["Evolved Harmony"]["mixing_score"]]
    bars = ax.bar(methods, mixing_scores, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel("Batch Mixing Score")
    ax.set_title("Batch Integration Quality\n(Higher = Better)")
    ax.set_ylim(0, 1)
    for bar, score in zip(bars, mixing_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Bio conservation score
    ax = axes2[1]
    bio_scores = [metrics["Original"]["bio_score"],
                  metrics["Official Harmony"]["bio_score"],
                  metrics["Evolved Harmony"]["bio_score"]]
    bars = ax.bar(methods, bio_scores, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel("Bio Conservation Score")
    ax.set_title("Biological Structure Preservation\n(Higher = Better)")
    ax.set_ylim(0, 1)
    for bar, score in zip(bars, bio_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Execution time (only for Harmony methods)
    ax = axes2[2]
    time_methods = ["Official\nHarmony", "Evolved\nHarmony"]
    times = [metrics["Official Harmony"]["time"], metrics["Evolved Harmony"]["time"]]
    bars = ax.bar(time_methods, times, color=["#58a6ff", "#238636"], edgecolor='white', linewidth=1.5)
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title("Computational Performance\n(Lower = Better)")
    speedup = time_official / time_evolved if time_evolved > 0 else 1
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
               f'{t:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Show speedup or slowdown
    if speedup >= 1:
        ax.text(0.5, 0.95, f"Speedup: {speedup:.2f}x", transform=ax.transAxes,
               ha='center', va='top', fontsize=11, color='#238636', fontweight='bold')
    else:
        ax.text(0.5, 0.95, f"Slowdown: {1/speedup:.2f}x", transform=ax.transAxes,
               ha='center', va='top', fontsize=11, color='#f85149', fontweight='bold')

    plt.suptitle("TMA Data: Official vs Evolved Harmony Performance", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig2.savefig(output_dir / "tma_performance_comparison_official.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: results_official/tma_performance_comparison_official.png")

    # Create figure 3: Combined score comparison
    print("\nCreating combined score figure...")
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    # Compute combined scores (same weights as evaluator: 45% mixing, 45% bio, 5% speed, 5% conv)
    combined_original = 0.45 * metrics["Original"]["mixing_score"] + 0.45 * metrics["Original"]["bio_score"]
    combined_official = 0.45 * metrics["Official Harmony"]["mixing_score"] + 0.45 * metrics["Official Harmony"]["bio_score"]
    combined_evolved = 0.45 * metrics["Evolved Harmony"]["mixing_score"] + 0.45 * metrics["Evolved Harmony"]["bio_score"]

    categories = ["Mixing\nScore", "Bio\nConservation", "Combined\nScore"]
    original_vals = [metrics["Original"]["mixing_score"], metrics["Original"]["bio_score"], combined_original]
    official_vals = [metrics["Official Harmony"]["mixing_score"], metrics["Official Harmony"]["bio_score"], combined_official]
    evolved_vals = [metrics["Evolved Harmony"]["mixing_score"], metrics["Evolved Harmony"]["bio_score"], combined_evolved]

    x = np.arange(len(categories))
    width = 0.25

    bars1 = ax3.bar(x - width, original_vals, width, label='Original (Uncorrected)', color='#8b949e', edgecolor='white')
    bars2 = ax3.bar(x, official_vals, width, label='Official Harmony', color='#58a6ff', edgecolor='white')
    bars3 = ax3.bar(x + width, evolved_vals, width, label='Evolved Harmony', color='#238636', edgecolor='white')

    ax3.set_ylabel('Score')
    ax3.set_title('TMA Data: Comprehensive Performance Comparison\n(Official Harmonypy vs Evolved)', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend(loc='upper left')
    ax3.set_ylim(0, 1)
    ax3.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig3.savefig(output_dir / "tma_combined_comparison_official.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: results_official/tma_combined_comparison_official.png")

    # Create figure 4: Improvement summary
    print("\nCreating improvement summary figure...")
    fig4, ax4 = plt.subplots(figsize=(8, 5))

    improvement_metrics = ["Mixing Score", "Bio Conservation", "Combined Score"]
    official_scores = [metrics["Official Harmony"]["mixing_score"],
                       metrics["Official Harmony"]["bio_score"],
                       combined_official]
    evolved_scores = [metrics["Evolved Harmony"]["mixing_score"],
                      metrics["Evolved Harmony"]["bio_score"],
                      combined_evolved]

    improvements = [(e - o) / o * 100 for e, o in zip(evolved_scores, official_scores)]

    colors_imp = ['#238636' if imp >= 0 else '#f85149' for imp in improvements]
    bars = ax4.barh(improvement_metrics, improvements, color=colors_imp, edgecolor='white', height=0.5)

    ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Improvement (%)')
    ax4.set_title('Evolution Improvement: Official → Evolved Harmony', fontsize=14, fontweight='bold')

    for bar, imp in zip(bars, improvements):
        width = bar.get_width()
        ax4.text(width + 0.5 if width >= 0 else width - 0.5, bar.get_y() + bar.get_height()/2,
                f'{imp:+.2f}%', ha='left' if width >= 0 else 'right', va='center', fontsize=11, fontweight='bold')

    ax4.set_xlim(min(improvements) - 3, max(improvements) + 5)
    plt.tight_layout()
    fig4.savefig(output_dir / "tma_improvement_summary.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: results_official/tma_improvement_summary.png")

    print("\nDone! Generated 4 figures in results_official/")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY (Official Harmonypy vs Evolved)")
    print("=" * 60)
    print(f"\nMixing Score:")
    print(f"  Official: {metrics['Official Harmony']['mixing_score']:.4f}")
    print(f"  Evolved:  {metrics['Evolved Harmony']['mixing_score']:.4f}")
    print(f"  Change:   {(metrics['Evolved Harmony']['mixing_score'] - metrics['Official Harmony']['mixing_score']) / metrics['Official Harmony']['mixing_score'] * 100:+.2f}%")

    print(f"\nBio Conservation:")
    print(f"  Official: {metrics['Official Harmony']['bio_score']:.4f}")
    print(f"  Evolved:  {metrics['Evolved Harmony']['bio_score']:.4f}")
    print(f"  Change:   {(metrics['Evolved Harmony']['bio_score'] - metrics['Official Harmony']['bio_score']) / metrics['Official Harmony']['bio_score'] * 100:+.2f}%")

    print(f"\nCombined Score:")
    print(f"  Official: {combined_official:.4f}")
    print(f"  Evolved:  {combined_evolved:.4f}")
    print(f"  Change:   {(combined_evolved - combined_official) / combined_official * 100:+.2f}%")

    print(f"\nExecution Time:")
    print(f"  Official: {time_official:.2f}s")
    print(f"  Evolved:  {time_evolved:.2f}s")
    if speedup >= 1:
        print(f"  Speedup:  {speedup:.2f}x faster")
    else:
        print(f"  Slowdown: {1/speedup:.2f}x slower")

    plt.show()
    return metrics


if __name__ == "__main__":
    run_comparison()
