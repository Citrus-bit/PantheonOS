#!/usr/bin/env python
"""
Compare Original vs Optimized Harmony Implementations.

This script demonstrates the improvements that Pantheon Evolution can achieve
by comparing the original and optimized versions of the Harmony algorithm.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from evaluator import generate_test_data, compute_batch_mixing_score, compute_bio_conservation_score, compute_convergence_score


def evaluate_implementation(harmony_module, X, batch_labels, true_labels, n_runs=3):
    """Evaluate a Harmony implementation with multiple runs."""
    execution_times = []
    mixing_scores = []
    bio_scores = []
    convergence_scores = []
    iterations_list = []

    for i in range(n_runs):
        start = time.time()
        hm = harmony_module.run_harmony(
            X.copy(),
            batch_labels.copy(),
            n_clusters=50,
            max_iter=10,
            random_state=42 + i,
        )
        elapsed = time.time() - start

        execution_times.append(elapsed)
        mixing_scores.append(compute_batch_mixing_score(hm.Z_corr, batch_labels))
        bio_scores.append(compute_bio_conservation_score(hm.Z_corr, X, true_labels))
        convergence_scores.append(compute_convergence_score(hm.objectives))
        iterations_list.append(len(hm.objectives))

    return {
        "execution_time": np.mean(execution_times),
        "execution_time_std": np.std(execution_times),
        "mixing_score": np.mean(mixing_scores),
        "mixing_score_std": np.std(mixing_scores),
        "bio_score": np.mean(bio_scores),
        "bio_score_std": np.std(bio_scores),
        "convergence_score": np.mean(convergence_scores),
        "iterations": np.mean(iterations_list),
    }


def compute_combined_score(metrics):
    """Compute combined score using evaluator weights."""
    speed_score = 1.0 / (1 + metrics["execution_time"])
    return (
        0.4 * metrics["mixing_score"] +
        0.3 * metrics["bio_score"] +
        0.2 * speed_score +
        0.1 * metrics["convergence_score"]
    )


def main():
    print("=" * 70)
    print("Pantheon Evolution: Harmony Algorithm Comparison")
    print("=" * 70)
    print()

    # Generate test data
    print("Generating synthetic single-cell data...")
    print("  - 2000 cells, 50 features, 3 batches, 5 biological clusters")
    print()

    X, batch_labels, true_labels = generate_test_data(
        n_cells=2000,
        n_features=50,
        n_batches=3,
        n_clusters=5,
        random_state=42,
    )

    # Import implementations
    import harmony as original
    import harmony_optimized as optimized

    # Evaluate original
    print("Evaluating ORIGINAL implementation (3 runs)...")
    original_metrics = evaluate_implementation(original, X, batch_labels, true_labels)
    original_combined = compute_combined_score(original_metrics)

    # Evaluate optimized
    print("Evaluating OPTIMIZED implementation (3 runs)...")
    optimized_metrics = evaluate_implementation(optimized, X, batch_labels, true_labels)
    optimized_combined = compute_combined_score(optimized_metrics)

    # Print comparison
    print()
    print("=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print()

    print(f"{'Metric':<25} {'Original':>15} {'Optimized':>15} {'Improvement':>15}")
    print("-" * 70)

    # Execution time
    orig_time = original_metrics["execution_time"]
    opt_time = optimized_metrics["execution_time"]
    speedup = orig_time / opt_time if opt_time > 0 else 0
    print(f"{'Execution Time (s)':<25} {orig_time:>14.3f}s {opt_time:>14.3f}s {speedup:>14.2f}x")

    # Iterations
    orig_iter = original_metrics["iterations"]
    opt_iter = optimized_metrics["iterations"]
    iter_reduction = (orig_iter - opt_iter) / orig_iter * 100 if orig_iter > 0 else 0
    print(f"{'Iterations':<25} {orig_iter:>15.1f} {opt_iter:>15.1f} {iter_reduction:>14.1f}%")

    # Mixing score
    orig_mix = original_metrics["mixing_score"]
    opt_mix = optimized_metrics["mixing_score"]
    mix_change = (opt_mix - orig_mix) / orig_mix * 100 if orig_mix > 0 else 0
    print(f"{'Batch Mixing Score':<25} {orig_mix:>15.4f} {opt_mix:>15.4f} {mix_change:>+14.1f}%")

    # Bio conservation
    orig_bio = original_metrics["bio_score"]
    opt_bio = optimized_metrics["bio_score"]
    bio_change = (opt_bio - orig_bio) / orig_bio * 100 if orig_bio > 0 else 0
    print(f"{'Bio Conservation Score':<25} {orig_bio:>15.4f} {opt_bio:>15.4f} {bio_change:>+14.1f}%")

    # Convergence
    orig_conv = original_metrics["convergence_score"]
    opt_conv = optimized_metrics["convergence_score"]
    conv_change = (opt_conv - orig_conv) / orig_conv * 100 if orig_conv > 0 else 0
    print(f"{'Convergence Score':<25} {orig_conv:>15.4f} {opt_conv:>15.4f} {conv_change:>+14.1f}%")

    print("-" * 70)

    # Combined scores
    combined_improvement = (optimized_combined - original_combined) / original_combined * 100
    print(f"{'COMBINED SCORE':<25} {original_combined:>15.4f} {optimized_combined:>15.4f} {combined_improvement:>+14.1f}%")

    print()
    print("=" * 70)
    print("KEY OPTIMIZATIONS APPLIED:")
    print("=" * 70)
    print("""
1. Pre-computed Squared Norms
   - Cache ||z||^2 and ||y||^2 to avoid redundant computations
   - Reduces distance calculation from O(n*d*k) to O(n*k)

2. Batched Cluster Corrections
   - Process all clusters together instead of per-cluster loops
   - Uses weighted regression for global batch effect estimation

3. Adaptive Early Termination
   - Track relative objective change with patience counter
   - Stop when convergence is stable for multiple iterations

4. Vectorized Diversity Penalty
   - Fully vectorized batch mixing computation
   - Clamp extreme values for numerical stability

5. Cholesky Decomposition
   - Use Cholesky instead of general solve for positive definite systems
   - About 2x faster for ridge regression
""")

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Starting Score:    {original_combined:.4f}
Final Score:       {optimized_combined:.4f}
Improvement:       {combined_improvement:+.1f}%
Speedup:           {speedup:.2f}x faster
""")

    return {
        "original": original_metrics,
        "optimized": optimized_metrics,
        "improvement": combined_improvement,
        "speedup": speedup,
    }


if __name__ == "__main__":
    main()
