"""
Evolution Batch Correction Examples.

This package contains examples of using Pantheon Evolution to optimize
batch correction algorithms for single-cell data analysis:

- evolution_harmonypy: Evolving the Harmony algorithm
- evolution_scanorama: Evolving the Scanorama algorithm

Shared components:
- metrics.py: Common evaluation metrics (batch mixing, bio conservation)
- data/: Shared TMA datasets for evaluation
"""

from .metrics import (
    load_tma_data,
    compute_batch_mixing_score,
    compute_bio_conservation_score,
    get_default_fitness_weights,
)

__all__ = [
    "load_tma_data",
    "compute_batch_mixing_score",
    "compute_bio_conservation_score",
    "get_default_fitness_weights",
]
