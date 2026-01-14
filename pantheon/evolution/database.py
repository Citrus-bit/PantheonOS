"""
Evolution database with MAP-Elites support.

Stores evolved programs with quality-diversity archiving and multi-island evolution.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pantheon.utils.log import logger

from .config import EvolutionConfig
from .program import CodebaseSnapshot, Program


@dataclass
class ExplorationRecord:
    """Record of a single exploration attempt."""
    iteration: int
    direction: str                    # One-sentence description of the IMPLEMENTED change
    category: str                     # Category: objective_function, optimization_method, etc.
    result: str                       # "success", "marginal", "neutral", "failure", "error"
    score_delta: float                # Score change (child - parent)
    parent_generation: int            # Generation of the parent program
    is_algorithmic: bool = True       # Whether this is an algorithmic change (vs code optimization)
    match_confidence: str = "medium"  # How well the direction matches the diff: "high", "medium", "low"
    error_message: Optional[str] = None  # Error message if result is "error"
    timestamp: float = 0.0            # When this exploration was attempted

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "direction": self.direction,
            "category": self.category,
            "result": self.result,
            "score_delta": self.score_delta,
            "parent_generation": self.parent_generation,
            "is_algorithmic": self.is_algorithmic,
            "match_confidence": self.match_confidence,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExplorationRecord":
        return cls(
            iteration=data.get("iteration", 0),
            direction=data.get("direction", ""),
            category=data.get("category", "other"),
            result=data.get("result", "neutral"),
            score_delta=data.get("score_delta", 0.0),
            parent_generation=data.get("parent_generation", 0),
            is_algorithmic=data.get("is_algorithmic", True),
            match_confidence=data.get("match_confidence", "medium"),
            error_message=data.get("error_message"),
            timestamp=data.get("timestamp", 0.0),
        )


@dataclass
class ExplorationHistory:
    """History of exploration attempts for avoiding repetition."""
    records: List[ExplorationRecord] = field(default_factory=list)

    def add_record(self, record: ExplorationRecord) -> None:
        """Add a new exploration record."""
        self.records.append(record)

    def get_direction_stats(self, direction: str) -> Dict[str, int]:
        """Get statistics for a specific direction (or similar)."""
        # Simple exact match for now, could use fuzzy matching later
        matching = [r for r in self.records if r.direction.lower() == direction.lower()]
        return {
            "attempts": len(matching),
            "successes": sum(1 for r in matching if r.result == "success"),
            "failures": sum(1 for r in matching if r.result == "failure"),
            "errors": sum(1 for r in matching if r.result == "error"),
        }

    def get_successful_directions(self, max_count: int = 5) -> List[str]:
        """Get list of successful directions (deduplicated)."""
        seen = set()
        result = []
        for r in reversed(self.records):  # Most recent first
            if r.result in ("success", "marginal") and r.is_algorithmic:
                direction_key = r.direction.lower()
                if direction_key not in seen:
                    seen.add(direction_key)
                    result.append(r.direction)
                    if len(result) >= max_count:
                        break
        return result

    def get_failed_directions(self, max_count: int = 5) -> List[str]:
        """Get list of failed directions (deduplicated)."""
        seen = set()
        result = []
        for r in reversed(self.records):
            if r.result == "failure" and r.is_algorithmic:
                direction_key = r.direction.lower()
                if direction_key not in seen:
                    seen.add(direction_key)
                    result.append(r.direction)
                    if len(result) >= max_count:
                        break
        return result

    def get_recent_records(self, max_count: int = 10, algorithmic_only: bool = True) -> List[ExplorationRecord]:
        """Get recent exploration records."""
        if algorithmic_only:
            filtered = [r for r in self.records if r.is_algorithmic]
        else:
            filtered = self.records
        return filtered[-max_count:]

    def format_for_prompt(
        self,
        max_recent: int = 10,
        max_directions: int = 5,
        max_direction_chars: int = 80,
        max_total_chars: int = 2000,
    ) -> str:
        """
        Format exploration history for inclusion in analyzer prompt.

        Args:
            max_recent: Maximum number of recent records to show
            max_directions: Maximum number of successful/failed directions to show
            max_direction_chars: Maximum characters per direction string
            max_total_chars: Maximum total characters for the entire history text

        Returns:
            Formatted history string, truncated if necessary
        """
        def truncate(s: str, max_len: int) -> str:
            if len(s) <= max_len:
                return s
            return s[:max_len-3] + "..."

        lines = []

        # Recent attempts
        recent = self.get_recent_records(max_recent, algorithmic_only=True)
        if recent:
            lines.append("### Recent Exploration Attempts")
            for r in recent:
                icon = {"success": "✓", "marginal": "○", "neutral": "·", "failure": "✗", "error": "⚠"}.get(r.result, "?")
                delta_str = f"{r.score_delta:+.1%}" if r.score_delta != 0 else "0%"
                direction_short = truncate(r.direction, max_direction_chars)
                lines.append(f"- {icon} \"{direction_short}\" ({delta_str})")
            lines.append("")

        # Successful directions
        successful = self.get_successful_directions(max_directions)
        if successful:
            lines.append("### Successful Directions (worth exploring further)")
            for d in successful:
                lines.append(f"- {truncate(d, max_direction_chars)}")
            lines.append("")

        # Failed directions
        failed = self.get_failed_directions(max_directions)
        if failed:
            lines.append("### Failed Directions (avoid these)")
            for d in failed:
                lines.append(f"- {truncate(d, max_direction_chars)}")
            lines.append("")

        if lines:
            lines.insert(0, "## Exploration History\n")
            lines.append("IMPORTANT: Propose something DIFFERENT from the above attempts. Do not repeat failed directions.")

        result = "\n".join(lines)

        # Truncate if over total limit
        if len(result) > max_total_chars:
            result = result[:max_total_chars-50] + "\n\n... (history truncated)"

        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "records": [r.to_dict() for r in self.records],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExplorationHistory":
        records = [ExplorationRecord.from_dict(r) for r in data.get("records", [])]
        return cls(records=records)


@dataclass
class EvolutionDatabase:
    """
    Program database with MAP-Elites and island-based evolution.

    Supports:
    - Multi-island populations for diversity
    - MAP-Elites grid for quality-diversity
    - Elite archive for exploitation
    - Various sampling strategies
    """

    config: EvolutionConfig = field(default_factory=EvolutionConfig)

    # Storage
    programs: Dict[str, Program] = field(default_factory=dict)
    islands: List[Set[str]] = field(default_factory=list)
    island_feature_maps: List[Dict[Tuple[int, ...], str]] = field(default_factory=list)
    archive: Set[str] = field(default_factory=set)
    best_program_id: Optional[str] = None

    # Statistics
    total_added: int = 0
    total_improved: int = 0

    # Exploration history for avoiding repeated attempts
    exploration_history: ExplorationHistory = field(default_factory=ExplorationHistory)

    # Observed feature ranges: {feature_name: (min_value, max_value)}
    feature_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Sequence counter for program ordering
    _next_order: int = 0

    # Thread safety lock for async operations
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self):
        """Initialize islands if not already set."""
        if not self.islands:
            self.islands = [set() for _ in range(self.config.num_islands)]
            self.island_feature_maps = [{} for _ in range(self.config.num_islands)]

    def add(
        self,
        program: Program,
        target_island: Optional[int] = None,
        reference_codes: Optional[List[str]] = None,
    ) -> bool:
        """
        Add a program to the database.

        Places program in MAP-Elites grid and updates archive if elite.

        Args:
            program: Program to add
            target_island: Specific island to add to (random if None)
            reference_codes: Reference codes for diversity calculation

        Returns:
            True if program was added (might replace existing)
        """
        # Assign sequential order number
        program.order = self._next_order
        self._next_order += 1

        self.total_added += 1

        # Store program
        self.programs[program.id] = program

        # Determine target island
        if target_island is None:
            target_island = random.randint(0, self.config.num_islands - 1)
        program.island_id = target_island

        # Add to island population
        self.islands[target_island].add(program.id)

        # Update observed feature ranges before computing bin
        self._update_feature_ranges(program)

        # Get effective ranges for bin calculation
        effective_ranges = self.get_effective_feature_ranges()

        # Calculate feature coordinates and bin
        feature_bin = program.feature_bin(
            self.config.feature_dimensions,
            self.config.feature_bins,
            reference_codes,
            feature_ranges=effective_ranges,
        )

        # MAP-Elites: check if this bin already has a program
        feature_map = self.island_feature_maps[target_island]
        existing_id = feature_map.get(feature_bin)

        added = False
        if existing_id is None:
            # Empty bin, add directly
            feature_map[feature_bin] = program.id
            added = True
        else:
            # Compare fitness
            existing = self.programs.get(existing_id)
            if existing:
                new_fitness = program.fitness_score(self.config.feature_dimensions)
                old_fitness = existing.fitness_score(self.config.feature_dimensions)
                if new_fitness > old_fitness:
                    feature_map[feature_bin] = program.id
                    added = True
                    self.total_improved += 1

        # Update best program
        self._update_best(program)

        # Update archive
        self._update_archive(program)

        # Log if improvement
        if added and self.config.log_improvements:
            fitness = program.fitness_score(self.config.feature_dimensions)
            logger.debug(
                f"Added program {program.id[:8]} to island {target_island}, "
                f"bin {feature_bin}, fitness {fitness:.4f}"
            )

        return added

    def _update_best(self, program: Program) -> None:
        """Update best program tracking."""
        if self.best_program_id is None:
            self.best_program_id = program.id
            return

        best = self.programs.get(self.best_program_id)
        if best is None:
            self.best_program_id = program.id
            return

        new_fitness = program.fitness_score(self.config.feature_dimensions)
        best_fitness = best.fitness_score(self.config.feature_dimensions)

        if new_fitness > best_fitness:
            self.best_program_id = program.id
            logger.info(
                f"New best program: {program.id[:8]} with fitness {new_fitness:.4f} "
                f"(previous: {best_fitness:.4f})"
            )

    def _update_archive(self, program: Program) -> None:
        """Update elite archive to keep top X% of programs by fitness."""
        self.archive.add(program.id)

        # Calculate dynamic archive size based on ratio
        total_programs = len(self.programs)
        target_size = max(1, int(total_programs * self.config.archive_ratio))

        # Trim archive if over target size
        if len(self.archive) > target_size:
            # Remove lowest fitness programs
            archive_programs = [
                (pid, self.programs[pid].fitness_score(self.config.feature_dimensions))
                for pid in self.archive
                if pid in self.programs
            ]
            archive_programs.sort(key=lambda x: x[1], reverse=True)

            self.archive = set(pid for pid, _ in archive_programs[:target_size])

    def _update_feature_ranges(self, program: Program) -> None:
        """Update observed min/max for each feature dimension."""
        coords = program.feature_coordinates(self.config.feature_dimensions)
        for dim, value in coords.items():
            if dim not in self.feature_ranges:
                self.feature_ranges[dim] = (value, value)
            else:
                old_min, old_max = self.feature_ranges[dim]
                self.feature_ranges[dim] = (min(old_min, value), max(old_max, value))

    def get_feature_range(self, dim: str) -> Tuple[float, float]:
        """
        Get effective range for a feature dimension with padding.

        Returns:
            Tuple of (min_value, max_value) with padding applied
        """
        if not self.config.feature_range_adaptive or dim not in self.feature_ranges:
            # Use default 0-1 range
            return (0.0, 1.0)

        min_val, max_val = self.feature_ranges[dim]
        range_size = max_val - min_val

        # Add padding
        padding = range_size * self.config.feature_range_padding
        padded_min = max(0.0, min_val - padding)
        padded_max = min(1.0, max_val + padding)

        # Ensure minimum range to avoid division by zero
        if padded_max - padded_min < 0.01:
            padded_min = max(0.0, min_val - 0.05)
            padded_max = min(1.0, max_val + 0.05)

        return (padded_min, padded_max)

    def get_effective_feature_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get effective ranges for all feature dimensions."""
        return {
            dim: self.get_feature_range(dim)
            for dim in self.config.feature_dimensions
        }

    def sample(
        self,
        num_inspirations: int = 2,
        island_id: Optional[int] = None,
    ) -> Tuple[Program, List[Program]]:
        """
        Sample a parent program and inspiration programs.

        Uses exploration/exploitation ratio to balance sampling strategy.

        Args:
            num_inspirations: Number of inspiration programs to sample
            island_id: Specific island to sample from (random if None)

        Returns:
            Tuple of (parent_program, list_of_inspirations)
        """
        if not self.programs:
            raise ValueError("Cannot sample from empty database")

        # Select parent using strategy
        parent = self._sample_parent(island_id)

        # Sample inspirations (diverse programs)
        inspirations = self._sample_inspirations(num_inspirations, exclude={parent.id})

        return parent, inspirations

    async def add_async(
        self,
        program: Program,
        target_island: Optional[int] = None,
        reference_codes: Optional[List[str]] = None,
    ) -> bool:
        """
        Thread-safe async version of add().

        Args:
            program: Program to add
            target_island: Specific island to add to (random if None)
            reference_codes: Reference codes for diversity calculation

        Returns:
            True if program was added (might replace existing)
        """
        async with self._lock:
            return self.add(program, target_island, reference_codes)

    async def sample_async(
        self,
        num_inspirations: int = 2,
        island_id: Optional[int] = None,
    ) -> Tuple[Program, List[Program]]:
        """
        Thread-safe async version of sample().

        Args:
            num_inspirations: Number of inspiration programs to sample
            island_id: Specific island to sample from (random if None)

        Returns:
            Tuple of (parent_program, list_of_inspirations)
        """
        async with self._lock:
            return self.sample(num_inspirations, island_id)

    def _sample_parent(self, island_id: Optional[int] = None) -> Program:
        """Sample a parent program using configured strategy."""
        rand_val = random.random()

        if rand_val < self.config.exploration_ratio:
            # Random sampling for exploration
            return self._sample_random(island_id)
        elif rand_val < self.config.exploration_ratio + self.config.exploitation_ratio:
            # Elite sampling for exploitation
            return self._sample_from_archive()
        else:
            # Fitness-weighted sampling
            return self._sample_weighted(island_id)

    def _sample_random(self, island_id: Optional[int] = None) -> Program:
        """Sample random program from population."""
        if island_id is not None and self.islands[island_id]:
            program_id = random.choice(list(self.islands[island_id]))
        else:
            program_id = random.choice(list(self.programs.keys()))
        return self.programs[program_id]

    def _sample_from_archive(self) -> Program:
        """Sample from elite archive with fitness-weighted probability."""
        if not self.archive:
            return self._sample_random()

        # Calculate fitness weights for archive members
        weights = []
        valid_candidates = []
        for pid in self.archive:
            program = self.programs.get(pid)
            if program:
                fitness = program.fitness_score(self.config.feature_dimensions)
                # Use fitness as weight (higher fitness = higher probability)
                # Add small epsilon to avoid zero weights
                weights.append(max(fitness, 0.001))
                valid_candidates.append(pid)

        if not valid_candidates:
            return self._sample_random()

        # Weighted random selection
        selected_id = random.choices(valid_candidates, weights=weights, k=1)[0]
        return self.programs[selected_id]

    def _sample_weighted(self, island_id: Optional[int] = None) -> Program:
        """Sample program weighted by fitness."""
        if island_id is not None and self.islands[island_id]:
            candidates = list(self.islands[island_id])
        else:
            candidates = list(self.programs.keys())

        if not candidates:
            return self._sample_random()

        # Calculate fitness weights
        weights = []
        valid_candidates = []
        for pid in candidates:
            program = self.programs.get(pid)
            if program:
                fitness = program.fitness_score(self.config.feature_dimensions)
                # Add small epsilon to avoid zero weights
                weights.append(max(fitness, 0.001))
                valid_candidates.append(pid)

        if not valid_candidates:
            return self._sample_random()

        # Weighted random selection
        total = sum(weights)
        weights = [w / total for w in weights]
        selected_id = random.choices(valid_candidates, weights=weights, k=1)[0]
        return self.programs[selected_id]

    def _sample_inspirations(
        self,
        num: int,
        exclude: Optional[Set[str]] = None,
    ) -> List[Program]:
        """Sample diverse inspiration programs."""
        exclude = exclude or set()
        inspirations = []

        # Sample from different islands for diversity
        available_islands = list(range(self.config.num_islands))
        random.shuffle(available_islands)

        for island_id in available_islands:
            if len(inspirations) >= num:
                break

            island_programs = self.islands[island_id] - exclude
            if island_programs:
                # Sample from this island's feature map for diversity
                feature_map = self.island_feature_maps[island_id]
                if feature_map:
                    # Sample from different bins
                    bins = list(feature_map.keys())
                    random.shuffle(bins)
                    for bin_key in bins:
                        if len(inspirations) >= num:
                            break
                        pid = feature_map[bin_key]
                        if pid not in exclude and pid in self.programs:
                            inspirations.append(self.programs[pid])
                            exclude.add(pid)

        # If still need more, sample randomly
        remaining = num - len(inspirations)
        if remaining > 0:
            all_ids = set(self.programs.keys()) - exclude
            sample_ids = random.sample(list(all_ids), min(remaining, len(all_ids)))
            for pid in sample_ids:
                if pid in self.programs:
                    inspirations.append(self.programs[pid])

        return inspirations

    def get_best_program(self) -> Optional[Program]:
        """Get the best program found so far."""
        if self.best_program_id:
            return self.programs.get(self.best_program_id)
        return None

    def get_top_programs(
        self,
        n: int = 5,
        metric: str = "combined_score",
        island_id: Optional[int] = None,
    ) -> List[Program]:
        """
        Get top N programs by metric.

        Args:
            n: Number of programs to return
            metric: Metric name to sort by
            island_id: Specific island (all if None)

        Returns:
            List of top programs
        """
        if island_id is not None:
            candidates = [
                self.programs[pid]
                for pid in self.islands[island_id]
                if pid in self.programs
            ]
        else:
            candidates = list(self.programs.values())

        # Sort by metric
        def get_metric(p: Program) -> float:
            if metric == "fitness":
                return p.fitness_score(self.config.feature_dimensions)
            return p.metrics.get(metric, 0.0)

        candidates.sort(key=get_metric, reverse=True)
        return candidates[:n]

    def migrate(self, migration_rate: Optional[float] = None) -> int:
        """
        Perform island migration.

        Copies top programs between islands for genetic diversity.

        Args:
            migration_rate: Fraction of programs to migrate (uses config if None)

        Returns:
            Number of programs migrated
        """
        if self.config.num_islands < 2:
            return 0

        migration_rate = migration_rate or self.config.migration_rate
        migrated = 0

        for source_island in range(self.config.num_islands):
            # Get top programs from source island
            top_programs = self.get_top_programs(
                n=max(1, int(len(self.islands[source_island]) * migration_rate)),
                island_id=source_island,
            )

            # Migrate to next island (ring topology)
            target_island = (source_island + 1) % self.config.num_islands

            for program in top_programs:
                # Don't duplicate, just add reference to target island
                self.islands[target_island].add(program.id)
                migrated += 1

        logger.debug(f"Migration complete: {migrated} programs migrated")
        return migrated

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        fitness_values = [
            p.fitness_score(self.config.feature_dimensions)
            for p in self.programs.values()
        ]

        return {
            "total_programs": len(self.programs),
            "total_added": self.total_added,
            "total_improved": self.total_improved,
            "archive_size": len(self.archive),
            "num_islands": self.config.num_islands,
            "island_sizes": [len(island) for island in self.islands],
            "feature_map_sizes": [len(fm) for fm in self.island_feature_maps],
            "best_fitness": max(fitness_values) if fitness_values else 0.0,
            "avg_fitness": sum(fitness_values) / len(fitness_values) if fitness_values else 0.0,
            "min_fitness": min(fitness_values) if fitness_values else 0.0,
        }

    def save(self, path: str) -> None:
        """
        Save database to directory.

        Args:
            path: Directory to save to
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "config": self.config.to_dict(),
            "best_program_id": self.best_program_id,
            "archive": list(self.archive),
            "islands": [list(island) for island in self.islands],
            "total_added": self.total_added,
            "total_improved": self.total_improved,
            "feature_ranges": {k: list(v) for k, v in self.feature_ranges.items()},
            "next_order": self._next_order,
        }
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save programs
        programs_dir = save_dir / "programs"
        programs_dir.mkdir(exist_ok=True)

        for program_id, program in self.programs.items():
            program.save(str(programs_dir / f"{program_id}.json"))

        # Save feature maps
        feature_maps_data = []
        for fm in self.island_feature_maps:
            # Convert tuple keys to strings for JSON
            fm_data = {str(k): v for k, v in fm.items()}
            feature_maps_data.append(fm_data)

        with open(save_dir / "feature_maps.json", "w") as f:
            json.dump(feature_maps_data, f, indent=2)

        # Save exploration history
        with open(save_dir / "exploration_history.json", "w") as f:
            json.dump(self.exploration_history.to_dict(), f, indent=2)

        logger.info(f"Saved database with {len(self.programs)} programs to {path}")

    @classmethod
    def load(cls, path: str) -> "EvolutionDatabase":
        """
        Load database from directory.

        Args:
            path: Directory to load from

        Returns:
            Loaded EvolutionDatabase
        """
        load_dir = Path(path)

        # Load metadata
        with open(load_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        config = EvolutionConfig.from_dict(metadata.get("config", {}))
        db = cls(config=config)

        db.best_program_id = metadata.get("best_program_id")
        db.archive = set(metadata.get("archive", []))
        db.islands = [set(island) for island in metadata.get("islands", [])]
        db.total_added = metadata.get("total_added", 0)
        db.total_improved = metadata.get("total_improved", 0)
        # Restore feature ranges
        feature_ranges_data = metadata.get("feature_ranges", {})
        db.feature_ranges = {k: tuple(v) for k, v in feature_ranges_data.items()}
        # Restore sequence counter
        db._next_order = metadata.get("next_order", 0)

        # Load programs
        programs_dir = load_dir / "programs"
        if programs_dir.exists():
            for program_file in programs_dir.glob("*.json"):
                program = Program.load(str(program_file))
                db.programs[program.id] = program

        # Load feature maps
        feature_maps_path = load_dir / "feature_maps.json"
        if feature_maps_path.exists():
            with open(feature_maps_path, "r") as f:
                feature_maps_data = json.load(f)

            db.island_feature_maps = []
            for fm_data in feature_maps_data:
                # Convert string keys back to tuples
                fm = {}
                for k, v in fm_data.items():
                    # Parse tuple from string like "(1, 2, 3)"
                    key = tuple(int(x) for x in k.strip("()").split(",") if x.strip())
                    fm[key] = v
                db.island_feature_maps.append(fm)

        # Ensure correct number of islands
        while len(db.islands) < config.num_islands:
            db.islands.append(set())
        while len(db.island_feature_maps) < config.num_islands:
            db.island_feature_maps.append({})

        # Load exploration history
        exploration_history_path = load_dir / "exploration_history.json"
        if exploration_history_path.exists():
            with open(exploration_history_path, "r") as f:
                history_data = json.load(f)
            db.exploration_history = ExplorationHistory.from_dict(history_data)

        logger.info(f"Loaded database with {len(db.programs)} programs from {path}")
        return db
