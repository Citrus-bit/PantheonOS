"""
CLI entry point for Pantheon Evolution.

Usage:
    python -m pantheon.evolution --initial program.py --evaluator evaluator.py \
        --objective "Optimize for speed" --iterations 100 --output results/
"""

import argparse
import asyncio
import sys
from pathlib import Path

from .config import EvolutionConfig
from .program import CodebaseSnapshot
from .team import EvolutionTeam


def main():
    parser = argparse.ArgumentParser(
        description="Evolutionary code optimization using LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evolve a single file
  python -m pantheon.evolution --initial sort.py --evaluator eval.py \\
      --objective "Optimize sorting speed"

  # Evolve a directory
  python -m pantheon.evolution --initial ./src --evaluator eval.py \\
      --objective "Improve performance" --iterations 200

  # Use configuration file
  python -m pantheon.evolution --initial program.py --evaluator eval.py \\
      --config evolution.yaml --output results/
        """,
    )

    # Required arguments
    parser.add_argument(
        "--initial", "-i",
        required=True,
        help="Initial code file or directory to evolve",
    )
    parser.add_argument(
        "--evaluator", "-e",
        required=True,
        help="Python file with evaluate(workspace_path) function",
    )
    parser.add_argument(
        "--objective", "-o",
        required=True,
        help="Natural language optimization objective",
    )

    # Optional arguments
    parser.add_argument(
        "--config", "-c",
        help="YAML configuration file",
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=100,
        help="Maximum number of iterations (default: 100)",
    )
    parser.add_argument(
        "--output", "-O",
        help="Output directory for results",
    )
    parser.add_argument(
        "--islands",
        type=int,
        default=3,
        help="Number of evolution islands (default: 3)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Maximum parallel evaluations (default: 4)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Model for mutation generation",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Load or create configuration
    if args.config:
        config = EvolutionConfig.from_yaml(args.config)
    else:
        config = EvolutionConfig()

    # Apply CLI overrides
    config = config.with_updates(
        max_iterations=args.iterations,
        num_islands=args.islands,
        max_parallel_evaluations=args.parallel,
        mutator_model=args.model,
        db_path=args.output,
        log_level="DEBUG" if args.verbose else "INFO",
    )

    # Validate configuration
    warnings = config.validate()
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)

    # Load initial code
    initial_path = Path(args.initial)
    if initial_path.is_dir():
        initial_code = CodebaseSnapshot.from_directory(str(initial_path))
        print(f"Loaded {initial_code.file_count()} files from {initial_path}")
    elif initial_path.is_file():
        initial_code = initial_path.read_text()
        print(f"Loaded {len(initial_code)} bytes from {initial_path}")
    else:
        print(f"Error: {initial_path} not found", file=sys.stderr)
        sys.exit(1)

    # Load evaluator code
    evaluator_path = Path(args.evaluator)
    if not evaluator_path.is_file():
        print(f"Error: evaluator file {evaluator_path} not found", file=sys.stderr)
        sys.exit(1)
    evaluator_code = evaluator_path.read_text()

    # Run evolution
    print(f"\nStarting evolution:")
    print(f"  Objective: {args.objective}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Islands: {args.islands}")
    print(f"  Model: {args.model}")
    print()

    async def run():
        team = EvolutionTeam(config=config)
        result = await team.evolve(
            initial_code=initial_code,
            evaluator_code=evaluator_code,
            objective=args.objective,
        )
        return result

    try:
        result = asyncio.run(run())
    except KeyboardInterrupt:
        print("\nEvolution interrupted by user")
        sys.exit(1)

    # Print results
    print("\n" + result.get_summary())

    # Save results if output specified
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save report
        result.save_report(str(output_dir / "report.json"))
        print(f"\nReport saved to {output_dir / 'report.json'}")

        # Save best code
        if result.best_program:
            best_dir = output_dir / "best"
            result.best_program.snapshot.to_workspace(str(best_dir))
            print(f"Best code saved to {best_dir}")

        # Save configuration
        config.to_yaml(str(output_dir / "config.yaml"))

    print(f"\nBest score: {result.best_score:.4f}")


if __name__ == "__main__":
    main()
