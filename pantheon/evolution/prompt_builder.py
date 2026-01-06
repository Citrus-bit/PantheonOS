"""
Evolution prompt builder for mutation generation.

Constructs prompts for the mutator agent with:
- Current program state
- Top performing programs
- Diverse inspirations
- Evaluation feedback
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .program import Program


# System prompt for the mutator agent
MUTATION_SYSTEM_PROMPT = """You are an expert code optimizer. Your task is to improve code through targeted mutations.

## Your Role
Given a codebase and optimization objective, generate improved versions through careful modifications.

## Output Format
Use the following diff format to show your changes. You can make multiple changes across multiple files:

File: path/to/file.py
<<<<<<< SEARCH
original code section to find
=======
improved code to replace with
>>>>>>> REPLACE

## Guidelines
1. Make targeted, surgical improvements - don't rewrite everything
2. Preserve working functionality
3. Focus on the optimization objective
4. Learn from high-performing examples
5. Consider both correctness and performance
6. Handle edge cases properly
7. Follow coding best practices

## Important
- Each SEARCH block must exactly match existing code
- You can make multiple changes in one response
- Always specify the file path before each change block
- Preserve code structure and indentation
"""

MUTATION_SYSTEM_PROMPT_CODEBASE = """You are an expert code optimizer working on a multi-file codebase.

## Your Role
Improve the codebase through targeted modifications to achieve the optimization objective.

## Output Format
For each change, specify the file and use SEARCH/REPLACE blocks:

File: path/to/file.py
<<<<<<< SEARCH
original code to find
=======
improved code
>>>>>>> REPLACE

You can:
- Modify existing files (SEARCH/REPLACE)
- Create new files (use empty SEARCH block)
- Delete files (use empty REPLACE block)

## Guidelines
1. Make minimal, targeted changes
2. Maintain consistency across files
3. Don't break imports or dependencies
4. Test your changes mentally
5. Follow the existing code style
6. Consider the whole codebase architecture
"""


class EvolutionPromptBuilder:
    """
    Builds prompts for the mutation agent.

    Combines current program, top performers, inspirations,
    and feedback into a structured prompt.
    """

    def __init__(
        self,
        max_code_length: int = 10000,
        max_top_programs: int = 3,
        max_inspirations: int = 2,
        include_artifacts: bool = True,
        max_artifact_length: int = 1000,
    ):
        """
        Initialize prompt builder.

        Args:
            max_code_length: Maximum code characters per program
            max_top_programs: Maximum number of top programs to include
            max_inspirations: Maximum number of inspiration programs
            include_artifacts: Whether to include evaluation artifacts
            max_artifact_length: Maximum artifact text length
        """
        self.max_code_length = max_code_length
        self.max_top_programs = max_top_programs
        self.max_inspirations = max_inspirations
        self.include_artifacts = include_artifacts
        self.max_artifact_length = max_artifact_length

    def get_system_prompt(self, is_codebase: bool = True) -> str:
        """Get the appropriate system prompt."""
        return MUTATION_SYSTEM_PROMPT_CODEBASE if is_codebase else MUTATION_SYSTEM_PROMPT

    def build_mutation_prompt(
        self,
        parent: Program,
        objective: str,
        top_programs: Optional[List[Program]] = None,
        inspirations: Optional[List[Program]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        iteration: Optional[int] = None,
    ) -> str:
        """
        Build a mutation prompt.

        Args:
            parent: Parent program to mutate
            objective: Optimization objective
            top_programs: Best performing programs
            inspirations: Diverse inspiration programs
            artifacts: Evaluation artifacts/feedback
            iteration: Current iteration number

        Returns:
            Formatted prompt string
        """
        parts = []

        # Header with objective
        parts.append(self._build_objective_section(objective, iteration))

        # Current program
        parts.append(self._build_current_program_section(parent))

        # Top performers
        if top_programs:
            parts.append(self._build_top_programs_section(top_programs))

        # Inspirations
        if inspirations:
            parts.append(self._build_inspirations_section(inspirations))

        # Artifacts/feedback
        if artifacts and self.include_artifacts:
            parts.append(self._build_artifacts_section(artifacts))

        # Task instructions
        parts.append(self._build_task_section())

        return "\n\n".join(parts)

    def _build_objective_section(
        self,
        objective: str,
        iteration: Optional[int] = None,
    ) -> str:
        """Build the objective section."""
        header = "## Optimization Objective"
        if iteration is not None:
            header += f" (Iteration {iteration})"
        return f"{header}\n\n{objective}"

    def _build_current_program_section(self, program: Program) -> str:
        """Build the current program section."""
        combined = program.metrics.get("combined_score", 0)
        parts = [f"## Current Program (Combined Score: {combined:.4f})"]

        # Show all detailed metrics
        if program.metrics:
            metrics_lines = []
            for key, value in sorted(program.metrics.items()):
                if key != "combined_score" and isinstance(value, (int, float)):
                    metrics_lines.append(f"  - {key}: {value:.4f}")
            if metrics_lines:
                parts.append("\nDetailed Metrics:")
                parts.extend(metrics_lines)

        # Add file listing
        parts.append(f"\nFiles: {program.file_count()} | Lines: {program.total_lines()}")

        # Add code for each file
        for path, content in sorted(program.snapshot.files.items()):
            truncated = self._truncate_code(content)
            parts.append(f"\n### {path}")
            parts.append(f"```python\n{truncated}\n```")

        return "\n".join(parts)

    def _build_top_programs_section(self, programs: List[Program]) -> str:
        """Build the top performers section."""
        parts = ["## Top Performing Programs"]
        parts.append("Learn from these high-scoring examples:\n")

        for i, prog in enumerate(programs[: self.max_top_programs]):
            combined = prog.metrics.get("combined_score", 0)
            parts.append(f"### #{i+1} (Combined Score: {combined:.4f})")

            # Show detailed metrics
            if prog.metrics:
                metrics_lines = []
                for key, value in sorted(prog.metrics.items()):
                    if key != "combined_score" and isinstance(value, (int, float)):
                        metrics_lines.append(f"  - {key}: {value:.4f}")
                if metrics_lines:
                    parts.append("Metrics:")
                    parts.extend(metrics_lines)

            # Show key differences or summary
            if prog.diff_from_parent:
                diff_preview = prog.diff_from_parent[:500]
                parts.append(f"Key changes:\n```diff\n{diff_preview}\n```")
            else:
                # Show code summary
                summary = prog.snapshot.to_summary(max_files=2, max_lines_per_file=30)
                parts.append(summary)

        return "\n\n".join(parts)

    def _build_inspirations_section(self, programs: List[Program]) -> str:
        """Build the inspirations section."""
        parts = ["## Diverse Inspirations"]
        parts.append("Consider these alternative approaches:\n")

        for i, prog in enumerate(programs[: self.max_inspirations]):
            combined = prog.metrics.get("combined_score", 0)
            parts.append(f"### Inspiration {i+1} (Combined Score: {combined:.4f})")

            # Show detailed metrics
            if prog.metrics:
                metrics_lines = []
                for key, value in sorted(prog.metrics.items()):
                    if key != "combined_score" and isinstance(value, (int, float)):
                        metrics_lines.append(f"  - {key}: {value:.4f}")
                if metrics_lines:
                    parts.append("Metrics:")
                    parts.extend(metrics_lines)

            # Show structural summary
            summary = prog.snapshot.to_summary(max_files=2, max_lines_per_file=20)
            parts.append(summary)

        return "\n\n".join(parts)

    def _build_artifacts_section(self, artifacts: Dict[str, Any]) -> str:
        """Build the artifacts/feedback section."""
        parts = ["## Evaluation Feedback"]

        if "llm_feedback" in artifacts and artifacts["llm_feedback"]:
            feedback = str(artifacts["llm_feedback"])[: self.max_artifact_length]
            parts.append(f"Previous assessment: {feedback}")

        if "issues" in artifacts and artifacts["issues"]:
            issues = artifacts["issues"]
            if isinstance(issues, list):
                issues_text = "\n".join(f"- {issue}" for issue in issues[:5])
            else:
                issues_text = str(issues)[: self.max_artifact_length]
            parts.append(f"Issues found:\n{issues_text}")

        if "suggestions" in artifacts and artifacts["suggestions"]:
            suggestions = artifacts["suggestions"]
            if isinstance(suggestions, list):
                suggestions_text = "\n".join(f"- {s}" for s in suggestions[:5])
            else:
                suggestions_text = str(suggestions)[: self.max_artifact_length]
            parts.append(f"Suggestions:\n{suggestions_text}")

        if "evaluation_error" in artifacts:
            error = str(artifacts["evaluation_error"])[: self.max_artifact_length]
            parts.append(f"⚠️ Evaluation error: {error}")

        if "stderr" in artifacts and artifacts["stderr"]:
            stderr = str(artifacts["stderr"])[: self.max_artifact_length]
            parts.append(f"⚠️ Runtime output:\n```\n{stderr}\n```")

        return "\n\n".join(parts)

    def _build_task_section(self) -> str:
        """Build the task instructions section."""
        return """## Your Task

Generate an improved version of the current program. Use SEARCH/REPLACE blocks to show your changes.

Remember:
- Focus on the optimization objective
- Make targeted improvements
- Learn from top performers
- Address any issues mentioned in feedback

Provide your changes now:"""

    def _truncate_code(self, code: str) -> str:
        """Truncate code to max length."""
        if len(code) <= self.max_code_length:
            return code

        # Try to truncate at a sensible point
        truncated = code[: self.max_code_length]
        last_newline = truncated.rfind("\n")
        if last_newline > self.max_code_length * 0.8:
            truncated = truncated[:last_newline]

        return truncated + "\n# ... (truncated)"


def build_simple_prompt(
    code: str,
    objective: str,
    feedback: Optional[str] = None,
) -> str:
    """
    Build a simple mutation prompt for single-file evolution.

    Args:
        code: Current code
        objective: Optimization objective
        feedback: Optional feedback from previous evaluation

    Returns:
        Formatted prompt
    """
    parts = [
        f"## Objective\n{objective}",
        f"## Current Code\n```python\n{code}\n```",
    ]

    if feedback:
        parts.append(f"## Feedback\n{feedback}")

    parts.append("""## Task
Improve this code. Use SEARCH/REPLACE format:

<<<<<<< SEARCH
original code
=======
improved code
>>>>>>> REPLACE
""")

    return "\n\n".join(parts)
