#!/usr/bin/env python
"""Run the comparison in a subprocess to avoid DLL issues."""

import subprocess
import sys

if __name__ == "__main__":
    # Run plot_comparison_official.py in a completely fresh subprocess
    result = subprocess.run(
        [sys.executable, "plot_comparison_official.py"],
        cwd=r"C:\Users\wzxu\Desktop\Pantheon\pantheon-agents-2\examples\evolution_harmonypy",
        capture_output=False,
        env=None,  # Use default environment
    )
    sys.exit(result.returncode)
