"""Run the full reproducible credit early-warning pipeline."""

from __future__ import annotations

import subprocess
import sys

PIPELINE_MODULES = [
    "src.data_panel",
    "src.eda",
    "src.target",
    "src.features",
    "src.model",
    "src.staging",
    "src.watchlist",
    "src.explain",
]


def run_pipeline(modules: list[str] | None = None) -> None:
    """Run each pipeline module in order with the active Python interpreter."""
    for module in modules or PIPELINE_MODULES:
        print(f"\n=== Running {module} ===", flush=True)
        subprocess.run([sys.executable, "-m", module], check=True)


def main() -> None:
    """Command-line entrypoint for the full project pipeline."""
    run_pipeline()


if __name__ == "__main__":
    main()
