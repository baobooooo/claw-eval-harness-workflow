"""Generic harness/benchmark/model evaluation runner.

This package is the refactored layer on top of the original Codex + SWE-bench
experiment scripts.  It deliberately keeps benchmark preparation, harness
execution, and model endpoint configuration separate so one can compare harness
choices while keeping the model and task set fixed.
"""

__all__ = ["__version__"]
__version__ = "0.2.0"
