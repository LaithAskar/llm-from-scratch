"""
sys.path is set via pyproject.toml [tool.pytest.ini_options] pythonpath.

layers.py is fully implemented — no monkey-patching required. tests/_dummies.py
remains for now in case it's useful, but the autouse patch_layers fixture has
been removed.
"""

from __future__ import annotations
