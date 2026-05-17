"""
Autouse fixture that swaps the layers.py stubs for working dummies, so the
surrounding infrastructure (model.py, train.py, eval.py, ablation.py) can be
tested end-to-end before layers.py is implemented.

sys.path is set via pyproject.toml [tool.pytest.ini_options] pythonpath.

When layers.py is implemented, delete this fixture and tests/_dummies.py —
the rest of the suite will run against the real code unchanged.
"""

from __future__ import annotations

import pytest

from _dummies import DummyBlock  # noqa: E402


@pytest.fixture(autouse=True)
def patch_layers(monkeypatch):
    """
    Replace still-stubbed layers with working dummies so the rest of the
    infrastructure (model.py / train.py / eval.py / ablation.py) can be
    exercised end-to-end. As real implementations land in layers.py, the
    corresponding patch lines come out of this fixture.

    Currently real (not patched): causal_mask, RMSNorm.
    Still stubs (patched here):   TransformerBlock (also pulls MHA via the block).
    """
    import layers
    import model as model_mod

    monkeypatch.setattr(layers, "TransformerBlock", DummyBlock)
    monkeypatch.setattr(model_mod, "TransformerBlock", DummyBlock)
    yield
