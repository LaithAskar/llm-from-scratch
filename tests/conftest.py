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

from _dummies import DummyBlock, DummyRMSNorm, dummy_causal_mask


@pytest.fixture(autouse=True)
def patch_layers(monkeypatch):
    import layers
    import model as model_mod

    monkeypatch.setattr(layers, "TransformerBlock", DummyBlock)
    monkeypatch.setattr(layers, "RMSNorm", DummyRMSNorm)
    monkeypatch.setattr(layers, "causal_mask", dummy_causal_mask)
    # model.py imported these symbols at module-load time; patch its namespace too.
    monkeypatch.setattr(model_mod, "TransformerBlock", DummyBlock)
    monkeypatch.setattr(model_mod, "RMSNorm", DummyRMSNorm)
    monkeypatch.setattr(model_mod, "causal_mask", dummy_causal_mask)
    yield
