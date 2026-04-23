"""
conftest.py — backend test configuration

Strategy:
  - Route tests (/encode, /similarity, /health) stub out the heavy ML
    modules (torch, sentence_transformers) so they can run in CI or
    lightweight environments without a 2+ GB install.
  - Encoder unit tests (test_encoder.py) are auto-skipped when torch
    isn't installed, since they need the real model to be meaningful.
"""

import sys
from unittest.mock import MagicMock
import pytest

# ── Check whether the real ML stack is available ──────────────────
_TORCH_AVAILABLE = True
try:
    import torch  # noqa: F401
except ImportError:
    _TORCH_AVAILABLE = False


def pytest_collection_modifyitems(config, items):
    """Auto-skip test_encoder.py if torch isn't installed."""
    skip_no_torch = pytest.mark.skip(
        reason="torch not installed — run `pip install -r requirements.txt` for full tests"
    )
    for item in items:
        if "test_encoder" in item.nodeid and not _TORCH_AVAILABLE:
            item.add_marker(skip_no_torch)


# ── Stub heavy ML deps BEFORE app modules are imported ────────────
# This lets test_routes.py import `app.main` cleanly even without torch.
if not _TORCH_AVAILABLE:
    # Build a torch stub with the attributes the app actually uses
    torch_mock = MagicMock()
    torch_mock.no_grad.return_value.__enter__ = lambda s: s
    torch_mock.no_grad.return_value.__exit__ = MagicMock(return_value=False)
    torch_mock.arange = MagicMock(return_value=MagicMock())
    torch_mock.stack = MagicMock(return_value=MagicMock())
    torch_mock.dot = MagicMock(return_value=MagicMock(item=lambda: 0.85))
    torch_mock.tensor = MagicMock(return_value=MagicMock())
    torch_mock.nn = MagicMock()

    st_mock = MagicMock()
    umap_mock = MagicMock()
    sklearn_mock = MagicMock()

    sys.modules.setdefault("torch", torch_mock)
    sys.modules.setdefault("torch.nn", torch_mock.nn)
    sys.modules.setdefault("torch.nn.functional", MagicMock())
    sys.modules.setdefault("sentence_transformers", st_mock)
    sys.modules.setdefault("transformers", MagicMock())
    sys.modules.setdefault("umap", umap_mock)
    sys.modules.setdefault("sklearn", sklearn_mock)
    sys.modules.setdefault("sklearn.decomposition", sklearn_mock.decomposition)

    # Stub get_model and encode_with_internals so routes return valid payloads
    # without touching real model weights
    import types

    encoder_stub = types.ModuleType("app.encoder")

    _FAKE_TOKENS = ["[CLS]", "hello", "world", "[SEP]"]
    _FAKE_EMBED = [0.01] * 384

    def _fake_encode(sentence, requested_layer=None):
        layers = {f"layer_{i}": [_FAKE_EMBED] * len(_FAKE_TOKENS) for i in range(6)}
        if requested_layer is not None:
            layers = {f"layer_{requested_layer}": layers[f"layer_{requested_layer}"]}
        return {
            "input_ids": [101, 7592, 2088, 102],
            "tokens": _FAKE_TOKENS,
            "token_embeds": [_FAKE_EMBED] * len(_FAKE_TOKENS),
            "pos_embeds": [_FAKE_EMBED] * len(_FAKE_TOKENS),
            "layer_outputs": layers,
            "attentions": [
                [[[1 / len(_FAKE_TOKENS)] * len(_FAKE_TOKENS)] * len(_FAKE_TOKENS)] * 12
            ] * 6,
            "pooled_embed": _FAKE_EMBED,
        }

    class _FakeTensor(list):
        def item(self):
            return 0.85

    def _fake_final_embedding(sentence):
        t = MagicMock()
        t.__iter__ = lambda s: iter(_FAKE_EMBED)
        t.numpy = lambda: __import__("numpy").array(_FAKE_EMBED) if True else None
        return t

    encoder_stub.get_model = MagicMock()
    encoder_stub.encode_with_internals = _fake_encode
    encoder_stub.get_final_embedding = _fake_final_embedding
    sys.modules["app.encoder"] = encoder_stub
