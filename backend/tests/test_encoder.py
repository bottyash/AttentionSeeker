"""
test_encoder.py — unit tests for encode_with_internals()

Run with:
    cd d:/projects/AttentionSeeker/backend
    pytest tests/test_encoder.py -v
"""

import math
import pytest
from app.encoder import encode_with_internals, get_model

KNOWN_SENTENCE = "The cat sat on the mat"


@pytest.fixture(scope="module")
def result():
    """Encode once; share across all tests in this module."""
    return encode_with_internals(KNOWN_SENTENCE)


# ── Tokenization ───────────────────────────────────────────────────

def test_tokens_present(result):
    """Result must include a non-empty token list."""
    assert "tokens" in result
    assert len(result["tokens"]) > 0


def test_cls_and_sep_present(result):
    """WordPiece always wraps with [CLS] ... [SEP]."""
    assert result["tokens"][0] == "[CLS]"
    assert result["tokens"][-1] == "[SEP]"


def test_token_count_matches_input_ids(result):
    """tokens and input_ids must be the same length."""
    assert len(result["tokens"]) == len(result["input_ids"])


# ── Embedding shapes ───────────────────────────────────────────────

def test_token_embeds_shape(result):
    """token_embeds: (seq_len, 384)"""
    seq_len = len(result["tokens"])
    assert len(result["token_embeds"]) == seq_len
    assert all(len(v) == 384 for v in result["token_embeds"])


def test_pos_embeds_shape(result):
    """pos_embeds: (seq_len, 384)"""
    seq_len = len(result["tokens"])
    assert len(result["pos_embeds"]) == seq_len
    assert all(len(v) == 384 for v in result["pos_embeds"])


# ── Layer outputs ──────────────────────────────────────────────────

def test_all_six_layers_present(result):
    """MiniLM-L6 has exactly 6 transformer layers."""
    assert len(result["layer_outputs"]) == 6
    for i in range(6):
        assert f"layer_{i}" in result["layer_outputs"]


def test_layer_output_shapes(result):
    """Each layer output: (seq_len, 384)"""
    seq_len = len(result["tokens"])
    for i in range(6):
        layer = result["layer_outputs"][f"layer_{i}"]
        assert len(layer) == seq_len, f"layer_{i} has wrong seq_len"
        assert all(len(row) == 384 for row in layer), f"layer_{i} has wrong hidden dim"


# ── Attention weights ──────────────────────────────────────────────

def test_attention_layers_count(result):
    """6 layers of attention."""
    assert len(result["attentions"]) == 6


def test_attention_heads_count(result):
    """12 heads per layer."""
    for layer_attn in result["attentions"]:
        assert len(layer_attn) == 12


def test_attention_shapes(result):
    """Attention: (6_layers, 12_heads, seq_len, seq_len)"""
    seq_len = len(result["tokens"])
    for layer_idx, layer_attn in enumerate(result["attentions"]):
        for head_idx, head in enumerate(layer_attn):
            assert len(head) == seq_len, f"layer {layer_idx} head {head_idx}: wrong rows"
            for row in head:
                assert len(row) == seq_len, f"layer {layer_idx} head {head_idx}: wrong cols"


def test_attention_sums_to_one(result):
    """Each attention row must sum to ~1.0 (softmax output)."""
    for layer_attn in result["attentions"]:
        for head in layer_attn:
            for row in head:
                total = sum(row)
                assert abs(total - 1.0) < 1e-4, f"Attention row sums to {total}"


# ── Pooled embedding ───────────────────────────────────────────────

def test_pooled_embed_dim(result):
    """Pooled embedding must be 384-dimensional."""
    assert len(result["pooled_embed"]) == 384


def test_pooled_embed_l2_norm(result):
    """Pooled embedding must be L2-normalized (norm ≈ 1.0)."""
    norm = math.sqrt(sum(v ** 2 for v in result["pooled_embed"]))
    assert abs(norm - 1.0) < 1e-5, f"L2 norm is {norm}, expected ~1.0"


# ── Lazy layer loading ─────────────────────────────────────────────

def test_requested_layer_returns_one_layer():
    """When requested_layer=2, only layer_2 should be in layer_outputs."""
    result = encode_with_internals(KNOWN_SENTENCE, requested_layer=2)
    assert list(result["layer_outputs"].keys()) == ["layer_2"]


def test_single_word_input():
    """Single-word input must still produce CLS, token, SEP (N=3)."""
    result = encode_with_internals("Hello")
    assert len(result["tokens"]) == 3
    assert result["tokens"][0] == "[CLS]"
    assert result["tokens"][-1] == "[SEP]"
