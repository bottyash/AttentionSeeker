"""
test_routes.py — integration tests for FastAPI endpoints

Run with:
    cd d:/projects/AttentionSeeker/backend
    .venv/Scripts/pytest tests/test_routes.py -v

When torch is NOT installed, the conftest.py stubs the encoder so these
route-level tests can still run. When torch IS installed, all tests use
the real model.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture(scope="module")
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ── /health ────────────────────────────────────────────────────────

async def test_health_returns_ok(client):
    res = await client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["model"] == "all-MiniLM-L6-v2"


# ── /encode ────────────────────────────────────────────────────────

async def test_encode_returns_tokens(client):
    res = await client.post("/encode", json={"sentence": "Hello world"})
    assert res.status_code == 200
    body = res.json()
    assert "data" in body
    assert "tokens" in body["data"]
    assert len(body["data"]["tokens"]) > 0


async def test_encode_has_correct_keys(client):
    res = await client.post("/encode", json={"sentence": "Testing the encoder"})
    assert res.status_code == 200
    data = res.json()["data"]
    for key in ("tokens", "input_ids", "token_embeds", "pos_embeds",
                "layer_outputs", "attentions", "pooled_embed"):
        assert key in data, f"Missing key: {key}"


async def test_encode_empty_sentence_returns_422(client):
    res = await client.post("/encode", json={"sentence": ""})
    assert res.status_code == 422


async def test_encode_whitespace_only_returns_422(client):
    res = await client.post("/encode", json={"sentence": "   "})
    assert res.status_code == 422


async def test_encode_with_layer_param(client):
    res = await client.post("/encode", json={"sentence": "Just one layer please", "layer": 3})
    assert res.status_code == 200
    layer_outputs = res.json()["data"]["layer_outputs"]
    assert list(layer_outputs.keys()) == ["layer_3"]


async def test_encode_token_count_matches(client):
    res = await client.post("/encode", json={"sentence": "The quick brown fox"})
    body = res.json()
    assert body["token_count"] == len(body["data"]["tokens"])


# ── /similarity ────────────────────────────────────────────────────

async def test_similarity_returns_score(client):
    res = await client.post(
        "/similarity",
        json={"sentence_a": "I love dogs", "sentence_b": "I love cats"},
    )
    assert res.status_code == 200
    score = res.json()["cosine_similarity"]
    assert -1.0 <= score <= 1.0


async def test_similarity_returns_umap_coords(client):
    res = await client.post(
        "/similarity",
        json={"sentence_a": "Machine learning", "sentence_b": "Deep learning"},
    )
    assert res.status_code == 200
    coords = res.json()["umap_coords"]
    assert len(coords) == 10
    assert len([c for c in coords if not c["is_reference"]]) == 2
    assert len([c for c in coords if c["is_reference"]]) == 8


async def test_similarity_empty_sentence_returns_422(client):
    res = await client.post(
        "/similarity",
        json={"sentence_a": "", "sentence_b": "Something"},
    )
    assert res.status_code == 422
