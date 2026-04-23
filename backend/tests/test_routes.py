"""
test_routes.py — integration tests for FastAPI endpoints

Run with:
    cd d:/projects/AttentionSeeker/backend
    pytest tests/test_routes.py -v
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture(scope="module")
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ── /health ────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_health_returns_ok(client):
    res = await client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["model"] == "all-MiniLM-L6-v2"


# ── /encode ────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_encode_returns_tokens(client):
    res = await client.post("/encode", json={"sentence": "Hello world"})
    assert res.status_code == 200
    body = res.json()
    assert "data" in body
    assert "tokens" in body["data"]
    assert len(body["data"]["tokens"]) > 0


@pytest.mark.anyio
async def test_encode_has_correct_keys(client):
    res = await client.post("/encode", json={"sentence": "Testing the encoder"})
    assert res.status_code == 200
    data = res.json()["data"]
    for key in ("tokens", "input_ids", "token_embeds", "pos_embeds", "layer_outputs", "attentions", "pooled_embed"):
        assert key in data, f"Missing key: {key}"


@pytest.mark.anyio
async def test_encode_empty_sentence_returns_422(client):
    res = await client.post("/encode", json={"sentence": ""})
    assert res.status_code == 422


@pytest.mark.anyio
async def test_encode_whitespace_only_returns_422(client):
    res = await client.post("/encode", json={"sentence": "   "})
    assert res.status_code == 422


@pytest.mark.anyio
async def test_encode_with_layer_param(client):
    """?layer=3 should only return layer_3 in layer_outputs."""
    res = await client.post("/encode", json={"sentence": "Just one layer please", "layer": 3})
    assert res.status_code == 200
    layer_outputs = res.json()["data"]["layer_outputs"]
    assert list(layer_outputs.keys()) == ["layer_3"]


@pytest.mark.anyio
async def test_encode_token_count_matches(client):
    """token_count in root must match len(tokens) in data."""
    res = await client.post("/encode", json={"sentence": "The quick brown fox"})
    body = res.json()
    assert body["token_count"] == len(body["data"]["tokens"])


# ── /similarity ────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_similarity_returns_score(client):
    res = await client.post(
        "/similarity",
        json={"sentence_a": "I love dogs", "sentence_b": "I love cats"},
    )
    assert res.status_code == 200
    body = res.json()
    assert "cosine_similarity" in body
    score = body["cosine_similarity"]
    assert -1.0 <= score <= 1.0, f"Cosine score out of range: {score}"


@pytest.mark.anyio
async def test_similarity_identical_sentences(client):
    """Two identical sentences should have cosine similarity ≈ 1.0."""
    res = await client.post(
        "/similarity",
        json={"sentence_a": "Hello world", "sentence_b": "Hello world"},
    )
    assert res.status_code == 200
    score = res.json()["cosine_similarity"]
    assert score > 0.99, f"Expected ~1.0, got {score}"


@pytest.mark.anyio
async def test_similarity_returns_umap_coords(client):
    """UMAP/PCA coords must include 2 focal + 8 reference = 10 points."""
    res = await client.post(
        "/similarity",
        json={"sentence_a": "Machine learning", "sentence_b": "Deep learning"},
    )
    assert res.status_code == 200
    coords = res.json()["umap_coords"]
    assert len(coords) == 10
    focal = [c for c in coords if not c["is_reference"]]
    refs = [c for c in coords if c["is_reference"]]
    assert len(focal) == 2
    assert len(refs) == 8


@pytest.mark.anyio
async def test_similarity_empty_sentence_returns_422(client):
    res = await client.post(
        "/similarity",
        json={"sentence_a": "", "sentence_b": "Something"},
    )
    assert res.status_code == 422
