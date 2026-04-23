---
title: AttentionSeeker
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# AttentionSeeker

**AttentionSeeker** is a step-by-step, interactive explainer for how sentence transformers turn raw text into dense vector embeddings — built on `all-MiniLM-L6-v2`.

> *It's not you, it's your embeddings.*

---

## Demo

> 📸 *Add a GIF or screenshot here — the attention heatmap or UMAP scatter makes a great hero visual.*

---

## The 9 stages

AttentionSeeker walks you through every internal stage of `all-MiniLM-L6-v2` — from raw text to a normalized 384-dimensional sentence embedding — with live, interactive visualizations at each step.

| Step | What you see |
|------|-------------|
| 1 · Raw Text | Live token-count estimate as you type |
| 2 · Tokenization | WordPiece token pills — click any to inspect its 384-dim histogram |
| 3 · Token Embeddings | Per-token lookup vectors (bar chart) |
| 4 · Positional Embeddings | Position-dependent offset vectors |
| 5 · Transformer Layers ×6 | Hidden states after each transformer layer |
| 6 · Attention Weights | Interactive heatmap — pick any of 6 layers × 12 heads |
| 7 · Mean Pooling | Animated collapse of all token vectors into one |
| 8 · L2 Normalization | Unit-length pooled vector |
| 9 · Similarity | Cosine score dial + UMAP/PCA scatter of your sentences vs reference corpus |

---

## Stack

| Layer | Tech |
|-------|------|
| Model | `sentence-transformers/all-MiniLM-L6-v2` — 22M params, 384-dim |
| Backend | Python 3.12 · FastAPI · PyTorch · `sentence-transformers` · `umap-learn` |
| Frontend | React 18 · Vite · D3.js · `@visx/heatmap` · framer-motion · Zustand |

---

## Running locally

### Requirements

- Python **3.12** — PyTorch does not yet support 3.13+
- Node.js ≥ 18

### Backend

```bash
cd backend

# Create and activate a Python 3.12 virtual environment
python3.12 -m venv .venv

source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows

# conda alternative:
# conda create -n attentionseeker python=3.12 && conda activate attentionseeker

pip install -r requirements.txt

python -m uvicorn app.main:app --reload --port 8000
```

The first run downloads `all-MiniLM-L6-v2` (~90 MB). Wait for `✅ Model ready.` before sending requests.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**

### Tests

```bash
# Backend — route tests run without torch via a conftest stub;
# encoder unit tests auto-skip when torch is unavailable
cd backend
pytest tests/ -v

# Frontend
cd frontend
npm run test
```

---

## Project structure

```
AttentionSeeker/
├── backend/
│   ├── app/
│   │   ├── main.py       # FastAPI app + lifespan model loading
│   │   ├── encoder.py    # Singleton model + hook-based internals extractor
│   │   ├── hooks.py      # PyTorch forward hooks for hidden states
│   │   ├── routes.py     # /health  /encode  /similarity
│   │   └── models.py     # Pydantic request/response schemas
│   ├── tests/
│   │   ├── conftest.py   # Torch stub for lightweight CI
│   │   ├── test_encoder.py
│   │   └── test_routes.py
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── components/   # TokenView · AttentionHeatmap · EmbeddingBar · PoolingAnim · SimilarityView · Skeleton · InfoCard
    │   ├── store/        # Zustand global state
    │   ├── api/          # Axios client
    │   └── App.jsx
    └── vite.config.js
```

---

## API reference

### `GET /health`
```json
{ "status": "ok", "model": "all-MiniLM-L6-v2" }
```

### `POST /encode`
```json
{ "sentence": "Hello world", "layer": 3 }
```
`layer` is optional. Returns tokens, embeddings, hidden states, attention weights, and the pooled vector.

### `POST /similarity`
```json
{ "sentence_a": "I love dogs", "sentence_b": "I love cats" }
```
Returns cosine similarity score + 2D UMAP/PCA projection of 10 sentences.

---

## License

MIT · contributions welcome via pull request.