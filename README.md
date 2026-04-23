# AttentionSeeker
### *It's not you, it's your embeddings.*

> An interactive, step-by-step visualizer for how sentence transformers encode text into dense vector embeddings.

---

## What it does

AttentionSeeker walks you through **9 internal stages** of `all-MiniLM-L6-v2` — from raw text to a normalized 384-dimensional sentence embedding — with live, interactive visualizations at every step.

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

- Python **3.12** (PyTorch does **not** support Python 3.14 yet)
- Node.js ≥ 18

### Backend

```bash
cd backend

# Create and activate a Python 3.12 venv
py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt

python -m uvicorn app.main:app --reload --port 8000
```

The first run downloads `all-MiniLM-L6-v2` (~90 MB). Wait for `✅ Model ready.` before encoding.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**

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

## API

### `GET /health`
```json
{ "status": "ok", "model": "all-MiniLM-L6-v2" }
```

### `POST /encode`
```json
{ "sentence": "Hello world", "layer": 3 }   // layer is optional
```
Returns tokens, embeddings, hidden states, attention weights, and pooled vector.

### `POST /similarity`
```json
{ "sentence_a": "I love dogs", "sentence_b": "I love cats" }
```
Returns cosine similarity score + 2D UMAP/PCA projection of 10 sentences.

---

## Tests

```bash
# Backend (requires Python 3.12 venv with requirements installed)
cd backend
pytest tests/ -v

# Frontend
cd frontend
npm run test
```

Backend route tests run without torch via a conftest stub. Encoder unit tests auto-skip when torch isn't available.