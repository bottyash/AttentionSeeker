# AttentionSeeker
### *It's not you, it's your embeddings.*

> An interactive, step-by-step visualizer for how sentence transformers encode text into dense vector embeddings.

<!-- Badges go here -->

---

## What it does

AttentionSeeker walks you through the 9 internal stages of `all-MiniLM-L6-v2` — from raw text all the way to a normalized 384-dimensional sentence embedding — with live, interactive visualizations at every step.

## Stack

- **Backend**: Python · FastAPI · PyTorch · sentence-transformers
- **Frontend**: React 18 · Vite · D3.js · @visx/heatmap · framer-motion
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (22M params, 384-dim)

## Status

🚧 Under active development