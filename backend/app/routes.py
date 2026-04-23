"""
routes.py — FastAPI route definitions
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import torch
import numpy as np

from .models import EncodeRequest, EncodeResponse, TokenEmbedResponse, SimilarityRequest, SimilarityResponse
from .encoder import encode_with_internals, get_final_embedding

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    from sklearn.decomposition import PCA
    UMAP_AVAILABLE = False

router = APIRouter()

# Reference sentences for UMAP context in the similarity view
REFERENCE_SENTENCES = [
    "The weather is nice today.",
    "I love eating pizza.",
    "Machine learning is fascinating.",
    "The stock market crashed yesterday.",
    "She plays the violin beautifully.",
    "Quantum computing will change everything.",
    "The dog chased the ball across the yard.",
    "Paris is the capital of France.",
]


@router.get("/health")
def health():
    return {"status": "ok", "model": "all-MiniLM-L6-v2"}


@router.post("/encode", response_model=EncodeResponse)
def encode(req: EncodeRequest):
    sentence = req.sentence.strip()
    if not sentence:
        raise HTTPException(status_code=422, detail="Sentence must not be empty.")

    try:
        result = encode_with_internals(sentence, requested_layer=req.layer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    token_count = len(result["tokens"])

    return EncodeResponse(
        data=TokenEmbedResponse(
            tokens=result["tokens"],
            input_ids=result["input_ids"],
            token_embeds=result["token_embeds"],
            pos_embeds=result["pos_embeds"],
            layer_outputs=result["layer_outputs"],
            attentions=result["attentions"],
            pooled_embed=result["pooled_embed"],
        ),
        token_count=token_count,
    )


@router.post("/similarity", response_model=SimilarityResponse)
def similarity(req: SimilarityRequest):
    a = req.sentence_a.strip()
    b = req.sentence_b.strip()
    if not a or not b:
        raise HTTPException(status_code=422, detail="Both sentences must be non-empty.")

    emb_a = get_final_embedding(a)
    emb_b = get_final_embedding(b)
    cosine_sim = float(torch.dot(emb_a, emb_b).item())

    # Build embedding matrix: [a, b, ...8 references]
    ref_embeddings = [get_final_embedding(s) for s in REFERENCE_SENTENCES]
    all_embeddings = torch.stack([emb_a, emb_b] + ref_embeddings).numpy()

    # 2D projection
    if UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(all_embeddings)
    else:
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(all_embeddings)

    labels = [a, b] + REFERENCE_SENTENCES
    is_reference = [False, False] + [True] * len(REFERENCE_SENTENCES)

    umap_coords = [
        {"label": labels[i], "x": float(coords_2d[i][0]), "y": float(coords_2d[i][1]), "is_reference": is_reference[i]}
        for i in range(len(labels))
    ]

    return SimilarityResponse(
        cosine_similarity=cosine_sim,
        umap_coords=umap_coords,
    )
