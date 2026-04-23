"""
encoder.py — model loading and intermediate state extraction

Singleton pattern: the SentenceTransformer is loaded once at startup
and reused across all requests. Never instantiate per-request.
"""

import torch
from sentence_transformers import SentenceTransformer
from .hooks import register_hooks

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        # Force eager attention — SDPA (default in newer transformers) does not
        # support output_attentions=True which we need for the heatmap.
        bert = _get_bert(_model)
        bert.config._attn_implementation = "eager"
    return _model



def _get_bert(model: SentenceTransformer):
    """Navigate the SentenceTransformer wrapper to the underlying BertModel."""
    return model._modules["0"].auto_model


def _get_tokenizer(model: SentenceTransformer):
    return model._modules["0"].tokenizer


def encode_with_internals(sentence: str, requested_layer: int | None = None) -> dict:
    """
    Encode a sentence and return all intermediate representations:
      - tokens, input_ids
      - token_embeds (word embedding lookup, float32)
      - pos_embeds   (positional embedding lookup, float32)
      - layer_outputs (hidden states per layer, float16 to save bandwidth)
      - attentions    (attention weights per layer/head)
      - pooled_embed  (mean-pooled + L2-normalized final vector)
    """
    model = get_model()
    bert = _get_bert(model)
    tokenizer = _get_tokenizer(model)

    # Enable attention output
    bert.config.output_attentions = True

    # Tokenize
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    input_ids = inputs["input_ids"]  # shape: (1, seq_len)
    seq_len = input_ids.shape[1]

    # Register hooks BEFORE forward pass
    activations, handles = register_hooks(bert)

    with torch.no_grad():
        outputs = bert(**inputs)

    # Always remove hooks after inference
    for h in handles:
        h.remove()

    # ── Token & positional embeddings ──────────────────────────────────────────
    token_embeds = bert.embeddings.word_embeddings(input_ids)[0]          # (seq, 384)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    pos_embeds = bert.embeddings.position_embeddings(position_ids)[0]     # (seq, 384)

    # ── Layer outputs ──────────────────────────────────────────────────────────
    if requested_layer is not None:
        key = f"layer_{requested_layer}"
        layer_outputs = {key: activations[key].tolist()}
    else:
        layer_outputs = {k: v.tolist() for k, v in activations.items()}

    # ── Mean pooling + L2 norm ─────────────────────────────────────────────────
    last_hidden = outputs.last_hidden_state[0]               # (seq_len, 384)
    pooled = last_hidden.mean(dim=0)                         # (384,)
    pooled_norm = torch.nn.functional.normalize(pooled.unsqueeze(0), dim=1)[0]

    return {
        "input_ids": input_ids[0].tolist(),
        "tokens": tokenizer.convert_ids_to_tokens(input_ids[0]),
        "token_embeds": token_embeds.tolist(),
        "pos_embeds": pos_embeds.tolist(),
        "layer_outputs": layer_outputs,
        "attentions": [a[0].tolist() for a in outputs.attentions],  # list[layer][head][row][col]
        "pooled_embed": pooled_norm.tolist(),
    }


def get_final_embedding(sentence: str) -> torch.Tensor:
    """Returns just the pooled, normalized 384-dim embedding (for similarity)."""
    model = get_model()
    with torch.no_grad():
        return torch.tensor(model.encode(sentence, normalize_embeddings=True))
