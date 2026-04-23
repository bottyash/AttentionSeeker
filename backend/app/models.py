from pydantic import BaseModel
from typing import List, Dict, Optional


class EncodeRequest(BaseModel):
    sentence: str
    layer: Optional[int] = None  # if specified, only return this layer's hidden states


class TokenEmbedResponse(BaseModel):
    tokens: List[str]
    input_ids: List[int]
    token_embeds: List[List[float]]
    pos_embeds: List[List[float]]
    layer_outputs: Dict[str, List[List[float]]]  # keyed "layer_0" … "layer_5"
    attentions: List[List[List[List[float]]]]     # [layer][head][row][col]
    pooled_embed: List[float]


class EncodeResponse(BaseModel):
    data: TokenEmbedResponse
    token_count: int


class SimilarityRequest(BaseModel):
    sentence_a: str
    sentence_b: str


class SimilarityResponse(BaseModel):
    cosine_similarity: float
    umap_coords: List[Dict]   # [{label, x, y, is_reference}, ...]
