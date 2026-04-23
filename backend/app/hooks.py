import torch
from typing import Dict, List, Tuple


def register_hooks(
    bert_model: torch.nn.Module,
) -> Tuple[Dict[str, torch.Tensor], List]:
    """
    Registers forward hooks on each BertLayer inside the underlying BertModel.

    Returns:
        activations: dict keyed 'layer_0' .. 'layer_N' holding hidden state tensors
        handles:     list of hook handles — must be removed after forward pass
    """
    activations: Dict[str, torch.Tensor] = {}
    handles: List = []

    for i, layer in enumerate(bert_model.encoder.layer):
        def make_hook(idx: int):
            def hook(module, input, output):
                # BertLayer output varies by transformers version + attn_implementation:
                #   Tuple mode:  output = (hidden_states, ...) → output[0] = (batch, seq, hidden)
                #   Tensor mode: output = hidden_states tensor  → output    = (batch, seq, hidden)
                # In both cases we want (seq_len, hidden_dim) stored per layer.
                hs = output[0] if isinstance(output, (tuple, list)) else output
                # hs is (batch, seq, hidden) or already (seq, hidden)
                if hs.dim() == 3:
                    hs = hs[0]   # squeeze batch dim → (seq, hidden)
                activations[f"layer_{idx}"] = hs.detach().half()
            return hook

        handles.append(layer.register_forward_hook(make_hook(i)))

    return activations, handles
