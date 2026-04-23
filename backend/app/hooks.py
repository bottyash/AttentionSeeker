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
                # output[0] is the hidden states tensor: (batch, seq_len, hidden)
                # Store as float16 to halve payload size
                activations[f"layer_{idx}"] = output[0].detach().half()
            return hook

        handles.append(layer.register_forward_hook(make_hook(i)))

    return activations, handles
