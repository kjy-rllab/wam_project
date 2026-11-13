"""Qwen3.5/3.8 internals for probing: what each layer's token mixer reads from every earlier token.

A Qwen3.5-style language model alternates three Gated DeltaNet layers (linear attention, a recurrent state) with one
gated softmax attention layer. Both write, at a query token t, a sum over source tokens i <= t:

    gated attention   out_t = W_O [ sigmoid(gate_t) * sum_i a_ti v_i ]             a_ti: softmax attention
    Gated DeltaNet    out_t = W_O [ scale_t * sum_i w_ti v_i ]                     w_ti: implicit attention

The DeltaNet state is S_i = exp(g_i) (I - beta_i k_i k_i^T) S_{i-1} + beta_i k_i v_i^T and the layer reads q_t^T S_t, so
w_ti = beta_i k_i^T r_i with r_t = q_t and r_{i-1} = exp(g_i) (I - beta_i k_i k_i^T) r_i: signed, not normalized, and
exact (scale_t is the gated RMS norm at t, a per-head rescaling). `trace` returns these weights for chosen query
tokens, with the size of what each source token adds to the residual stream.
"""
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, causal_conv1d_fn, l2norm


def load(path: str, device_map: str | dict = 'cuda', dtype: torch.dtype = torch.bfloat16, weights: str | None = None) -> tuple:
    """weights: vlm.safetensors of a full-mode qwen_vla.train checkpoint, the fine-tuned VLM up to its tap (the layers
    above the tap keep the released weights, so only the residual stream up to the tap is the fine-tuned model's)."""
    processor = AutoProcessor.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype, device_map=device_map).eval()
    if weights:
        missing, unexpected = model.model.load_state_dict(load_file(weights), strict=False)
        assert not unexpected and all('language_model.layers.' in k or 'language_model.norm.' in k for k in missing), (missing[:3], unexpected[:3])
    # The patch embedding is a Conv3d whose kernel is the whole patch; in bf16 it can run 100x slower than this matmul
    proj = model.model.visual.patch_embed.proj
    model.model.visual.patch_embed.forward = lambda x: F.linear(x.to(proj.weight.dtype), proj.weight.flatten(1), proj.bias)
    return model, processor


def chat(processor: object, prompts: list[tuple[list, str]], text_first: bool = False) -> dict:
    """Batched inputs of user turns, each (images, text): left-padded, generation prompt added, thinking off."""
    messages = []
    for images, text in prompts:
        content = [{'type': 'image', 'image': image} for image in images]
        text = [{'type': 'text', 'text': text}]
        messages.append([{'role': 'user', 'content': text + content if text_first else content + text}])
    processor.tokenizer.padding_side = 'left'
    return processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=True,
        processor_kwargs={'padding': True, 'return_tensors': 'pt'},
    )


def image_tokens(model: object, inputs: dict) -> list[Tensor]:
    """Positions of every image's tokens in a single sequence, each as a [rows, cols] grid."""
    positions = (inputs['input_ids'][0] == model.config.image_token_id).nonzero()[:, 0]
    merge = model.config.vision_config.spatial_merge_size
    grids = [(int(h) // merge, int(w) // merge) for _, h, w in inputs['image_grid_thw']]
    return [p.reshape(g) for p, g in zip(positions.split([h * w for h, w in grids]), grids)]


def source_norms(x: Tensor, out_proj: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """x [T, heads, dim]: what every source token passes to one query token, before the output projection.
    Returns the norm of each head's write to the residual stream [heads, T], of all heads together [T], and the
    summed write [hidden] (the layer's output at the query token)."""
    tokens, heads, dim = x.shape
    gram = out_proj.T @ out_proj
    blocks = torch.stack([gram[h * dim:(h + 1) * dim, h * dim:(h + 1) * dim] for h in range(heads)])
    per_head = torch.einsum('thd,hde,the->ht', x, blocks, x).clamp(min=0).sqrt()
    flat = x.reshape(tokens, heads * dim)
    return per_head, ((flat @ gram) * flat).sum(-1).clamp(min=0).sqrt(), flat.sum(0) @ out_proj.T


@torch.no_grad()
def attention_rows(module: object, hidden: Tensor, position_embeddings: tuple, queries: Tensor) -> dict:
    """Gated softmax attention of one layer from the query tokens. hidden [T, D] is the layer's normed input."""
    tokens, head_dim = hidden.shape[0], module.head_dim
    hidden = hidden.float()
    query, gate = (hidden @ module.q_proj.weight.float().T).view(tokens, -1, 2 * head_dim).chunk(2, dim=-1)
    key = (hidden @ module.k_proj.weight.float().T).view(tokens, -1, head_dim)
    value = (hidden @ module.v_proj.weight.float().T).view(tokens, -1, head_dim)
    cos, sin = (p[0].float() for p in position_embeddings)
    query, key = apply_rotary_pos_emb(module.q_norm(query).transpose(0, 1), module.k_norm(key).transpose(0, 1), cos, sin, 0)
    key = key.repeat_interleave(module.num_key_value_groups, dim=0)
    value = value.repeat_interleave(module.num_key_value_groups, dim=1)

    logits = query[:, queries] @ key.transpose(1, 2) * module.scaling  # [heads, Q, T]
    future = torch.arange(tokens, device=hidden.device)[None] > queries[:, None]
    weights = logits.masked_fill(future, float('-inf')).softmax(-1)
    out = {'weights': weights, 'head_norm': [], 'norm': [], 'output': []}
    for q, t in enumerate(queries):
        x = weights[:, q].T[..., None] * value * gate[t].sigmoid()[None]
        for name, part in zip(('head_norm', 'norm', 'output'), source_norms(x, module.o_proj.weight.float())):
            out[name].append(part)
    return {k: torch.stack(v) if isinstance(v, list) else v for k, v in out.items()}


@torch.no_grad()
def delta_rows(module: object, hidden: Tensor, queries: Tensor) -> dict:
    """Implicit attention of one Gated DeltaNet layer from the query tokens. hidden [T, D] is its normed input."""
    tokens = hidden.shape[0]
    mixed = causal_conv1d_fn(
        module.in_proj_qkv(hidden[None]).transpose(1, 2), module.conv1d.weight.squeeze(1), module.conv1d.bias,
        activation=module.activation,
    )[0].T.float()
    query, key, value = mixed.split([module.key_dim, module.key_dim, module.value_dim], dim=-1)
    repeat = module.num_v_heads // module.num_k_heads
    query = l2norm(query.reshape(tokens, -1, module.head_k_dim)).repeat_interleave(repeat, dim=1) * module.head_k_dim**-0.5
    key = l2norm(key.reshape(tokens, -1, module.head_k_dim)).repeat_interleave(repeat, dim=1)
    value = value.reshape(tokens, -1, module.head_v_dim)
    beta = module.in_proj_b(hidden).float().sigmoid()  # [T, heads]
    decay = (-module.A_log.float().exp() * F.softplus(module.in_proj_a(hidden).float() + module.dt_bias.float())).exp()
    gate = F.silu(module.in_proj_z(hidden).float().reshape(tokens, -1, module.head_v_dim))

    # Backward over the sources: r starts as q_t at i = t, and passes through one state update per step
    weights = torch.zeros(len(queries), module.num_v_heads, tokens, device=hidden.device)
    r = torch.zeros(len(queries), module.num_v_heads, module.head_k_dim, device=hidden.device)
    for i in range(int(queries.max()), -1, -1):
        r[queries == i] = query[i]
        kr = (r * key[i]).sum(-1)
        weights[:, :, i] = beta[i] * kr
        r = decay[i][:, None] * (r - (beta[i] * kr)[..., None] * key[i])
    weights = weights.transpose(0, 1)  # [heads, Q, T]

    out = {'weights': weights, 'head_norm': [], 'norm': [], 'output': []}
    for q, t in enumerate(queries):
        x = weights[:, q].T[..., None] * value  # [T, heads, dim]
        rms = x.sum(0).pow(2).mean(-1, keepdim=True).add(module.norm.variance_epsilon).rsqrt()
        x = x * (rms * module.norm.weight.float() * gate[t])[None]
        for name, part in zip(('head_norm', 'norm', 'output'), source_norms(x, module.out_proj.weight.float())):
            out[name].append(part)
    return {k: torch.stack(v) if isinstance(v, list) else v for k, v in out.items()}


@torch.no_grad()
def trace(model: object, inputs: dict, queries: Tensor, check: bool = False) -> dict:
    """One forward pass of a single sequence. Returns 'hidden' [layers + 1, T, D] (the residual stream: embeddings,
    then after every layer) and, per layer, 'rows': weights [heads, Q, T] from the query tokens to every token,
    head_norm [Q, heads, T] and norm [Q, T] (sizes of what each source token adds to the residual stream at the
    query). With check, 'error' is the relative difference between the summed writes and the layer's real output."""
    layers = model.model.language_model.layers
    rows, handles = [None] * len(layers), []
    for index, layer in enumerate(layers):
        linear = layer.block_type == 'linear_attention'
        mixer = layer.linear_attn if linear else layer.self_attn

        def hook(module, args, kwargs, output, index=index, linear=linear):
            hidden = kwargs['hidden_states'][0]
            rows[index] = delta_rows(module, hidden, queries) if linear else attention_rows(
                module, hidden, kwargs['position_embeddings'], queries)
            real = (output if linear else output[0])[0, queries].float()
            rows[index]['error'] = (rows[index].pop('output') - real).norm(dim=-1) / real.norm(dim=-1)

        handles.append(mixer.register_forward_hook(hook, with_kwargs=True))
    try:
        outputs = model.model(**inputs, output_hidden_states=True)
    finally:
        for handle in handles:
            handle.remove()
    if not check:
        for row in rows:
            del row['error']
    return {'hidden': torch.stack([h[0] for h in outputs.hidden_states]), 'rows': rows}
