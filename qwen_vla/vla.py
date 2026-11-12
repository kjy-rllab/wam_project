"""A vision-language-action model: a Qwen VLM read at one layer by a flow-matching action expert.

The VLM encodes the camera images and the instruction; its residual stream after layer `tap` (the layers above are
dropped) is the context. The action expert is a transformer over one state token and the noisy action chunk that
cross-attends to the context and predicts the flow from actions to noise (as tmrl's FlowPolicy). Embodiments differ in
state and action size, so each has its own input and output maps; a batch pads them to the largest.

Three ways to train (mode):
    frozen   only the action expert
    full     the VLM too, kept near its pretrained weights (small learning rate and a pull back, see train.py)
    adapter  VLM frozen; a small transformer adds a correction to the VLM's output: context + adapter(context)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EmbodimentLinear(nn.Module):
    """One linear map per embodiment, applied to padded vectors."""

    def __init__(self, embodiments: int, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(embodiments, out_dim, in_dim) * in_dim**-0.5)
        self.bias = nn.Parameter(torch.zeros(embodiments, out_dim))

    def forward(self, x: Tensor, embodiment: Tensor) -> Tensor:
        return torch.einsum('bti,boi->bto', x, self.weight[embodiment]) + self.bias[embodiment][:, None]


class ResidualAdapter(nn.Module):
    """context + up(transformer(down(context))): bidirectional over the tokens, zero at initialization."""

    def __init__(self, dim: int, width: int = 1024, num_layers: int = 4, num_heads: int = 16) -> None:
        super().__init__()
        self.norm, self.down, self.up = nn.LayerNorm(dim), nn.Linear(dim, width), nn.Linear(width, dim)
        layer = nn.TransformerEncoderLayer(width, num_heads, 4 * width, dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, context: Tensor, mask: Tensor) -> Tensor:
        return context + self.up(self.encoder(self.down(self.norm(context)), src_key_padding_mask=~mask))


class ActionExpert(nn.Module):
    def __init__(
        self,
        context_dim: int,
        state_dim: int,
        action_dim: int,
        embodiments: int,
        action_len: int = 16,
        embed_dim: int = 1024,
        num_layers: int = 16,
        num_heads: int = 16,
        num_train_steps: int = 1000,
        num_inference_steps: int = 10,
        timeshift: float = 1.0,
    ) -> None:
        super().__init__()
        self.action_shape, self.embed_dim, self.num_train_steps = (action_len, action_dim), embed_dim, num_train_steps
        timesteps = torch.linspace(1, 0, num_inference_steps + 1)
        self.register_buffer('timesteps', (timeshift * timesteps) / (1 + (timeshift - 1) * timesteps))

        self.context_proj = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, embed_dim))
        self.state_in = EmbodimentLinear(embodiments, max(state_dim, 1), embed_dim)
        self.action_in = EmbodimentLinear(embodiments, action_dim, embed_dim)
        self.action_time = nn.Sequential(nn.Linear(2 * embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.position = nn.Parameter(torch.randn(1 + action_len, embed_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(embed_dim, num_heads, 4 * embed_dim, dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
        self.blocks = nn.TransformerDecoder(layer, num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.action_out = EmbodimentLinear(embodiments, embed_dim, action_dim)

    def forward(self, action: Tensor, t_action: Tensor, state: Tensor, embodiment: Tensor, context: Tensor, context_mask: Tensor) -> Tensor:
        """Flow of the noisy action chunk given the VLM's tokens."""
        context = self.context_proj(context.to(self.position.dtype))
        half = self.embed_dim // 2
        angles = t_action[:, None].float() * torch.exp(-math.log(10000) * torch.arange(half, device=action.device) / half)
        time = torch.cat([angles.sin(), angles.cos()], dim=-1)[:, None].expand(-1, action.shape[1], -1).to(action.dtype)
        tokens = self.action_time(torch.cat([self.action_in(action, embodiment), time], dim=-1))
        tokens = torch.cat([self.state_in(state[:, None], embodiment), tokens], dim=1) + self.position
        tokens = self.blocks(tokens, context, memory_key_padding_mask=~context_mask)
        return self.action_out(self.norm(tokens[:, 1:]), embodiment)


class VLA(nn.Module):
    def __init__(self, vlm: nn.Module, tap: int, state_dim: int, action_dim: int, embodiments: int, mode: str = 'frozen', **expert: object) -> None:
        super().__init__()
        self.mode = mode
        self.vlm = vlm.model  # vision encoder and language model, without the output head
        self.vlm.language_model.layers = self.vlm.language_model.layers[:tap]
        self.vlm.language_model.norm = nn.Identity()  # the released norm belongs to the last layer
        self.vlm.requires_grad_(mode == 'full')
        dim = vlm.config.text_config.hidden_size
        self.adapter = ResidualAdapter(dim) if mode == 'adapter' else None
        self.expert = ActionExpert(dim, state_dim, action_dim, embodiments, **expert)

    def context(self, inputs: dict) -> tuple[Tensor, Tensor]:
        """The context tokens of a batch of prompts, and which of them are real."""
        mask = inputs['attention_mask'].bool()
        with torch.set_grad_enabled(self.mode == 'full' and torch.is_grad_enabled()):
            hidden = self.vlm(**inputs).last_hidden_state
        if self.adapter is not None:
            hidden = self.adapter(hidden.to(self.adapter.down.weight.dtype), mask)
        return hidden, mask

    @torch.no_grad()
    def sample(self, inputs: dict, state: Tensor, embodiment: Tensor, action_noise: Tensor | None = None) -> Tensor:
        context, mask = self.context(inputs)
        expert = self.expert
        action = torch.randn((len(state), *expert.action_shape), device=state.device) if action_noise is None else action_noise
        for tcont, tcont_next in zip(expert.timesteps[:-1], expert.timesteps[1:]):
            t_action = (tcont * expert.num_train_steps).expand(len(state))
            action = action + (tcont_next - tcont) * expert(action, t_action, state, embodiment, context, mask).float()
        return action

    def forward(self, inputs: dict, state: Tensor, action: Tensor, valid: Tensor, embodiment: Tensor) -> tuple[Tensor, dict[str, float]]:
        """valid [B, action_len, action_dim]: real steps and dimensions of each sample's action chunk."""
        context, mask = self.context(inputs)
        action_noise = torch.randn_like(action)
        tcont_action = torch.rand(len(action), device=action.device)
        direction_action = action_noise - action
        noisy_action = action + tcont_action.view(-1, 1, 1) * direction_action
        flow = self.expert(noisy_action, tcont_action * self.expert.num_train_steps, state, embodiment, context, mask)
        loss = (F.mse_loss(flow.float(), direction_action, reduction='none') * valid).sum() / valid.sum()
        return loss, {'loss': loss.item()}
