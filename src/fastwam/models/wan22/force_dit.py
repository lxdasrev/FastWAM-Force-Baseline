from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .wan_video_dit import (
    CrossAttention,
    RMSNorm,
    SelfAttention,
    modulate,
    precompute_freqs_cis,
    sinusoidal_embedding_1d,
)


class ForceEncoder(nn.Module):
    """Encode a scalar for every (time, finger) pair into force tokens."""

    def __init__(
        self,
        hidden_dim: int,
        max_timesteps: int,
        num_fingers: int,
        observed_steps: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_timesteps = int(max_timesteps)
        self.num_fingers = int(num_fingers)
        self.observed_steps = int(observed_steps)
        self.value_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, self.hidden_dim),
        )
        self.finger_embedding = nn.Embedding(self.num_fingers, self.hidden_dim)
        self.temporal_embedding = nn.Embedding(self.max_timesteps, self.hidden_dim)
        self.observed_future_embedding = nn.Embedding(2, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        force: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if force.ndim != 3:
            raise ValueError(f"`force` must be [B,T,F], got {tuple(force.shape)}")
        batch_size, timesteps, fingers = force.shape
        if fingers != self.num_fingers:
            raise ValueError(f"Expected {self.num_fingers} fingers, got {fingers}")
        if timesteps > self.max_timesteps:
            raise ValueError(f"Force sequence {timesteps} exceeds max_timesteps={self.max_timesteps}")

        device = force.device
        time_ids = torch.arange(timesteps, device=device)
        finger_ids = torch.arange(fingers, device=device)
        type_ids = (time_ids >= self.observed_steps).long()
        tokens = self.value_mlp(force.unsqueeze(-1))
        tokens = tokens + self.temporal_embedding(time_ids)[None, :, None, :]
        tokens = tokens + self.finger_embedding(finger_ids)[None, None, :, :]
        tokens = tokens + self.observed_future_embedding(type_ids)[None, :, None, :]
        tokens = self.norm(tokens)
        if valid_mask is not None:
            if valid_mask.shape != force.shape:
                raise ValueError(
                    f"`valid_mask` must match force shape {tuple(force.shape)}, got {tuple(valid_mask.shape)}"
                )
            tokens = tokens * valid_mask.to(device=device, dtype=tokens.dtype).unsqueeze(-1)
        return tokens


class ForceHead(nn.Module):
    def __init__(self, hidden_dim: int, eps: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.modulation = nn.Parameter(torch.zeros(1, 2, hidden_dim))
        self.proj = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, token_time: torch.Tensor) -> torch.Tensor:
        shift, scale = (self.modulation + token_time.unsqueeze(2)).chunk(2, dim=2)
        return self.proj(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))


class ForceBlock(nn.Module):
    """Temporal -> spatial -> optional bridge -> context cross-attn -> FFN."""

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_heads: int,
        attn_head_dim: int,
        bridge_num_heads: int,
        bridge_head_dim: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.bridge_num_heads = int(bridge_num_heads)
        self.bridge_head_dim = int(bridge_head_dim)
        self.bridge_inner_dim = self.bridge_num_heads * self.bridge_head_dim

        self.temporal_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.spatial_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = CrossAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.norm_temporal = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm_spatial = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm_bridge = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm_cross = nn.LayerNorm(hidden_dim, eps=eps)
        self.norm_ffn = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        # temporal shift/scale/gate, spatial shift/scale/gate, FFN shift/scale/gate
        self.modulation = nn.Parameter(torch.randn(1, 9, hidden_dim) / hidden_dim**0.5)

        self.bridge_q = nn.Linear(hidden_dim, self.bridge_inner_dim)
        self.bridge_k = nn.Linear(hidden_dim, self.bridge_inner_dim)
        self.bridge_v = nn.Linear(hidden_dim, self.bridge_inner_dim)
        self.bridge_o = nn.Linear(self.bridge_inner_dim, hidden_dim)
        self.bridge_norm_q = RMSNorm(self.bridge_inner_dim, eps=eps)
        self.bridge_norm_k = RMSNorm(self.bridge_inner_dim, eps=eps)
        # Force is newly initialized, so let it read the shared bridge from the
        # first step. Video/Action preservation is handled by MoT's zero VA gate.
        self.bridge_gate = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _modulations(t_mod: torch.Tensor, base: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if t_mod.ndim != 5:
            raise ValueError(f"Force t_mod must be [B,T,F,9,D], got {tuple(t_mod.shape)}")
        return tuple((base[None] + t_mod).unbind(dim=3))

    def local_attention(
        self,
        x: torch.Tensor,
        t_mod: torch.Tensor,
        temporal_freqs: torch.Tensor,
        spatial_freqs: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Force tokens must be [B,T,F,D], got {tuple(x.shape)}")
        batch_size, timesteps, fingers, hidden = x.shape
        mods = self._modulations(t_mod, self.modulation.to(dtype=x.dtype, device=x.device))
        t_shift, t_scale, t_gate, s_shift, s_scale, s_gate, _, _, _ = mods

        temporal_input = modulate(self.norm_temporal(x), t_shift, t_scale)
        temporal_input = temporal_input.permute(0, 2, 1, 3).reshape(batch_size * fingers, timesteps, hidden)
        temporal_out = self.temporal_attn(
            temporal_input,
            temporal_freqs[:timesteps].view(timesteps, 1, -1),
            self_attn_mask=temporal_mask[:timesteps, :timesteps],
        )
        temporal_out = temporal_out.reshape(batch_size, fingers, timesteps, hidden).permute(0, 2, 1, 3)
        x = x + t_gate * temporal_out

        spatial_input = modulate(self.norm_spatial(x), s_shift, s_scale)
        spatial_input = spatial_input.reshape(batch_size * timesteps, fingers, hidden)
        spatial_out = self.spatial_attn(
            spatial_input,
            spatial_freqs[:fingers].view(fingers, 1, -1),
        )
        spatial_out = spatial_out.reshape(batch_size, timesteps, fingers, hidden)
        return x + s_gate * spatial_out

    def prepare_bridge(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bridge_input = self.norm_bridge(x_flat)
        return (
            self.bridge_norm_q(self.bridge_q(bridge_input)),
            self.bridge_norm_k(self.bridge_k(bridge_input)),
            self.bridge_v(bridge_input),
        )

    def finish(
        self,
        x: torch.Tensor,
        t_mod: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        bridge_attention: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, timesteps, fingers, hidden = x.shape
        flat = x.reshape(batch_size, timesteps * fingers, hidden)
        if bridge_attention is not None:
            flat = flat + torch.tanh(self.bridge_gate) * self.bridge_o(bridge_attention)

        cross_mask = context_mask
        if cross_mask.ndim == 2:
            cross_mask = cross_mask[:, None, :].expand(-1, flat.shape[1], -1)
        if cross_mask.ndim == 3:
            cross_mask = cross_mask.unsqueeze(1)
        flat = flat + self.cross_attn(self.norm_cross(flat), context, ctx_mask=cross_mask)

        mods = self._modulations(t_mod, self.modulation.to(dtype=x.dtype, device=x.device))
        f_shift, f_scale, f_gate = mods[6:9]
        f_shift = f_shift.reshape(batch_size, timesteps * fingers, hidden)
        f_scale = f_scale.reshape(batch_size, timesteps * fingers, hidden)
        f_gate = f_gate.reshape(batch_size, timesteps * fingers, hidden)
        ffn_input = modulate(self.norm_ffn(flat), f_shift, f_scale)
        flat = flat + f_gate * self.ffn(ffn_input)
        return flat.reshape(batch_size, timesteps, fingers, hidden)


class ForceDiT(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 512,
        ffn_dim: int = 2048,
        num_layers: int = 10,
        num_heads: int = 8,
        attn_head_dim: int = 64,
        bridge_num_heads: int = 24,
        bridge_head_dim: int = 128,
        text_dim: int = 4096,
        freq_dim: int = 256,
        eps: float = 1e-6,
        history_steps: int = 16,
        future_steps: int = 32,
        num_fingers: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.bridge_num_heads = int(bridge_num_heads)
        self.bridge_head_dim = int(bridge_head_dim)
        self.freq_dim = int(freq_dim)
        self.history_steps = int(history_steps)
        self.observed_steps = self.history_steps + 1
        self.future_steps = int(future_steps)
        self.total_steps = self.observed_steps + self.future_steps
        self.num_fingers = int(num_fingers)

        self.encoder = ForceEncoder(
            hidden_dim=self.hidden_dim,
            max_timesteps=self.total_steps,
            num_fingers=self.num_fingers,
            observed_steps=self.observed_steps,
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim * 9))
        self.blocks = nn.ModuleList(
            [
                ForceBlock(
                    hidden_dim=self.hidden_dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_heads,
                    attn_head_dim=attn_head_dim,
                    bridge_num_heads=bridge_num_heads,
                    bridge_head_dim=bridge_head_dim,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = ForceHead(self.hidden_dim, eps=eps)
        self.temporal_freqs = precompute_freqs_cis(attn_head_dim, end=self.total_steps)
        self.spatial_freqs = precompute_freqs_cis(attn_head_dim, end=self.num_fingers)

    def _apply(self, fn):
        result = super()._apply(fn)
        device = next(self.parameters()).device
        self.temporal_freqs = self.temporal_freqs.to(device=device)
        self.spatial_freqs = self.spatial_freqs.to(device=device)
        return result

    def build_temporal_mask(self, device: torch.device) -> torch.Tensor:
        mask = torch.ones((self.total_steps, self.total_steps), dtype=torch.bool, device=device)
        mask[: self.observed_steps, self.observed_steps :] = False
        return mask

    def prepare(
        self,
        force: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if force.shape[1:] != (self.total_steps, self.num_fingers):
            raise ValueError(
                f"Force must be [B,{self.total_steps},{self.num_fingers}], got {tuple(force.shape)}"
            )
        batch_size = force.shape[0]
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        future_time = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        clean_time = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, torch.zeros_like(timestep))
        )
        token_time = future_time[:, None, None, :].expand(-1, self.total_steps, self.num_fingers, -1).clone()
        token_time[:, : self.observed_steps] = clean_time[:, None, None, :]
        t_mod = self.time_projection(token_time).unflatten(-1, (9, self.hidden_dim))
        tokens = self.encoder(force, valid_mask=valid_mask)
        return {
            "tokens": tokens,
            "t_mod": t_mod,
            "token_time": token_time.reshape(batch_size, self.total_steps * self.num_fingers, self.hidden_dim),
            "context": self.text_embedding(context),
            "context_mask": context_mask,
            "temporal_mask": self.build_temporal_mask(force.device),
        }

    def run_unbridged_block(self, block_idx: int, x: torch.Tensor, state: dict[str, torch.Tensor]) -> torch.Tensor:
        block = self.blocks[block_idx]
        x = block.local_attention(
            x,
            state["t_mod"],
            self.temporal_freqs,
            self.spatial_freqs,
            state["temporal_mask"],
        )
        return block.finish(x, state["t_mod"], state["context"], state["context_mask"])

    def post(self, x: torch.Tensor, token_time: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, fingers, hidden = x.shape
        flat = x.reshape(batch_size, timesteps * fingers, hidden)
        return self.head(flat, token_time).reshape(batch_size, timesteps, fingers)
