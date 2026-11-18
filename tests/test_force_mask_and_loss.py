from types import SimpleNamespace

import torch

from fastwam.models.wan22.fastwam import FastWAM, compute_masked_force_flow_loss
from fastwam.models.wan22.force_dit import ForceBlock
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import precompute_freqs_cis


def _tiny_force_block() -> ForceBlock:
    torch.manual_seed(1)
    return ForceBlock(
        hidden_dim=16,
        ffn_dim=32,
        num_heads=2,
        attn_head_dim=8,
        bridge_num_heads=2,
        bridge_head_dim=8,
        eps=1e-6,
    )


def test_invalid_force_keys_do_not_change_valid_local_attention_outputs():
    block = _tiny_force_block().eval()
    timesteps, fingers = 4, 2
    valid = torch.tensor([[[1, 1], [1, 0], [1, 0], [1, 1]]], dtype=torch.bool)
    x = torch.randn(1, timesteps, fingers, 16)
    changed = x.clone()
    changed[~valid] = float("nan")
    t_mod = torch.zeros(1, timesteps, fingers, 9, 16)
    temporal_freqs = precompute_freqs_cis(8, end=timesteps)
    spatial_freqs = precompute_freqs_cis(8, end=fingers)
    temporal_mask = torch.ones(timesteps, timesteps, dtype=torch.bool)

    first = block.local_attention(
        x, t_mod, temporal_freqs, spatial_freqs, temporal_mask, valid
    )
    second = block.local_attention(
        changed, t_mod, temporal_freqs, spatial_freqs, temporal_mask, valid
    )
    torch.testing.assert_close(first[valid], second[valid])
    assert torch.count_nonzero(first[~valid]) == 0
    assert torch.count_nonzero(second[~valid]) == 0


def test_all_invalid_force_rows_are_zero_and_finite():
    block = _tiny_force_block().eval()
    timesteps, fingers = 3, 2
    valid = torch.zeros(2, timesteps, fingers, dtype=torch.bool)
    x = torch.randn(2, timesteps, fingers, 16, requires_grad=True)
    t_mod = torch.zeros(2, timesteps, fingers, 9, 16)
    output = block.local_attention(
        x,
        t_mod,
        precompute_freqs_cis(8, end=timesteps),
        precompute_freqs_cis(8, end=fingers),
        torch.ones(timesteps, timesteps, dtype=torch.bool),
        valid,
    )
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) == 0
    output.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_force_mot_mask_removes_invalid_force_keys_and_has_no_empty_rows():
    dummy = SimpleNamespace(
        force_expert=SimpleNamespace(observed_steps=2, total_steps=3, num_fingers=2),
        video_expert=SimpleNamespace(
            build_video_to_video_mask=lambda **kwargs: torch.ones(
                kwargs["video_seq_len"], kwargs["video_seq_len"], dtype=torch.bool
            )
        ),
    )
    valid = torch.tensor([[[1, 0], [0, 0], [1, 0]]], dtype=torch.bool)
    mask = FastWAM._build_force_mot_attention_mask(
        dummy,
        video_seq_len=2,
        action_seq_len=2,
        video_tokens_per_frame=1,
        device=torch.device("cpu"),
        force_valid=valid,
    )
    force_columns = mask[0, 0, :, 2:8]
    assert torch.equal(force_columns.any(dim=0), valid.flatten())
    assert mask.any(dim=-1).all()


def test_force_flow_loss_is_per_sample_scheduler_weighted():
    pred = torch.tensor([[[1.0], [3.0]], [[2.0], [100.0]]], requires_grad=True)
    target = torch.zeros_like(pred)
    valid = torch.tensor([[[1], [1]], [[1], [0]]], dtype=torch.bool)
    raw, weighted, fraction, valid_samples = compute_masked_force_flow_loss(
        pred, target, valid, torch.tensor([2.0, 4.0])
    )
    # Per-sample MSEs are 5 and 4, not a global element-weighted reduction.
    torch.testing.assert_close(raw, torch.tensor(4.5))
    torch.testing.assert_close(weighted, torch.tensor(13.0))
    torch.testing.assert_close(fraction, torch.tensor(0.75))
    torch.testing.assert_close(valid_samples, torch.tensor(2.0))
    weighted.backward()
    assert pred.grad[1, 1].item() == 0.0


def test_all_invalid_force_loss_is_differentiable_zero():
    pred = torch.randn(2, 3, 2, requires_grad=True)
    raw, weighted, fraction, valid_samples = compute_masked_force_flow_loss(
        pred,
        torch.zeros_like(pred),
        torch.zeros_like(pred, dtype=torch.bool),
        torch.ones(2),
    )
    assert raw.item() == 0.0
    assert weighted.item() == 0.0
    assert fraction.item() == 0.0
    assert valid_samples.item() == 0.0
    weighted.backward()
    assert torch.isfinite(pred.grad).all()


def test_zero_force_gate_exactly_preserves_va_and_gate_gets_gradient():
    mixed_va = torch.randn(2, 5, 8)
    mixed_vfa = torch.randn(2, 8, 8)
    gate = torch.tensor(0.0, requires_grad=True)
    video, action = MoT._mix_va_with_force_gate(
        mixed_va,
        mixed_vfa,
        video_len=3,
        force_len=3,
        gate_parameter=gate,
    )
    assert torch.equal(video, mixed_va[:, :3])
    assert torch.equal(action, mixed_va[:, 3:])
    (video.square().mean() + action.square().mean()).backward()
    assert gate.grad is not None
    assert torch.isfinite(gate.grad)
