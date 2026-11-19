from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.utils.logging_config import get_logger
from fastwam.datasets.opentouch.contract import verify_artifact_hashes

from .action_dit import ActionDiT
from .force_dit import ForceDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


def compute_masked_force_flow_loss(
    pred_force: torch.Tensor,
    target_force: torch.Tensor,
    valid_force: torch.Tensor,
    training_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw/weighted per-sample Force MSE and validity diagnostics."""
    if pred_force.shape != target_force.shape or pred_force.shape != valid_force.shape:
        raise ValueError(
            "pred_force, target_force and valid_force must have identical [B,T,F] shapes, got "
            f"{tuple(pred_force.shape)}, {tuple(target_force.shape)}, {tuple(valid_force.shape)}"
        )
    element_loss = F.mse_loss(
        pred_force.float(), target_force.float(), reduction="none"
    )
    valid_float = valid_force.to(device=element_loss.device, dtype=element_loss.dtype)
    valid_elements_per_sample = valid_float.sum(dim=(1, 2))
    sample_has_force = valid_elements_per_sample > 0
    sample_valid_float = sample_has_force.to(dtype=element_loss.dtype)
    sample_mse = (
        (element_loss * valid_float).sum(dim=(1, 2))
        / valid_elements_per_sample.clamp(min=1.0)
    )
    valid_sample_count = sample_valid_float.sum()
    denominator = valid_sample_count.clamp(min=1.0)
    raw_loss = (sample_mse * sample_valid_float).sum() / denominator
    weight = training_weight.to(device=sample_mse.device, dtype=sample_mse.dtype)
    if weight.ndim == 0:
        weight = weight.expand_as(sample_mse)
    if weight.shape != sample_mse.shape:
        raise ValueError(
            f"training_weight must be scalar or [B], got {tuple(weight.shape)} for B={sample_mse.shape[0]}"
        )
    weighted_loss = (sample_mse * weight * sample_valid_float).sum() / denominator
    differentiable_zero = pred_force.float().sum() * 0.0
    has_any = (valid_sample_count > 0).to(dtype=raw_loss.dtype)
    raw_loss = raw_loss * has_any + differentiable_zero
    weighted_loss = weighted_loss * has_any + differentiable_zero
    return raw_loss, weighted_loss, valid_float.mean(), valid_sample_count


class FastWAM(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        force_expert: Optional[ForceDiT] = None,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        force_train_shift: float = 1.0,
        force_infer_shift: float = 1.0,
        force_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        loss_lambda_force: float = 0.25,
        compile_training_denoise: bool = False,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.force_expert = force_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        if self.force_expert is not None:
            self.train_force_scheduler = WanContinuousFlowMatchScheduler(
                num_train_timesteps=force_num_train_timesteps,
                shift=force_train_shift,
            )
            self.infer_force_scheduler = WanContinuousFlowMatchScheduler(
                num_train_timesteps=force_num_train_timesteps,
                shift=force_infer_shift,
            )
        else:
            self.train_force_scheduler = None
            self.infer_force_scheduler = None
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.loss_lambda_force = float(loss_lambda_force)
        self.experiment_mode = "force" if self.force_expert is not None else "baseline"
        self.initialization_mode = "wan_actiondit"
        self.initialization_checkpoint = None
        self.initialization_source_step = None
        self.proprio_initialization = "random_canonical"
        self.compile_training_denoise = bool(compile_training_denoise)
        self.mot.compile_training_layers = self.compile_training_denoise

        self.to(self.device)

    @staticmethod
    def _checkpoint_experiment_mode(payload: Mapping[str, Any]) -> Optional[str]:
        mode = payload.get("experiment_mode")
        if mode in {"force", "baseline"}:
            return str(mode)
        if "force_meta" in payload:
            return "force"
        mot_state = payload.get("mot")
        if isinstance(mot_state, Mapping):
            if any(
                str(key).startswith(("force_expert.", "force_va_gates."))
                for key in mot_state
            ):
                return "force"
            return "baseline"
        return None

    def set_initialization_metadata(
        self,
        *,
        mode: str,
        checkpoint: Optional[str] = None,
        source_step: Optional[int] = None,
    ) -> None:
        if mode not in {"wan_actiondit", "official_fastwam", "unknown_legacy"}:
            raise ValueError(f"Unsupported initialization mode: {mode}")
        self.initialization_mode = mode
        self.initialization_checkpoint = checkpoint
        self.initialization_source_step = source_step

    def initialize_from_official_fastwam(self, path: str | Path) -> Mapping[str, Any]:
        """Strictly initialize only the standard Video/Action MoT experts.

        OpenTouch proprio and all force-specific parameters deliberately retain
        their fresh initialization because their input semantics/architecture are
        not part of the official two-expert checkpoint contract.
        """
        payload = torch.load(path, map_location="cpu")
        if "mot" not in payload:
            if "dit" in payload:
                raise ValueError(
                    "Official FastWAM initialization requires a standard `mot` checkpoint; "
                    "legacy/pure-video `dit` checkpoints are not supported."
                )
            raise ValueError(f"Official FastWAM checkpoint has no `mot` state: {path}")
        if payload.get("experiment_mode") == "force" or "force_meta" in payload:
            raise ValueError("A force experiment checkpoint cannot be used as official FastWAM initialization.")

        source_state = payload["mot"]
        if not isinstance(source_state, Mapping):
            raise TypeError("Official checkpoint `mot` must be a state-dict mapping.")
        force_prefixes = ("force_expert.", "force_va_gates.")
        source_force_keys = [
            key for key in source_state if str(key).startswith(force_prefixes)
        ]
        if source_force_keys:
            raise ValueError(
                "Official initialization checkpoint contains force parameters: "
                f"{source_force_keys[:5]}"
            )

        target_state = self.mot.state_dict()
        expected_keys = {
            key for key in target_state if not str(key).startswith(force_prefixes)
        }
        source_keys = set(source_state)
        missing = sorted(expected_keys - source_keys)
        unexpected = sorted(source_keys - expected_keys)
        shape_mismatch = sorted(
            key
            for key in expected_keys & source_keys
            if tuple(target_state[key].shape) != tuple(source_state[key].shape)
        )
        if missing or unexpected or shape_mismatch:
            raise ValueError(
                "Official FastWAM checkpoint is not the standard compatible Video+Action MoT: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}, "
                f"shape_mismatch={shape_mismatch[:10]}"
            )

        merged_state = dict(target_state)
        merged_state.update(source_state)
        self.mot.load_state_dict(merged_state, strict=True)
        self.set_initialization_metadata(
            mode="official_fastwam",
            checkpoint=str(Path(path).resolve()),
            source_step=(None if payload.get("step") is None else int(payload["step"])),
        )
        logger.info(
            "Initialized %d Video/Action MoT tensors from official FastWAM checkpoint %s; "
            "OpenTouch proprio and force tensors remain freshly initialized.",
            len(source_state),
            path,
        )
        return payload

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        force_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = False,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        force_train_shift: float = 1.0,
        force_infer_shift: float = 1.0,
        force_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        loss_lambda_force: float = 0.25,
        compile_training_denoise: bool = False,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAM.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        force_expert = None
        force_bridge_layers = None
        if force_dit_config is not None:
            force_cfg = dict(force_dit_config)
            force_bridge_layers = force_cfg.pop(
                "bridge_layers",
                {8: 1, 10: 2, 13: 3, 15: 4, 17: 5, 20: 6, 22: 7},
            )
            force_cfg.setdefault("bridge_num_heads", int(video_expert.num_heads))
            force_cfg.setdefault("bridge_head_dim", int(video_expert.attn_head_dim))
            force_cfg.setdefault("text_dim", int(video_dit_config["text_dim"]))
            force_expert = ForceDiT(**force_cfg).to(device=device, dtype=torch_dtype)

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            force_expert=force_expert,
            force_bridge_layers=force_bridge_layers,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            force_expert=force_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            force_train_shift=force_train_shift,
            force_infer_shift=force_infer_shift,
            force_num_train_timesteps=force_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            loss_lambda_force=loss_lambda_force,
            compile_training_denoise=compile_training_denoise,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if tiled:
            raise NotImplementedError("Batched VAE encoding does not support tiled encoding.")
        if not hasattr(self, "_vae_encode_compiled"):
            self._vae_encode_compiled = torch.compile(
                self.vae.model.encode,
                backend="cudagraphs",
                fullgraph=True,
            )
        return self._vae_encode_compiled(
            video_tensor.to(self.device),
            self.vae.scale,
        ).clone()

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        if tiled:
            raise NotImplementedError("Batched VAE image encoding does not support tiled encoding.")
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        return self.vae.model.encode(image.unsqueeze(0), self.vae.scale)

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)
        context = sample.get("context")
        context_mask = sample.get("context_mask")
        if context is None and context_mask is None:
            prompt = sample.get("prompt")
            if prompt is None:
                raise ValueError("FastWAM training requires `context/context_mask` or `prompt`.")
            context, context_mask = self.encode_prompt(prompt)
        elif context is None or context_mask is None:
            raise ValueError("`context` and `context_mask` must both exist when either is provided.")

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        force = None
        force_valid = None
        force_is_pad = None
        if self.force_expert is not None:
            force = sample.get("force")
            if force is None:
                raise ValueError("`sample['force']` is required when ForceExpert is enabled.")
            expected_force_shape = (
                batch_size,
                self.force_expert.total_steps,
                self.force_expert.num_fingers,
            )
            if tuple(force.shape) != expected_force_shape:
                raise ValueError(
                    f"`sample['force']` must have shape {expected_force_shape}, got {tuple(force.shape)}"
                )
            force = force.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            force_valid = sample.get("force_valid")
            if force_valid is None:
                force_valid = torch.ones_like(force, dtype=torch.bool)
            elif tuple(force_valid.shape) != expected_force_shape:
                raise ValueError(
                    f"`sample['force_valid']` must have shape {expected_force_shape}, got {tuple(force_valid.shape)}"
                )
            force_valid = force_valid.to(device=self.device, dtype=torch.bool, non_blocking=True)
            force_is_pad = sample.get("force_is_pad")
            if force_is_pad is None:
                force_is_pad = torch.zeros(
                    (batch_size, self.force_expert.total_steps),
                    device=self.device,
                    dtype=torch.bool,
                )
            elif tuple(force_is_pad.shape) != (batch_size, self.force_expert.total_steps):
                raise ValueError(
                    "`sample['force_is_pad']` must be [B,T], got "
                    f"{tuple(force_is_pad.shape)}"
                )
            force_is_pad = force_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
            # Keep one authoritative validity mask throughout ForceEncoder,
            # local attention, MoT and the force objective.
            force_valid = (
                force_valid
                & (~force_is_pad[:, :, None])
                & torch.isfinite(force)
            )
            force = torch.where(force_valid, force, torch.zeros_like(force))

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
            "force": force,
            "force_valid": force_valid,
            "force_is_pad": force_is_pad,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> first-frame video only
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    @torch.no_grad()
    def _build_force_mot_attention_mask(
        self,
        *,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        force_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.force_expert is None:
            raise RuntimeError("Force attention mask requested without ForceExpert.")
        observed_force_tokens = self.force_expert.observed_steps * self.force_expert.num_fingers
        total_force_tokens = self.force_expert.total_steps * self.force_expert.num_fingers
        force_start = video_seq_len
        force_future_start = force_start + observed_force_tokens
        action_start = force_start + total_force_tokens
        total = action_start + action_seq_len
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)

        # V0/V+ keep the original FastWAM video mask and never read force/action.
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)

        # F0 -> V0,F0.
        mask[force_start:force_future_start, :first_frame_tokens] = True
        mask[force_start:force_future_start, force_start:force_future_start] = True
        # F+ -> V0,F0,F+,A.
        mask[force_future_start:action_start, :first_frame_tokens] = True
        mask[force_future_start:action_start, force_start:total] = True
        # A -> V0,F0,F+,A.
        mask[action_start:, :first_frame_tokens] = True
        mask[action_start:, force_start:total] = True
        if force_valid is None:
            return mask
        expected = (
            force_valid.shape[0],
            self.force_expert.total_steps,
            self.force_expert.num_fingers,
        )
        if tuple(force_valid.shape) != expected:
            raise ValueError(
                "force_valid must be [B,T,F] for the Force MoT mask, got "
                f"{tuple(force_valid.shape)}"
            )
        batch_size = force_valid.shape[0]
        batched_mask = mask[None, None].expand(batch_size, 1, total, total).clone()
        flat_valid = force_valid.to(device=device, dtype=torch.bool).flatten(1)
        batched_mask[..., force_start:action_start] &= flat_valid[:, None, None, :]

        # Every Video and Action query already has a valid Video/Action key.
        # Invalid Force queries are cleared after the ForceBlock, but keep a
        # safe placeholder here for kernels that reject fully masked rows.
        empty_rows = ~batched_mask.any(dim=-1, keepdim=True)
        batched_mask[..., :1] |= empty_rows
        return batched_mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def _joint_denoise_core(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        action_condition: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the tensor-only video/action core shared by training and inference."""
        (
            video_tokens,
            t_video,
            t_mod_video,
            context_video,
            context_mask_video,
            freqs_video,
            f_video,
            h_video,
            w_video,
            _tokens_per_frame,
        ) = self.video_expert.prepare(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action_condition,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        (
            action_tokens,
            _t_action,
            t_mod_action,
            context_action,
            context_mask_action,
            freqs_action,
        ) = self.action_expert.prepare(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        video_tokens, action_tokens = self.mot.forward_joint_core(
            video_tokens=video_tokens,
            action_tokens=action_tokens,
            video_freqs=freqs_video,
            action_freqs=freqs_action,
            video_t_mod=t_mod_video,
            action_t_mod=t_mod_action,
            video_context=context_video,
            video_context_mask=context_mask_video,
            action_context=context_action,
            action_context_mask=context_mask_action,
            attention_mask=attention_mask,
        )
        return (
            self.video_expert.post(video_tokens, t_video, f_video, h_video, w_video),
            self.action_expert.post(action_tokens),
        )

    def _joint_force_denoise_core(
        self,
        *,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        latents_force: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        timestep_force: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        va_attention_mask: torch.Tensor,
        vfa_attention_mask: torch.Tensor,
        force_valid: Optional[torch.Tensor],
        fuse_vae_embedding_in_latents: bool,
        action_condition: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.force_expert is None:
            raise RuntimeError("Force denoise core requested without ForceExpert.")
        (
            video_tokens,
            t_video,
            t_mod_video,
            context_video,
            context_mask_video,
            freqs_video,
            f_video,
            h_video,
            w_video,
            _tokens_per_frame,
        ) = self.video_expert.prepare(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action_condition,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        (
            action_tokens,
            _t_action,
            t_mod_action,
            context_action,
            context_mask_action,
            freqs_action,
        ) = self.action_expert.prepare(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        force_state = self.force_expert.prepare(
            force=latents_force,
            timestep=timestep_force,
            context=context,
            context_mask=context_mask,
            valid_mask=force_valid,
        )
        video_tokens, action_tokens, force_tokens = self.mot.forward_joint_force_core(
            video_tokens=video_tokens,
            action_tokens=action_tokens,
            force_tokens=force_state["tokens"],
            video_freqs=freqs_video,
            action_freqs=freqs_action,
            video_t_mod=t_mod_video,
            action_t_mod=t_mod_action,
            force_state=force_state,
            video_context=context_video,
            video_context_mask=context_mask_video,
            action_context=context_action,
            action_context_mask=context_mask_action,
            va_attention_mask=va_attention_mask,
            vfa_attention_mask=vfa_attention_mask,
        )
        return (
            self.video_expert.post(video_tokens, t_video, f_video, h_video, w_video),
            self.action_expert.post(action_tokens),
            self.force_expert.post(
                force_tokens,
                force_state["token_time"],
                valid_mask=force_state["valid_mask"],
            ),
        )

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        force = inputs["force"]
        force_valid = inputs["force_valid"]
        force_is_pad = inputs["force_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        noisy_force = None
        target_force_future = None
        timestep_force = None
        if self.force_expert is not None:
            future = force[:, self.force_expert.observed_steps :]
            noise_force = torch.randn_like(future)
            timestep_force = self.train_force_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=force.dtype,
            )
            noisy_future = self.train_force_scheduler.add_noise(future, noise_force, timestep_force)
            target_force_future = self.train_force_scheduler.training_target(
                future, noise_force, timestep_force
            )
            noisy_force = force.clone()
            noisy_force[:, self.force_expert.observed_steps :] = noisy_future

        patch_t, patch_h, patch_w = (int(size) for size in self.video_expert.patch_size)
        latent_t, latent_h, latent_w = latents.shape[-3:]
        tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=(latent_t // patch_t) * tokens_per_frame,
            action_seq_len=noisy_action.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=latents.device,
        )
        pred_force = None
        if self.force_expert is None:
            pred_video, pred_action = self._joint_denoise_core(
                latents_video=latents,
                latents_action=noisy_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                attention_mask=attention_mask,
                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
                action_condition=action,
            )
        else:
            force_attention_mask = self._build_force_mot_attention_mask(
                video_seq_len=(latent_t // patch_t) * tokens_per_frame,
                action_seq_len=noisy_action.shape[1],
                video_tokens_per_frame=tokens_per_frame,
                device=latents.device,
                force_valid=force_valid,
            )
            pred_video, pred_action, pred_force = self._joint_force_denoise_core(
                latents_video=latents,
                latents_action=noisy_action,
                latents_force=noisy_force,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                timestep_force=timestep_force,
                context=context,
                context_mask=context_mask,
                va_attention_mask=attention_mask,
                vfa_attention_mask=force_attention_mask,
                force_valid=force_valid,
                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
                action_condition=action,
            )

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2) # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_force = None
        raw_force_loss = None
        scheduler_weighted_force_loss = None
        force_valid_fraction = None
        force_valid_samples = None
        if self.force_expert is not None:
            pred_force_future = pred_force[:, self.force_expert.observed_steps :]
            valid_force = force_valid[:, self.force_expert.observed_steps :]
            force_weight = self.train_force_scheduler.training_weight(timestep_force).to(
                pred_force_future.device, dtype=torch.float32
            )
            (
                raw_force_loss,
                scheduler_weighted_force_loss,
                force_valid_fraction,
                force_valid_samples,
            ) = compute_masked_force_flow_loss(
                pred_force_future,
                target_force_future,
                valid_force,
                force_weight,
            )
            loss_force = self.loss_lambda_force * scheduler_weighted_force_loss

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        if loss_force is not None:
            loss_total = loss_total + loss_force
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        if loss_force is not None:
            loss_dict.update(
                {
                    "raw_force_loss": float(raw_force_loss.detach().item()),
                    "scheduler_weighted_force_loss": float(
                        scheduler_weighted_force_loss.detach().item()
                    ),
                    "loss_force": float(loss_force.detach().item()),
                    "force_valid_fraction": float(force_valid_fraction.detach().item()),
                    "force_valid_samples": float(force_valid_samples.detach().item()),
                }
            )
            for layer, parameter in self.mot.force_va_gates.items():
                loss_dict[f"force_gate_raw_{layer}"] = float(parameter.detach().float().item())
                loss_dict[f"force_gate_tanh_{layer}"] = float(
                    torch.tanh(parameter.detach().float()).item()
                )
            for layer, stats in self.mot._last_force_bridge_stats.items():
                loss_dict[f"force_bridge_video_delta_rms_{layer}"] = float(stats[1].item())
                loss_dict[f"force_bridge_action_delta_rms_{layer}"] = float(stats[2].item())
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patch_t, patch_h, patch_w = (int(size) for size in self.video_expert.patch_size)
        latent_t, latent_h, latent_w = latents_video.shape[-3:]
        tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=(latent_t // patch_t) * tokens_per_frame,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=latents_video.device,
        )
        return self._joint_denoise_core(
            latents_video=latents_video,
            latents_action=latents_action,
            timestep_video=timestep_video,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            attention_mask=attention_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            action_condition=gt_action,
        )

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    def _denoise_action_with_video_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_cache_k: list[torch.Tensor],
        video_cache_v: list[torch.Tensor],
        action_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        (
            action_tokens,
            _t,
            action_t_mod,
            action_context,
            action_context_mask,
            action_freqs,
        ) = self.action_expert.prepare(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache_tensor(
            action_tokens=action_tokens,
            action_freqs=action_freqs,
            action_t_mod=action_t_mod,
            action_context=action_context,
            action_context_mask=action_context_mask,
            video_cache_k=video_cache_k,
            video_cache_v=video_cache_v,
            action_attention_mask=action_attention_mask,
        )
        return self.action_expert.post(action_tokens)

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        """Legacy dictionary-cache path retained for the optional IDM variant."""
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
        compile_action_infer: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
                compile_action_infer=compile_action_infer,
            )["action"]
        
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        patch_t, patch_h, patch_w = (int(size) for size in self.video_expert.patch_size)
        tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)
        joint_attention_mask = self._build_mot_attention_mask(
            video_seq_len=(latent_t // patch_t) * tokens_per_frame,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=self.device,
        )
        if compile_action_infer:
            if action is not None:
                raise ValueError(
                    "`compile_action_infer=True` does not support action conditioning in `infer_joint`."
                )
            if not hasattr(self, "_joint_denoise_core_compiled_inference"):
                self._joint_denoise_core_compiled_inference = torch.compile(
                    self._joint_denoise_core,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
            joint_denoise_core = self._joint_denoise_core_compiled_inference
        else:
            joint_denoise_core = self._joint_denoise_core

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            if compile_action_infer:
                torch.compiler.cudagraph_mark_step_begin()
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = joint_denoise_core(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                attention_mask=joint_attention_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                action_condition=action,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        compile_action_infer: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        (
            video_tokens,
            _t_video,
            video_t_mod,
            video_context,
            video_context_mask,
            video_freqs,
            _f_video,
            _h_video,
            _w_video,
            tokens_per_frame,
        ) = self.video_expert.prepare(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_tokens.shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=video_tokens.device,
        )
        video_attention_mask = attention_mask[:video_seq_len, :video_seq_len]
        action_attention_mask = attention_mask[video_seq_len:, :]
        if compile_action_infer:
            if not hasattr(self, "_prefill_video_cache_compiled"):
                self._prefill_video_cache_compiled = torch.compile(
                    self.mot.prefill_video_cache_tensor,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
            if not hasattr(self, "_denoise_action_with_video_cache_compiled"):
                self._denoise_action_with_video_cache_compiled = torch.compile(
                    self._denoise_action_with_video_cache,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
            prefill_video_cache = self._prefill_video_cache_compiled
            denoise_action_with_video_cache = self._denoise_action_with_video_cache_compiled
        else:
            prefill_video_cache = self.mot.prefill_video_cache_tensor
            denoise_action_with_video_cache = self._denoise_action_with_video_cache
        if compile_action_infer:
            torch.compiler.cudagraph_mark_step_begin()
        video_cache_k, video_cache_v = prefill_video_cache(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context=video_context,
            video_context_mask=video_context_mask,
            video_attention_mask=video_attention_mask,
        )
        if compile_action_infer:
            # Inductor reduce-overhead may return graph-owned buffers that are overwritten on replay.
            video_cache_k = [cache.clone() for cache in video_cache_k]
            video_cache_v = [cache.clone() for cache in video_cache_v]

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            if compile_action_infer:
                torch.compiler.cudagraph_mark_step_begin()
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = denoise_action_with_video_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_cache_k=video_cache_k,
                video_cache_v=video_cache_v,
                action_attention_mask=action_attention_mask,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def infer_action_force(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        force_history: torch.Tensor,
        force_history_valid: Optional[torch.Tensor] = None,
        force_history_is_pad: Optional[torch.Tensor] = None,
        action_horizon: int = 32,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 10,
        action_sigma_shift: Optional[float] = None,
        force_sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        """Jointly generate a 32-step action chunk and future two-finger pressure."""
        self.eval()
        if self.force_expert is None:
            raise RuntimeError("`infer_action_force` requires a configured ForceExpert.")
        if action_horizon != self.force_expert.future_steps:
            raise ValueError(
                f"Action horizon must equal force future horizon {self.force_expert.future_steps}, got {action_horizon}"
            )
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[:2] != (1, 3):
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        if input_image.shape[-2] % 16 or input_image.shape[-1] % 16:
            raise ValueError("Input image height and width must be multiples of 16.")

        if force_history.ndim == 2:
            force_history = force_history.unsqueeze(0)
        expected_history = (1, self.force_expert.observed_steps, self.force_expert.num_fingers)
        if tuple(force_history.shape) != expected_history:
            raise ValueError(f"`force_history` must be {expected_history}, got {tuple(force_history.shape)}")
        force_history = force_history.to(device=self.device, dtype=self.torch_dtype)
        if force_history_valid is None:
            force_history_valid = torch.ones(expected_history, device=self.device, dtype=torch.bool)
        else:
            if force_history_valid.ndim == 2:
                force_history_valid = force_history_valid.unsqueeze(0)
            if tuple(force_history_valid.shape) != expected_history:
                raise ValueError(
                    f"`force_history_valid` must be {expected_history}, got {tuple(force_history_valid.shape)}"
                )
            force_history_valid = force_history_valid.to(device=self.device, dtype=torch.bool)
        if force_history_is_pad is not None:
            if force_history_is_pad.ndim == 1:
                force_history_is_pad = force_history_is_pad.unsqueeze(0)
            expected_pad = (1, self.force_expert.observed_steps)
            if tuple(force_history_is_pad.shape) != expected_pad:
                raise ValueError(
                    f"`force_history_is_pad` must be {expected_pad}, got {tuple(force_history_is_pad.shape)}"
                )
            force_history_valid &= ~force_history_is_pad.to(
                device=self.device, dtype=torch.bool
            )[:, :, None]
        force_history_valid &= torch.isfinite(force_history)
        force_history = torch.where(
            force_history_valid, force_history, torch.zeros_like(force_history)
        )

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("Proprio was provided but the proprio encoder is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if tuple(proprio.shape) != (1, self.proprio_dim):
                raise ValueError(f"`proprio` must be [1,{self.proprio_dim}], got {tuple(proprio.shape)}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        if prompt is not None and (context is not None or context_mask is not None):
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if prompt is not None:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("Provide either `prompt` or both `context/context_mask`.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(context, context_mask, proprio)

        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        force_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed + 1)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        future_force = torch.randn(
            (1, self.force_expert.future_steps, self.force_expert.num_fingers),
            generator=force_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        timestep_video = torch.zeros((1,), device=self.device, dtype=self.torch_dtype)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        patch_t, patch_h, patch_w = (int(size) for size in self.video_expert.patch_size)
        latent_t, latent_h, latent_w = first_frame_latents.shape[-3:]
        tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)
        video_seq_len = (latent_t // patch_t) * tokens_per_frame
        va_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_horizon,
            video_tokens_per_frame=tokens_per_frame,
            device=self.device,
        )
        force_valid = torch.ones(
            (1, self.force_expert.total_steps, self.force_expert.num_fingers),
            device=self.device,
            dtype=torch.bool,
        )
        force_valid[:, : self.force_expert.observed_steps] = force_history_valid
        vfa_mask = self._build_force_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_horizon,
            video_tokens_per_frame=tokens_per_frame,
            device=self.device,
            force_valid=force_valid,
        )
        action_ts, action_deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=self.torch_dtype,
            shift_override=action_sigma_shift,
        )
        force_ts, force_deltas = self.infer_force_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=self.torch_dtype,
            shift_override=force_sigma_shift,
        )
        for action_t, action_delta, force_t, force_delta in zip(
            action_ts, action_deltas, force_ts, force_deltas
        ):
            latents_force = torch.cat([force_history, future_force], dim=1)
            _pred_video, pred_action, pred_force = self._joint_force_denoise_core(
                latents_video=first_frame_latents,
                latents_action=latents_action,
                latents_force=latents_force,
                timestep_video=timestep_video,
                timestep_action=action_t.reshape(1),
                timestep_force=force_t.reshape(1),
                context=context,
                context_mask=context_mask,
                va_attention_mask=va_mask,
                vfa_attention_mask=vfa_mask,
                force_valid=force_valid,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            latents_action = self.infer_action_scheduler.step(pred_action, action_delta, latents_action)
            future_force = self.infer_force_scheduler.step(
                pred_force[:, self.force_expert.observed_steps :], force_delta, future_force
            )

        full_force = torch.cat([force_history, future_force], dim=1)
        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
            "future_force": future_force[0].detach().to(device="cpu", dtype=torch.float32),
            "force": full_force[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    def save_checkpoint(
        self,
        path,
        optimizer=None,
        step=None,
        data_artifact_hashes: Optional[Mapping[str, str]] = None,
        training_metadata: Optional[Mapping[str, Any]] = None,
    ):
        payload = {
            "mot": self.mot.state_dict(),
            # Preserve compatibility for non-OpenTouch tasks that do not
            # provide data sidecars.
            "schema_version": 4 if data_artifact_hashes is not None else 2,
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "experiment_mode": self.experiment_mode,
            "initialization_mode": self.initialization_mode,
            "initialization_checkpoint": self.initialization_checkpoint,
            "initialization_source_step": self.initialization_source_step,
            "proprio_initialization": self.proprio_initialization,
        }
        if training_metadata is not None:
            payload.update(dict(training_metadata))
        if self.force_expert is not None:
            payload["force_meta"] = {
                "history_steps": self.force_expert.history_steps,
                "future_steps": self.force_expert.future_steps,
                "num_fingers": self.force_expert.num_fingers,
                "bridge_layers": dict(self.mot.force_bridge_layers),
            }
        if data_artifact_hashes is not None:
            payload["data_artifact_hashes"] = dict(data_artifact_hashes)
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(
        self,
        path,
        optimizer=None,
        data_artifact_dir: Optional[str] = None,
        strict_experiment_mode: bool = False,
    ):
        payload = torch.load(path, map_location="cpu")
        checkpoint_mode = self._checkpoint_experiment_mode(payload)
        if strict_experiment_mode:
            if checkpoint_mode is None:
                raise ValueError(f"Cannot infer experiment mode from checkpoint: {path}")
            if checkpoint_mode != self.experiment_mode:
                raise ValueError(
                    f"Checkpoint experiment_mode={checkpoint_mode!r} does not match "
                    f"model experiment_mode={self.experiment_mode!r}."
                )
        expected_artifact_hashes = payload.get("data_artifact_hashes")
        if expected_artifact_hashes is not None:
            artifact_dir = (
                Path(data_artifact_dir)
                if data_artifact_dir is not None
                else Path(path).with_suffix("")
            )
            verify_artifact_hashes(artifact_dir, expected_artifact_hashes)
            logger.info("Verified checkpoint data artifacts in %s", artifact_dir)
        elif int(payload.get("schema_version", 0)) >= 3:
            raise ValueError(
                f"Checkpoint schema_version={payload.get('schema_version')} has no data_artifact_hashes"
            )
        if "mot" in payload:
            incompatible = self.mot.load_state_dict(
                payload["mot"], strict=bool(strict_experiment_mode)
            )
            if incompatible.missing_keys:
                logger.warning(
                    "Checkpoint is missing %d MoT keys (expected for official checkpoints without ForceExpert): %s",
                    len(incompatible.missing_keys),
                    incompatible.missing_keys[:10],
                )
            if incompatible.unexpected_keys:
                logger.warning(
                    "Checkpoint contains %d unexpected MoT keys: %s",
                    len(incompatible.unexpected_keys),
                    incompatible.unexpected_keys[:10],
                )
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        self.initialization_mode = str(payload.get("initialization_mode", "unknown_legacy"))
        self.initialization_checkpoint = payload.get("initialization_checkpoint")
        self.initialization_source_step = payload.get("initialization_source_step")
        self.proprio_initialization = str(
            payload.get("proprio_initialization", "unknown_legacy")
        )
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
