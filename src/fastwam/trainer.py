import logging
import json
import inspect
import os
import re
import shutil
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .datasets.opentouch.evaluation import infer_opentouch_sample
from .datasets.opentouch.contract import artifact_hashes
from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.pretrained_learning_rate = float(
            cfg.get("pretrained_learning_rate", self.learning_rate)
        )
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        self.force_warmup_steps = int(cfg.get("force_warmup_steps", 0))
        self.joint_training_steps = int(
            cfg.get(
                "joint_training_steps",
                0 if self.max_steps is None else self.max_steps - self.force_warmup_steps,
            )
        )
        self.train_video_expert = bool(cfg.get("train_video_expert", True))
        self.allow_data_contract_mismatch = bool(
            cfg.get("allow_data_contract_mismatch", False)
        )
        self.force_ablation_every = int(cfg.get("force_ablation_every", 2000))
        self._active_training_stage = None

        model_uses_force = getattr(self.model, "force_expert", None) is not None
        for name, dataset in (("train", self.train_dataset), ("val", self.val_dataset)):
            if dataset is not None and hasattr(dataset, "include_force"):
                if bool(dataset.include_force) != model_uses_force:
                    raise ValueError(
                        f"OpenTouch {name} dataset include_force={dataset.include_force} does not match "
                        f"model force_expert enabled={model_uses_force}."
                    )
        if not model_uses_force and self.force_warmup_steps != 0:
            raise ValueError("force_warmup_steps must be 0 for the two-expert baseline.")
        if self.max_steps is not None and self.joint_training_steps > 0:
            expected_steps = self.force_warmup_steps + self.joint_training_steps
            if self.max_steps != expected_steps:
                raise ValueError(
                    f"max_steps={self.max_steps} must equal force_warmup_steps + "
                    f"joint_training_steps={expected_steps}."
                )
        
        self.resume = cfg.resume
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )

        deepspeed_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if deepspeed_plugin is None:
            zero_stage = "disabled"
        else:
            deepspeed_config = getattr(deepspeed_plugin, "deepspeed_config", None) or {}
            zero_stage = deepspeed_config.get("zero_optimization", {}).get("stage", "unknown")

        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            zero_stage,
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(
            self.model,
            train_video_expert=self.train_video_expert,
            force_warmup=False,
        )
        trainable_params = [param for param in self.model.dit.parameters() if param.requires_grad]
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend(
                param for param in proprio_encoder.parameters() if param.requires_grad
            )
        force_expert = getattr(self.model, "force_expert", None)
        new_param_ids = set()
        if force_expert is not None:
            new_param_ids.update(id(param) for param in force_expert.parameters())
            force_va_gates = getattr(self.model.mot, "force_va_gates", None)
            if force_va_gates is not None:
                new_param_ids.update(id(param) for param in force_va_gates.parameters())
        if proprio_encoder is not None:
            new_param_ids.update(id(param) for param in proprio_encoder.parameters())
        unique_params = {id(param): param for param in trainable_params}
        new_params = [param for param_id, param in unique_params.items() if param_id in new_param_ids]
        pretrained_params = [param for param_id, param in unique_params.items() if param_id not in new_param_ids]
        param_groups = []
        if pretrained_params:
            param_groups.append({"params": pretrained_params, "lr": self.pretrained_learning_rate})
        if new_params:
            param_groups.append({"params": new_params, "lr": self.learning_rate})
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        logger.info(
            "Optimizer parameter groups: pretrained=%d lr=%.2e, force/proprio=%d lr=%.2e",
            sum(param.numel() for param in pretrained_params),
            self.pretrained_learning_rate,
            sum(param.numel() for param in new_params),
            self.learning_rate,
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        scheduler_train_steps = (
            self.joint_training_steps if self.joint_training_steps > 0 else total_train_steps
        )
        warmup_steps = int(scheduler_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=scheduler_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.joint_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.run_data_artifact_dir = os.path.join(self.output_dir, "data_artifacts")
        if hasattr(self.train_dataset, "export_data_artifacts"):
            if self.accelerator.is_main_process:
                self.data_artifact_hashes = self.train_dataset.export_data_artifacts(
                    self.run_data_artifact_dir
                )
            self.accelerator.wait_for_everyone()
            self.train_dataset.validate_data_artifacts(self.run_data_artifact_dir)
            self.data_artifact_hashes = artifact_hashes(self.run_data_artifact_dir)
        else:
            self.data_artifact_hashes = None

        if bool(getattr(self.model, "compile_training_denoise", False)):
            logger.info("Compiling training denoise forward/backward before DeepSpeed initialization.")
            compile_start = time.perf_counter()
            warmup_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=self.train_sampler,
                num_workers=0,
            )
            warmup_sample = next(iter(warmup_loader))
            with self.accelerator.autocast():
                warmup_loss, _ = self.model.training_loss(warmup_sample)
            warmup_loss.backward()
            self.optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.accelerator.device)
            logger.info(
                "Finished training denoise compile warmup in %.2f seconds.",
                time.perf_counter() - compile_start,
            )
            set_global_seed(self.seed)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        checkpoint_artifact_dir = resume_path.with_suffix("")
        self._validate_checkpoint_artifacts(checkpoint_artifact_dir)
        payload = self.accelerator.unwrap_model(self.model).load_checkpoint(
            str(resume_path),
            optimizer=None,
            data_artifact_dir=str(checkpoint_artifact_dir),
            strict_experiment_mode=True,
        )
        restored_step = payload.get("global_step", payload.get("step"))
        if restored_step is not None:
            self.global_step = int(restored_step)
            self.joint_step = int(
                payload.get("joint_step", max(0, self.global_step - self.force_warmup_steps))
            )
            saved_phase = payload.get("phase")
            derived_phase = self._training_phase()
            if saved_phase is not None and saved_phase != derived_phase:
                raise ValueError(
                    f"Training state phase={saved_phase!r} is inconsistent with "
                    f"global_step={self.global_step} and configured warmup; expected {derived_phase!r}."
                )
        logger.warning(
            "Loaded .pt weights at global_step=%d joint_step=%d; optimizer/scheduler state was not restored. "
            "Use a checkpoint/state directory for an exact training resume.",
            self.global_step,
            self.joint_step,
        )

    def _validate_checkpoint_artifacts(self, artifact_dir: str | Path) -> None:
        if not hasattr(self.train_dataset, "validate_data_artifacts"):
            return
        try:
            self.train_dataset.validate_data_artifacts(artifact_dir)
        except (FileNotFoundError, ValueError) as error:
            if self.allow_data_contract_mismatch:
                logger.warning(
                    "Ignoring checkpoint/data contract mismatch because "
                    "allow_data_contract_mismatch=true: %s",
                    error,
                )
                return
            raise

    def _set_dit_only_train_mode(self):
        model = self.accelerator.unwrap_model(self.model)
        force_warmup = bool(
            getattr(model, "force_expert", None) is not None
            and self.global_step < self.force_warmup_steps
        )
        stage = "force_warmup" if force_warmup else "joint"
        if stage == self._active_training_stage and model.dit.training:
            return
        if stage != self._active_training_stage:
            logger.info(
                "Setting training stage=%s train_video_expert=%s at step=%d",
                stage,
                self.train_video_expert,
                self.global_step,
            )
            self._active_training_stage = stage
        self._apply_dit_only_train_mode(
            model,
            train_video_expert=self.train_video_expert,
            force_warmup=force_warmup,
        )

    def _training_phase(self) -> str:
        model = self.accelerator.unwrap_model(self.model)
        if getattr(model, "force_expert", None) is not None and self.global_step < self.force_warmup_steps:
            return "force_warmup"
        return "joint"

    @staticmethod
    def _apply_dit_only_train_mode(
        model,
        *,
        train_video_expert: bool = True,
        force_warmup: bool = False,
    ):
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)
        if not train_video_expert or force_warmup:
            model.video_expert.eval()
            model.video_expert.requires_grad_(False)
        if force_warmup:
            model.action_expert.eval()
            model.action_expert.requires_grad_(False)
            if proprio_encoder is not None:
                proprio_encoder.eval()
                proprio_encoder.requires_grad_(False)
            force_va_gates = getattr(model.mot, "force_va_gates", None)
            if force_va_gates is not None:
                force_va_gates.requires_grad_(False)
            force_expert = getattr(model, "force_expert", None)
            if force_expert is not None:
                force_expert.train()
                force_expert.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)
        force = sample.get("force", None)
        force_valid = sample.get("force_valid", None)
        force_is_pad = sample.get("force_is_pad", None)
        action_is_pad = sample.get("action_is_pad", None)
        image_is_pad = sample.get("image_is_pad", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        if force is not None:
            if force.ndim == 2:
                force = force.unsqueeze(0)
            if force.ndim != 3:
                raise ValueError(f"`force` must be [B,T,F], got {tuple(force.shape)}")
        if force_valid is not None and force_valid.ndim == 2:
            force_valid = force_valid.unsqueeze(0)
        if force_is_pad is not None and force_is_pad.ndim == 1:
            force_is_pad = force_is_pad.unsqueeze(0)
        if action_is_pad is not None and action_is_pad.ndim == 1:
            action_is_pad = action_is_pad.unsqueeze(0)
        if image_is_pad is not None and image_is_pad.ndim == 1:
            image_is_pad = image_is_pad.unsqueeze(0)

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
            "force": force,
            "force_valid": force_valid,
            "force_is_pad": force_is_pad,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    def _is_opentouch_force_eval(self, model, sample) -> bool:
        return bool(
            sample.get("force") is not None
            and getattr(model, "force_expert", None) is not None
            and hasattr(model, "infer_action_force")
            and hasattr(self.val_dataset, "denormalize_action")
            and hasattr(self.val_dataset, "denormalize_force")
        )

    def _is_opentouch_action_eval(self, model, sample) -> bool:
        return bool(
            sample.get("force") is None
            and getattr(model, "force_expert", None) is None
            and hasattr(model, "infer_action")
            and hasattr(self.val_dataset, "denormalize_action")
            and hasattr(self.val_dataset, "include_force")
        )

    @torch.no_grad()
    def _evaluate_opentouch_action(
        self,
        *,
        model,
        sample,
        eval_index: int,
        val_loss: float,
        val_loss_dict: dict,
        was_dit_training: bool,
    ) -> dict[str, float | str]:
        with self.accelerator.autocast():
            inference = infer_opentouch_sample(
                model=model,
                dataset=self.val_dataset,
                sample=sample,
                num_inference_steps=self.eval_num_inference_steps,
                seed=int(self.seed + eval_index),
            )
        local_metrics = dict(inference["metrics"])
        local_metrics.update(
            {
                "val_loss": float(val_loss),
                "val_loss_video": float(val_loss_dict.get("loss_video", float("nan"))),
                "val_loss_action": float(val_loss_dict.get("loss_action", float("nan"))),
            }
        )
        metric_names = [
            "val_loss",
            "val_loss_video",
            "val_loss_action",
            "action_mae",
            "action_rmse",
            "action_translation_mae_m",
            "action_rotation_mae_rad",
            "action_gripper_mae",
        ]
        gathered = self.accelerator.gather_for_metrics(
            torch.tensor(
                [local_metrics[name] for name in metric_names],
                device=self.accelerator.device,
                dtype=torch.float32,
            ).unsqueeze(0)
        )
        means = torch.nanmean(gathered, dim=0).cpu().tolist()
        result: dict[str, float | str] = {
            name: float(value) for name, value in zip(metric_names, means)
        }
        result["eval_mode"] = "opentouch_action"
        if self.accelerator.is_main_process:
            logger.info(
                "[eval-opentouch-action] index=%d prompt=%r action_mae=%.6f",
                eval_index,
                sample["prompt"][0],
                result["action_mae"],
            )
        if was_dit_training:
            self._set_dit_only_train_mode()
        return result

    @staticmethod
    def _format_metric_vector(values: np.ndarray) -> str:
        return "[" + ", ".join(f"{float(value):.5f}" for value in np.asarray(values).reshape(-1)) + "]"

    @torch.no_grad()
    def _evaluate_opentouch_force(
        self,
        *,
        model,
        sample,
        eval_index: int,
        val_loss: float,
        val_loss_dict: dict,
        was_dit_training: bool,
    ) -> dict[str, float | str]:
        observed_steps = int(model.force_expert.observed_steps)
        with self.accelerator.autocast():
            inference = infer_opentouch_sample(
                model=model,
                dataset=self.val_dataset,
                sample=sample,
                num_inference_steps=self.eval_num_inference_steps,
                seed=int(self.seed + eval_index),
            )
        local_metrics = dict(inference["metrics"])
        pred_action = inference["pred_action"]
        gt_action = inference["gt_action"]
        action_valid = inference["action_valid"]
        pred_force = inference["pred_force"]
        gt_force = inference["gt_force"]
        force_valid = inference["force_valid"]
        infer_kwargs = inference["inference_kwargs"]
        force_history = infer_kwargs["force_history"]
        history_valid = infer_kwargs["force_history_valid"]
        history_is_pad = infer_kwargs["force_history_is_pad"]
        run_ablations = bool(
            self.force_ablation_every > 0
            and self.joint_step > 0
            and self.joint_step % self.force_ablation_every == 0
        )
        ablation_step_gains: dict[str, np.ndarray] = {}
        if run_ablations:
            variants = {
                "zero": {
                    **infer_kwargs,
                    "force_history": torch.zeros_like(force_history),
                }
            }
            permutation = torch.randperm(
                observed_steps,
                generator=torch.Generator(device="cpu").manual_seed(
                    self.seed + eval_index + 17
                ),
            )
            variants["time_shuffled"] = {
                **infer_kwargs,
                "force_history": force_history[permutation],
                "force_history_valid": history_valid[permutation],
                "force_history_is_pad": (
                    None if history_is_pad is None else history_is_pad[permutation]
                ),
            }
            shuffled_index = (eval_index + max(1, len(self.val_dataset) // 2)) % len(
                self.val_dataset
            )
            if hasattr(self.val_dataset, "find_contact_window"):
                shuffled_index = self.val_dataset.find_contact_window(shuffled_index)
            shuffled_sample = self.val_dataset[shuffled_index]
            shuffled_force = shuffled_sample["force"][:observed_steps]
            shuffled_valid = shuffled_sample.get(
                "force_valid", torch.ones_like(shuffled_force, dtype=torch.bool)
            )[:observed_steps]
            shuffled_pad = shuffled_sample.get("force_is_pad")
            if shuffled_pad is not None:
                shuffled_pad = shuffled_pad[:observed_steps]
            variants["shuffled"] = {
                **infer_kwargs,
                "force_history": shuffled_force,
                "force_history_valid": shuffled_valid,
                "force_history_is_pad": shuffled_pad,
            }

            action_mask = np.broadcast_to(action_valid[:, None], gt_action.shape)
            correct_sq = np.square(pred_action - gt_action)
            correct_step_mse = correct_sq.mean(axis=1)
            correct_mse = float(correct_sq[action_mask].mean()) if action_mask.any() else float("nan")
            local_metrics["ablation_action_mse_correct"] = correct_mse
            for name, variant_kwargs in variants.items():
                with self.accelerator.autocast():
                    variant_pred = model.infer_action_force(**variant_kwargs)
                variant_action = self.val_dataset.denormalize_action(
                    variant_pred["action"].detach().cpu().numpy()
                )
                variant_sq = np.square(variant_action - gt_action)
                variant_step_mse = variant_sq.mean(axis=1)
                variant_mse = (
                    float(variant_sq[action_mask].mean())
                    if action_mask.any()
                    else float("nan")
                )
                local_metrics[f"ablation_action_mse_{name}"] = variant_mse
                local_metrics[f"ablation_relative_gain_vs_{name}"] = (
                    (variant_mse - correct_mse) / max(variant_mse, 1e-12)
                )
                step_gain = (variant_step_mse - correct_step_mse) / np.maximum(
                    variant_step_mse, 1e-12
                )
                step_gain[~action_valid] = np.nan
                ablation_step_gains[name] = step_gain.astype(np.float32)
        local_metrics["val_loss"] = float(val_loss)
        for key in (
            "loss_video",
            "loss_action",
            "raw_force_loss",
            "scheduler_weighted_force_loss",
            "loss_force",
            "force_valid_fraction",
            "force_valid_samples",
        ):
            local_metrics[f"val_{key}"] = float(val_loss_dict.get(key, float("nan")))

        metric_names = [
            "val_loss",
            "val_loss_video",
            "val_loss_action",
            "val_raw_force_loss",
            "val_scheduler_weighted_force_loss",
            "val_loss_force",
            "val_force_valid_fraction",
            "val_force_valid_samples",
            "action_mae",
            "action_rmse",
            "action_translation_mae_m",
            "action_rotation_mae_rad",
            "action_gripper_mae",
            "force_mae",
            "force_rmse",
            "force_thumb_mae",
            "force_index_mae",
        ]
        if run_ablations:
            metric_names.extend(
                [
                    "ablation_action_mse_correct",
                    "ablation_action_mse_zero",
                    "ablation_action_mse_shuffled",
                    "ablation_action_mse_time_shuffled",
                    "ablation_relative_gain_vs_zero",
                    "ablation_relative_gain_vs_shuffled",
                    "ablation_relative_gain_vs_time_shuffled",
                ]
            )
        local_tensor = torch.tensor(
            [local_metrics[name] for name in metric_names],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered = self.accelerator.gather_for_metrics(local_tensor)
        mean_values = torch.nanmean(gathered, dim=0).cpu().tolist()
        result: dict[str, float | str] = {
            name: float(value) for name, value in zip(metric_names, mean_values)
        }
        if run_ablations:
            for variant_name in ("zero", "shuffled", "time_shuffled"):
                metric_name = f"ablation_relative_gain_vs_{variant_name}"
                step_tensor = torch.as_tensor(
                    ablation_step_gains[variant_name],
                    device=self.accelerator.device,
                    dtype=torch.float32,
                ).unsqueeze(0)
                samples = (
                    self.accelerator.gather_for_metrics(step_tensor)
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                samples = samples[np.isfinite(samples)]
                if samples.size:
                    rng = np.random.default_rng(self.seed + self.global_step)
                    bootstrap = rng.choice(
                        samples,
                        size=(1000, samples.size),
                        replace=True,
                    ).mean(axis=1)
                    result[f"{metric_name}_ci95_low"] = float(
                        np.quantile(bootstrap, 0.025)
                    )
                    result[f"{metric_name}_ci95_high"] = float(
                        np.quantile(bootstrap, 0.975)
                    )
            shuffled_gain = result.get(
                "ablation_relative_gain_vs_shuffled", float("nan")
            )
            shuffled_ci_low = result.get(
                "ablation_relative_gain_vs_shuffled_ci95_low", float("nan")
            )
            result["force_usage_pass"] = float(
                np.isfinite(shuffled_gain)
                and np.isfinite(shuffled_ci_low)
                and shuffled_gain >= 0.02
                and shuffled_ci_low > 0.0
            )
        result["eval_mode"] = "opentouch_action_force"

        if self.accelerator.is_main_process:
            logger.info(
                "[eval-opentouch-sample] index=%d prompt=%r action_units=[m,m,m,rad,rad,rad,gripper] force_order=[thumb,index]",
                eval_index,
                sample["prompt"][0],
            )
            compare_steps = sorted(
                {
                    0,
                    min(7, gt_action.shape[0] - 1),
                    min(15, gt_action.shape[0] - 1),
                    gt_action.shape[0] - 1,
                }
            )
            for step in compare_steps:
                logger.info(
                    "[eval-opentouch-compare] future_step=%02d action_pred=%s action_gt=%s force_pred=%s force_gt=%s action_valid=%s force_valid=%s",
                    step + 1,
                    self._format_metric_vector(pred_action[step]),
                    self._format_metric_vector(gt_action[step]),
                    self._format_metric_vector(pred_force[step]),
                    self._format_metric_vector(gt_force[step]),
                    bool(action_valid[step]),
                    force_valid[step].tolist(),
                )

        if was_dit_training:
            self._set_dit_only_train_mode()
        return result

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # Each rank owns one stable validation window.  The set no longer
        # changes with global_step, so model comparisons use identical data.
        eval_index = (self.seed + self.accelerator.process_index) % len(self.val_dataset)
        if bool(getattr(self.val_dataset, "include_force", False)) and hasattr(
            self.val_dataset, "find_contact_window"
        ):
            eval_index = self.val_dataset.find_contact_window(eval_index)
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, val_loss_dict = model.training_loss(sample)
            val_loss = val_loss.float().item()

        if self._is_opentouch_force_eval(model, sample):
            run_generation = bool(
                self.force_ablation_every > 0
                and self.joint_step > 0
                and self.joint_step % self.force_ablation_every == 0
            )
            if not run_generation:
                metric_names = [
                    "val_loss",
                    "val_loss_video",
                    "val_loss_action",
                    "val_raw_force_loss",
                    "val_scheduler_weighted_force_loss",
                    "val_loss_force",
                    "val_force_valid_fraction",
                    "val_force_valid_samples",
                ]
                local_values = [
                    val_loss,
                    float(val_loss_dict.get("loss_video", float("nan"))),
                    float(val_loss_dict.get("loss_action", float("nan"))),
                    float(val_loss_dict.get("raw_force_loss", float("nan"))),
                    float(
                        val_loss_dict.get(
                            "scheduler_weighted_force_loss", float("nan")
                        )
                    ),
                    float(val_loss_dict.get("loss_force", float("nan"))),
                    float(
                        val_loss_dict.get("force_valid_fraction", float("nan"))
                    ),
                    float(val_loss_dict.get("force_valid_samples", float("nan"))),
                ]
                gathered = self.accelerator.gather_for_metrics(
                    torch.tensor(
                        local_values,
                        device=self.accelerator.device,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                )
                means = torch.nanmean(gathered, dim=0).cpu().tolist()
                result = {
                    name: float(value) for name, value in zip(metric_names, means)
                }
                result["eval_mode"] = "opentouch_loss_only"
                if was_dit_training:
                    self._set_dit_only_train_mode()
                return result
            return self._evaluate_opentouch_force(
                model=model,
                sample=sample,
                eval_index=eval_index,
                val_loss=val_loss,
                val_loss_dict=val_loss_dict,
                was_dit_training=was_dit_training,
            )
        if self._is_opentouch_action_eval(model, sample):
            return self._evaluate_opentouch_action(
                model=model,
                sample=sample,
                eval_index=eval_index,
                val_loss=val_loss,
                val_loss_dict=val_loss_dict,
                was_dit_training=was_dit_training,
            )
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        checkpoint_artifact_dir = os.path.join(self.weights_dir, step_tag)
        checkpoint_artifact_hashes = self._copy_data_artifacts(checkpoint_artifact_dir)
        model.save_checkpoint(
            ckpt_path,
            optimizer=None,
            step=self.global_step,
            data_artifact_hashes=checkpoint_artifact_hashes,
            training_metadata={
                "global_step": int(self.global_step),
                "joint_step": int(self.joint_step),
                "phase": self._training_phase(),
            },
        )
        return ckpt_path

    def _copy_data_artifacts(self, destination: str | Path):
        if self.data_artifact_hashes is None:
            return None
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for filename in self.data_artifact_hashes:
            shutil.copy2(Path(self.run_data_artifact_dir) / filename, destination / filename)
        copied_hashes = artifact_hashes(destination)
        if copied_hashes != self.data_artifact_hashes:
            raise ValueError(
                f"Copied data artifact hashes differ: {copied_hashes} != {self.data_artifact_hashes}"
            )
        return copied_hashes

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        model = self.accelerator.unwrap_model(self.model)
        payload = {
            "global_step": int(self.global_step),
            "joint_step": int(self.joint_step),
            "phase": self._training_phase(),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
            "data_artifact_hashes": self.data_artifact_hashes,
            "experiment_mode": getattr(model, "experiment_mode", None),
            "initialization_mode": getattr(model, "initialization_mode", "unknown_legacy"),
            "initialization_checkpoint": getattr(model, "initialization_checkpoint", None),
            "initialization_source_step": getattr(model, "initialization_source_step", None),
            "proprio_initialization": getattr(model, "proprio_initialization", "unknown_legacy"),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        if self.accelerator.is_main_process:
            self._copy_data_artifacts(state_path)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self._validate_checkpoint_artifacts(state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        payload = None
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            model = self.accelerator.unwrap_model(self.model)
            saved_mode = payload.get("experiment_mode")
            if saved_mode is not None and saved_mode != getattr(model, "experiment_mode", None):
                raise ValueError(
                    f"Training state experiment_mode={saved_mode!r} does not match "
                    f"model experiment_mode={getattr(model, 'experiment_mode', None)!r}."
                )
        self.accelerator.load_state(input_dir=state_dir)
        if payload is not None:
            self.global_step = int(payload["global_step"])
            self.joint_step = int(
                payload.get("joint_step", max(0, self.global_step - self.force_warmup_steps))
            )
            model = self.accelerator.unwrap_model(self.model)
            if hasattr(model, "set_initialization_metadata"):
                model.set_initialization_metadata(
                    mode=str(payload.get("initialization_mode", "unknown_legacy")),
                    checkpoint=payload.get("initialization_checkpoint"),
                    source_step=payload.get("initialization_source_step"),
                )
                model.proprio_initialization = str(
                    payload.get("proprio_initialization", "unknown_legacy")
                )
            expected_hashes = payload.get("data_artifact_hashes")
            if expected_hashes is not None:
                actual_hashes = artifact_hashes(state_dir)
                if actual_hashes != expected_hashes:
                    raise ValueError(
                        f"Training-state data artifact hashes differ: {actual_hashes} != {expected_hashes}"
                    )

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.joint_step = max(0, self.global_step - self.force_warmup_steps)
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        while self.global_step < self.max_steps:
            # This is a no-op except at the Force-warmup -> joint boundary.
            self._set_dit_only_train_mode()
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    completed_joint_update = self._training_phase() == "joint"
                    gate_grad_metrics = {}
                    unwrapped_for_grads = self.accelerator.unwrap_model(self.model)
                    force_va_gates = getattr(unwrapped_for_grads.mot, "force_va_gates", None)
                    if force_va_gates is not None:
                        for layer, parameter in force_va_gates.items():
                            grad = parameter.grad
                            gate_grad_metrics[f"force_gate_grad_{layer}"] = (
                                0.0 if grad is None else float(grad.detach().float().abs().item())
                            )
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if (
                        not self.accelerator.optimizer_step_was_skipped
                        and completed_joint_update
                    ):
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    if completed_joint_update:
                        self.joint_step += 1
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    for key, value in gate_grad_metrics.items():
                        metric_tensor = torch.tensor(
                            value, device=loss.device, dtype=torch.float32
                        ).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d joint_step=%d phase=%s loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            self.joint_step,
                            self._training_phase(),
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "train/global_step": self.global_step,
                            "train/joint_step": self.joint_step,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                    if (
                        completed_joint_update
                        and self.eval_every > 0
                        and self.val_dataset is not None
                        and self.joint_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            if metrics.get("eval_mode") in {
                                "opentouch_action",
                                "opentouch_action_force",
                                "opentouch_loss_only",
                            }:
                                if metrics.get("eval_mode") == "opentouch_action":
                                    logger.info(
                                        "[eval-opentouch-baseline] step=%d joint_step=%d val_loss=%.4f action_mae=%.6f action_rmse=%.6f translation_mae_m=%.6f rotation_mae_rad=%.6f gripper_mae=%.6f",
                                        self.global_step,
                                        self.joint_step,
                                        metrics["val_loss"],
                                        metrics["action_mae"],
                                        metrics["action_rmse"],
                                        metrics["action_translation_mae_m"],
                                        metrics["action_rotation_mae_rad"],
                                        metrics["action_gripper_mae"],
                                    )
                                    eval_payload = {
                                        f"eval/{key}": float(value)
                                        for key, value in metrics.items()
                                        if key != "eval_mode" and isinstance(value, (int, float))
                                    }
                                elif metrics.get("eval_mode") == "opentouch_loss_only":
                                    logger.info(
                                        "[eval-opentouch-loss-only] step=%d val_loss=%.4f video=%.4f action=%.4f force_raw=%.4f force_scheduler_weighted=%.4f force_final=%.4f valid_fraction=%.4f valid_samples=%.1f",
                                        self.global_step,
                                        metrics["val_loss"],
                                        metrics["val_loss_video"],
                                        metrics["val_loss_action"],
                                        metrics["val_raw_force_loss"],
                                        metrics["val_scheduler_weighted_force_loss"],
                                        metrics["val_loss_force"],
                                        metrics["val_force_valid_fraction"],
                                        metrics["val_force_valid_samples"],
                                    )
                                    eval_payload = {
                                        f"eval/{key}": float(value)
                                        for key, value in metrics.items()
                                        if key != "eval_mode"
                                        and isinstance(value, (int, float))
                                    }
                                else:
                                    description = (
                                        "[eval-opentouch] step=%d val_loss=%.4f "
                                        "action_mae=%.6f action_rmse=%.6f "
                                        "translation_mae_m=%.6f rotation_mae_rad=%.6f gripper_mae=%.6f "
                                        "force_mae=%.4f force_rmse=%.4f thumb_mae=%.4f index_mae=%.4f"
                                    ) % (
                                        self.global_step,
                                        metrics["val_loss"],
                                        metrics["action_mae"],
                                        metrics["action_rmse"],
                                        metrics["action_translation_mae_m"],
                                        metrics["action_rotation_mae_rad"],
                                        metrics["action_gripper_mae"],
                                        metrics["force_mae"],
                                        metrics["force_rmse"],
                                        metrics["force_thumb_mae"],
                                        metrics["force_index_mae"],
                                    )
                                    logger.info(description)
                                    logger.info(
                                        "[eval-opentouch-loss] video=%.4f action=%.4f force_raw=%.4f force_scheduler_weighted=%.4f force_final=%.4f valid_fraction=%.4f valid_samples=%.1f",
                                        metrics["val_loss_video"],
                                        metrics["val_loss_action"],
                                        metrics["val_raw_force_loss"],
                                        metrics["val_scheduler_weighted_force_loss"],
                                        metrics["val_loss_force"],
                                        metrics["val_force_valid_fraction"],
                                        metrics["val_force_valid_samples"],
                                    )
                                    if "ablation_relative_gain_vs_zero" in metrics:
                                        logger.info(
                                            "[eval-force-ablation] gain_vs_zero=%.4f gain_vs_shuffled=%.4f gain_vs_time_shuffled=%.4f shuffled_ci95=[%.4f, %.4f] usage_pass=%s",
                                            metrics["ablation_relative_gain_vs_zero"],
                                            metrics["ablation_relative_gain_vs_shuffled"],
                                            metrics["ablation_relative_gain_vs_time_shuffled"],
                                            metrics.get(
                                                "ablation_relative_gain_vs_shuffled_ci95_low",
                                                float("nan"),
                                            ),
                                            metrics.get(
                                                "ablation_relative_gain_vs_shuffled_ci95_high",
                                                float("nan"),
                                            ),
                                            bool(metrics.get("force_usage_pass", 0.0)),
                                        )
                                    eval_payload = {
                                        f"eval/{key}": float(value)
                                        for key, value in metrics.items()
                                        if key != "eval_mode"
                                        and isinstance(value, (int, float))
                                    }
                            else:
                                description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                    self.global_step,
                                    metrics["val_loss"],
                                    metrics["psnr_rd"],
                                    metrics["ssim_rd"],
                                )
                                if "action_l2" in metrics:
                                    description += " action_l2=%.4f" % metrics["action_l2"]
                                if "action_l1" in metrics:
                                    description += " action_l1=%.4f" % metrics["action_l1"]
                                logger.info(description)
                                eval_payload = {
                                    "eval/val_loss": float(metrics["val_loss"]),
                                    "eval/psnr_rg": float(metrics["psnr_rg"]),
                                    "eval/ssim_rg": float(metrics["ssim_rg"]),
                                    "eval/psnr_rd": float(metrics["psnr_rd"]),
                                    "eval/ssim_rd": float(metrics["ssim_rd"]),
                                    "eval/psnr_dg": float(metrics["psnr_dg"]),
                                    "eval/ssim_dg": float(metrics["ssim_dg"]),
                                }
                                if "action_l2" in metrics:
                                    eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                                if "action_l1" in metrics:
                                    eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                    if (
                        completed_joint_update
                        and self.save_every > 0
                        and self.joint_step % self.save_every == 0
                    ):
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        
