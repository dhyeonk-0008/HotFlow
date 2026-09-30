"""Training script for Approach B: cross-attention conditioned flow matching.

Supports both single-GPU and DDP multi-GPU training.

Usage:
    # Single GPU
    python hotflow/train_b.py --config hotflow/configs/train_b.yaml --device cuda:0

    # DDP multi-GPU
    torchrun --nproc_per_node=4 hotflow/train_b.py --config hotflow/configs/train_b.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.distributed as distrib
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

REPO_ROOT = Path(__file__).resolve().parents[1]
PEPFLOW_ROOT = REPO_ROOT / "PepFlowww"
# Remove the script's own directory (hotflow/) that Python auto-adds to sys.path[0],
# otherwise `from data import ...` resolves to hotflow/data/ instead of PepFlowww/data/.
_script_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _script_dir]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PEPFLOW_ROOT))

import wandb  # noqa: E402
from pepflow.utils.data import PaddingCollate, DEFAULT_PAD_VALUES, DEFAULT_NO_PADDING  # noqa: E402
from pepflow.utils.misc import (  # noqa: E402
    BlackHole, load_config, seed_all,
    get_logger, get_new_log_dir, current_milli_time,
)
from pepflow.utils.train import (  # noqa: E402
    count_parameters, get_optimizer, get_scheduler,
    log_losses, recursive_to, sum_weighted_losses,
)
from models_con.pep_dataloader import PepDataset  # noqa: E402
from models_con.utils import process_dic  # noqa: E402

from hotflow.models.flow_model_b import FlowModelB  # noqa: E402
from hotflow.data.dataset_b import PepDatasetB, HOTSPOT_PAD_VALUES, HOTSPOT_NO_PADDING  # noqa: E402
from hotflow.data.transforms import (  # noqa: E402
    HotspotAnnotationTransform, PepHARAnchorTransform,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train FlowModelB (Approach B)")
    parser.add_argument("--config", type=str, default="hotflow/configs/train_b.yaml")
    parser.add_argument("--logdir", type=str, default="./logs_b")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--device", type=str, default=None,
                        help="Device for single-GPU training (e.g. cuda:0). "
                             "Omit for DDP.")
    parser.add_argument("--local-rank", type=int, default=0,
                        help="Local rank for DDP (set by torchrun).")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--name", type=str, default="hotflow_b")
    return parser.parse_args()


def inf_iterator_ddp(loader, sampler=None):
    """Infinite iterator that calls sampler.set_epoch() on each wrap-around."""
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def build_collate_fn():
    """Build PaddingCollate with hotspot field support."""
    pad_values = {**DEFAULT_PAD_VALUES, **HOTSPOT_PAD_VALUES}
    no_padding = set(DEFAULT_NO_PADDING) | HOTSPOT_NO_PADDING
    return PaddingCollate(eight=False, pad_values=pad_values, no_padding=no_padding)


def build_model(config, device):
    """Instantiate FlowModelB with hotspot config."""
    hs_cfg = config.hotspot
    model = FlowModelB(
        config.model,
        d_hotspot=hs_cfg.d_hotspot,
        cross_attn_heads=hs_cfg.cross_attn_heads,
        cross_attn_blocks=tuple(hs_cfg.cross_attn_blocks),
        hotspot_dropout_p=hs_cfg.hotspot_dropout_p,
        num_hotspot_anchors=hs_cfg.num_anchors,
        contact_distance_cutoff=hs_cfg.contact_distance_cutoff,
        contact_loss_target=hs_cfg.contact_loss_target,
        contact_loss_margin=hs_cfg.contact_loss_margin,
        use_se3_invariant=getattr(hs_cfg, "use_se3_invariant", True),
        use_anchor_self_attn=getattr(hs_cfg, "use_anchor_self_attn", True),
        num_self_attn_layers=getattr(hs_cfg, "num_self_attn_layers", 1),
    )

    # Load pretrained PepFlow weights
    pretrained_cfg = config.get("pretrained", None)
    if pretrained_cfg and pretrained_cfg.checkpoint:
        ckpt_path = pretrained_cfg.checkpoint
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(str(REPO_ROOT), ckpt_path)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt["model"] if "model" in ckpt else ckpt
            # Handle DDP prefix
            cleaned = {}
            for k, v in state_dict.items():
                key = k.replace("module.", "") if k.startswith("module.") else k
                cleaned[key] = v
            # Also try process_dic from PepFlow for other prefix patterns
            try:
                cleaned = process_dic(cleaned)
            except Exception:
                pass
            missing, unexpected = model.load_state_dict(cleaned, strict=False)
            print(f"[Pretrained] Loaded from {ckpt_path}")
            print(f"  Missing keys: {len(missing)} (expected: cross-attn + hotspot encoder)")
            print(f"  Unexpected keys: {len(unexpected)}")
        else:
            print(f"[Warning] Pretrained checkpoint not found: {ckpt_path}")

    return model.to(device)


def main():
    args = parse_args()

    # Determine if DDP or single-GPU based on torchrun environment variables
    use_ddp = ("WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1)
    if use_ddp:
        # DDP: local_rank from torchrun
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        distrib.init_process_group(backend="nccl")
        is_main = local_rank == 0
    else:
        device = torch.device(args.device or "cuda:0")
        local_rank = 0
        is_main = True

    # Load config
    config, config_name = load_config(args.config)
    seed_all(config.train.seed + local_rank * 100)

    # Logging
    if args.debug or not is_main:
        wandb.init(mode="disabled")  # no-op wandb so log_losses doesn't crash
        if args.debug and is_main:
            if args.resume:
                log_dir = os.path.dirname(os.path.dirname(args.resume))
            else:
                log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
            ckpt_dir = os.path.join(log_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            logger = get_logger("train_b", log_dir)
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
        else:
            logger = get_logger("train_b", None, local_rank)
    else:
        run = wandb.init(
            project=args.name, config=dict(config),
            name=f"{config_name}[{args.tag}]" if args.tag else config_name,
        )
        if args.resume:
            log_dir = os.path.dirname(os.path.dirname(args.resume))
        else:
            log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
        ckpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        logger = get_logger("train_b", log_dir)
        shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    logger.info(f"Args: {args}")
    logger.info(f"Config: {config}")

    # Data
    logger.info("Loading datasets...")
    collate_fn = build_collate_fn()

    train_base = PepDataset(
        structure_dir=config.dataset.train.structure_dir,
        dataset_dir=config.dataset.train.dataset_dir,
        name=config.dataset.train.name,
        transform=None,
        reset=config.dataset.train.reset,
    )

    # Build hotspot transform from config.
    # If `hotspot.pephar_anchor_cache` is set, load PepHAR-EBM pre-computed
    # anchors from the LMDB cache. Otherwise fall back to the original
    # GT-contact transform (matches the proposal §3.1 default).
    hs_cfg = config.hotspot
    pephar_cache = hs_cfg.get("pephar_anchor_cache", None)
    if pephar_cache:
        cache_abs = pephar_cache if os.path.isabs(pephar_cache) else \
            os.path.join(str(REPO_ROOT), pephar_cache)
        if not os.path.exists(cache_abs):
            raise FileNotFoundError(
                f"pephar_anchor_cache not found: {cache_abs}\n"
                f"Run `python hotflow/scripts/precompute_pephar_anchors.py` first."
            )
        hotspot_transform = PepHARAnchorTransform(
            cache_path=cache_abs,
            num_anchors=hs_cfg.num_anchors,
            distance_cutoff=hs_cfg.contact_distance_cutoff,
            augment_sigma=float(hs_cfg.get("anchor_augment_sigma", 0.0)),
            fallback_gt_on_miss=bool(hs_cfg.get("fallback_gt_on_miss", True)),
        )
        logger.info(f"[Hotspot] PepHARAnchorTransform from {cache_abs} "
                    f"(augment_sigma={hs_cfg.get('anchor_augment_sigma', 0.0)})")
    else:
        hotspot_transform = HotspotAnnotationTransform(
            num_anchors=hs_cfg.num_anchors,
            distance_cutoff=hs_cfg.contact_distance_cutoff,
        )
        logger.info("[Hotspot] HotspotAnnotationTransform (GT contact, default)")

    train_dataset = PepDatasetB(
        train_base,
        num_anchors=hs_cfg.num_anchors,
        distance_cutoff=hs_cfg.contact_distance_cutoff,
        hotspot_transform=hotspot_transform,
    )

    train_sampler = None
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(
            train_dataset, batch_size=config.train.batch_size,
            collate_fn=collate_fn, sampler=train_sampler,
            num_workers=args.num_workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=config.train.batch_size,
            collate_fn=collate_fn, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )

    train_iterator = inf_iterator_ddp(train_loader, sampler=train_sampler)
    logger.info(f"Train dataset: {len(train_dataset)} samples")

    # Model
    logger.info("Building model...")
    model = build_model(config, device)

    if use_ddp:
        # find_unused_parameters=True: hotspot_dropout_p (default 0.1) drops
        # the cross-attn / HotspotEncoder path entirely on ~10% of steps,
        # leaving those params without grads on that step. DDP needs to be
        # told this is OK or it errors with "Expected to have finished
        # reduction in the prior iteration".
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    logger.info(f"Number of parameters: {count_parameters(model):,}")

    # Optimizer & Scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_first = 1

    # Resume
    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        it_first = ckpt["iteration"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])

    # Training loop
    accum_steps = getattr(config.train, "grad_accum_steps", 1)
    warmup_iters = getattr(config.train, "warmup_iters", 0)
    base_lr = config.train.optimizer.lr
    if is_main:
        logger.info(f"Gradient accumulation steps: {accum_steps} (effective batch size: {config.train.batch_size * accum_steps})")
        if warmup_iters > 0:
            logger.info(f"LR warmup: {warmup_iters} iters (0 -> {base_lr})")

    def set_lr(it):
        """Linear warmup then constant."""
        if warmup_iters > 0 and it <= warmup_iters:
            lr = base_lr * it / warmup_iters
        else:
            lr = base_lr
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    def train_step(it):
        time_start = current_milli_time()
        model.train()
        optimizer.zero_grad()
        set_lr(it)

        accum_loss = 0.0
        accum_loss_dict = {}
        nan_detected = False

        for accum_idx in range(accum_steps):
            batch = recursive_to(next(train_iterator), device)

            # DDP + grad accumulation: skip grad all-reduce on non-final accum
            # steps via `no_sync()`. This avoids two issues at once:
            #   (a) Wasteful all-reduce on intermediate accum steps.
            #   (b) DDP reducer state confusion when the set of "unused" params
            #       differs between accum steps (e.g., hotspot dropout fires
            #       on accum 1 but not accum 2), which under
            #       find_unused_parameters=True can hang grad sync.
            if use_ddp and accum_idx < accum_steps - 1:
                sync_ctx = model.no_sync()
            else:
                sync_ctx = contextlib.nullcontext()

            with sync_ctx:
                # Forward
                loss_dict = model(batch)
                loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                loss = loss / accum_steps

                # NaN guard. Under DDP we cannot just `return` here: peer ranks
                # would still be in backward() awaiting our grad all-reduce,
                # producing a silent hang. Instead, sanitize the loss in-graph
                # so backward proceeds with a zero contribution from this
                # accum step. The post-backward NaN-grad rescue handles any
                # residual NaN grads from other paths.
                loss_finite = torch.isfinite(loss)
                if not bool(loss_finite.item()):
                    nan_detected = True
                    if is_main:
                        logger.warning(f"[Iter {it}] NaN/Inf loss in accum {accum_idx}, zeroing")
                    loss = torch.where(loss_finite, loss, torch.zeros_like(loss))

                # Backward
                loss.backward()

                accum_loss += loss.item()
                for k, v in loss_dict.items():
                    if k not in accum_loss_dict:
                        accum_loss_dict[k] = 0.0
                    accum_loss_dict[k] += v.item() / accum_steps

        time_forward_end = current_milli_time()

        # NaN grad rescue
        for param in model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                param.grad[torch.isnan(param.grad)] = 0

        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

        optimizer.step()
        time_backward_end = current_milli_time()

        # Logging
        if is_main:
            scalar_dict = {
                "grad": orig_grad_norm,
                "lr": optimizer.param_groups[0]["lr"],
                "time_forward": (time_forward_end - time_start) / 1000,
                "time_backward": (time_backward_end - time_forward_end) / 1000,
            }
            log_losses(loss, loss_dict, scalar_dict, it=it, tag="train", logger=logger)

    try:
        for it in range(it_first, config.train.max_iters + 1):
            train_step(it)

            if it % config.train.val_freq == 0 and is_main:
                ckpt_path = os.path.join(ckpt_dir, f"{it}.pt")
                torch.save(
                    {
                        "config": config,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "iteration": it,
                    },
                    ckpt_path,
                )
                logger.info(f"Checkpoint saved: {ckpt_path}")

    except KeyboardInterrupt:
        logger.info("Terminating...")
    finally:
        if use_ddp:
            distrib.destroy_process_group()


if __name__ == "__main__":
    main()
