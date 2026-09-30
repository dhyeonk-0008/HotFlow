"""Training script for HotspotPredictor (Option D).

Run from project root:
    PYTHONPATH=$PWD python -m hotflow.train_hotspot \
        --config hotflow/configs/train_hotspot.yaml \
        --logdir logs_hotspot
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "PepFlowww") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "PepFlowww"))

from pepflow.modules.protein.constants import BBHeavyAtom  # noqa: E402
from pepflow.utils.data import (  # noqa: E402
    DEFAULT_NO_PADDING,
    DEFAULT_PAD_VALUES,
    PaddingCollate,
)
from pepflow.utils.train import recursive_to  # noqa: E402
from models_con.pep_dataloader import PepDataset  # noqa: E402

from hotflow.data.dataset_b import (  # noqa: E402
    HOTSPOT_NO_PADDING,
    HOTSPOT_PAD_VALUES,
    PepDatasetB,
)
from hotflow.models.hotspot_predictor import (  # noqa: E402
    HotspotPredictor,
    hungarian_match_and_loss,
)
from hotflow.models.hotspot_predictor_se3 import HotspotPredictorSE3  # noqa: E402
from hotflow.models.hotspot_predictor_v2 import HotspotPredictorV2  # noqa: E402


def build_loaders(config):
    collate_pad = {**DEFAULT_PAD_VALUES, **HOTSPOT_PAD_VALUES}
    collate_nopad = set(DEFAULT_NO_PADDING) | HOTSPOT_NO_PADDING
    collate_fn = PaddingCollate(eight=False, pad_values=collate_pad, no_padding=collate_nopad)

    tcfg = config.dataset.train
    train_base = PepDataset(
        structure_dir=tcfg.structure_dir,
        dataset_dir=tcfg.dataset_dir,
        name=tcfg.name,
        transform=None,
        reset=False,
    )
    train_ds = PepDatasetB(
        train_base,
        num_anchors=config.hotspot.num_anchors,
        distance_cutoff=config.hotspot.contact_distance_cutoff,
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.train.batch_size, shuffle=True,
        num_workers=config.train.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
    )

    vcfg = config.dataset.val
    if Path(vcfg.structure_dir).exists():
        val_base = PepDataset(
            structure_dir=vcfg.structure_dir,
            dataset_dir=vcfg.dataset_dir,
            name=vcfg.name,
            transform=None,
            reset=False,
        )
        val_ds = PepDatasetB(
            val_base,
            num_anchors=config.hotspot.num_anchors,
            distance_cutoff=config.hotspot.contact_distance_cutoff,
        )
        val_loader = DataLoader(
            val_ds, batch_size=config.train.batch_size, shuffle=False,
            num_workers=config.train.num_workers, collate_fn=collate_fn,
            pin_memory=True,
        )
    else:
        val_loader = None

    return train_loader, val_loader


def _random_rotation(B: int, device) -> torch.Tensor:
    """Generate B random SO(3) rotation matrices uniformly via QR of Gaussian."""
    A = torch.randn(B, 3, 3, device=device)
    Q, _ = torch.linalg.qr(A)
    # Force determinant +1 (proper rotation, not reflection)
    det = torch.det(Q)
    Q = Q.clone()
    Q[..., 0] = Q[..., 0] * det.sign().unsqueeze(-1)
    return Q


def _rotate_coords_batched(coords: torch.Tensor, R_g: torch.Tensor) -> torch.Tensor:
    """Apply per-sample rotation. coords (B, ..., 3), R_g (B, 3, 3) → (B, ..., 3)."""
    B = coords.shape[0]
    orig_shape = coords.shape
    flat = coords.reshape(B, -1, 3)  # (B, N, 3)
    rotated = flat @ R_g.transpose(-2, -1)  # x_new = R @ x ⇔ x.T @ R.T
    return rotated.reshape(orig_shape)


def extract_receptor(batch, device, mask_prob: float = 0.0):
    """Slice receptor backbone (N, CA, C) from a collated PepDatasetB batch.

    Args:
        mask_prob: if > 0, randomly drop this fraction of valid receptor
            residues from rec_mask each forward pass (BERT-style augmentation).

    Returns:
        rec_coords: (B, R_max, 3, 3) — (N, CA, C)
        rec_aa:     (B, R_max) long
        rec_mask:   (B, R_max) bool
    """
    gen_mask = batch["generate_mask"].bool()
    res_mask = batch["res_mask"].bool()
    rec_mask = (~gen_mask) & res_mask  # (B, L_max)

    pos = batch["pos_heavyatom"]  # (B, L_max, A, 3)
    aa = batch["aa"]              # (B, L_max)

    N_i = int(BBHeavyAtom.N); CA_i = int(BBHeavyAtom.CA); C_i = int(BBHeavyAtom.C)
    # Stack along atom axis to get (B, L_max, 3, 3): order = N, CA, C
    rec_coords_full = torch.stack(
        [pos[..., N_i, :], pos[..., CA_i, :], pos[..., C_i, :]], dim=-2,
    )
    # Where receptor mask is False, zero-out coords/aa to avoid noisy features
    rec_coords_full = rec_coords_full * rec_mask[..., None, None].float()
    aa_safe = torch.where(rec_mask, aa, torch.full_like(aa, 20))

    if mask_prob > 0.0:
        # Drop fraction of valid receptor residues per sample (training aug)
        # Guarantee each sample keeps at least 5 valid residues
        drop = torch.rand_like(rec_mask, dtype=torch.float) < mask_prob
        new_mask = rec_mask & ~drop
        kept = new_mask.sum(dim=-1)
        min_keep = 5
        for b in range(rec_mask.shape[0]):
            if int(kept[b].item()) < min_keep and bool(rec_mask[b].any()):
                # Restore enough random valid residues to hit min_keep
                valid_idx = rec_mask[b].nonzero(as_tuple=True)[0]
                need = min_keep - int(kept[b].item())
                # Force keep `need` random valid positions
                perm = valid_idx[torch.randperm(valid_idx.numel())[:need]]
                new_mask[b, perm] = True
        rec_mask = new_mask
        # Also zero coords of newly-dropped residues
        rec_coords_full = rec_coords_full * rec_mask[..., None, None].float()
        aa_safe = torch.where(rec_mask, aa_safe, torch.full_like(aa_safe, 20))

    return (
        recursive_to(rec_coords_full, device),
        recursive_to(aa_safe, device),
        recursive_to(rec_mask, device),
    )


def extract_gt_anchors(batch, device):
    gt_coords_dataset = batch["anchor_coords"].to(device)  # (B, K, 3, 3) — (CA, C, N) order per transform
    gt_aa = batch["anchor_types"].to(device)
    gt_mask = batch["anchor_mask"].to(device).bool()

    # transforms.py stores [CA, C, N] but model predicts [N, CA, C] — reorder GT.
    # (anchor_coords[..., 0]=CA, 1=C, 2=N → permute to N=2, CA=0, C=1)
    gt_coords = torch.stack(
        [gt_coords_dataset[..., 2, :], gt_coords_dataset[..., 0, :], gt_coords_dataset[..., 1, :]],
        dim=-2,
    )
    return gt_coords, gt_aa, gt_mask


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches: int = 50):
    model.eval()
    total_total = 0.0
    total_coord = 0.0
    total_aa = 0.0
    n_batches = 0
    n_matched_total = 0
    ca_err_sum = 0.0
    aa_correct = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        rec_coords, rec_aa, rec_mask = extract_receptor(batch, device)
        gt_coords, gt_aa, gt_mask = extract_gt_anchors(batch, device)
        pred_coords, pred_aa = model(rec_coords, rec_aa, rec_mask)
        out = hungarian_match_and_loss(
            pred_coords, pred_aa, gt_coords, gt_aa, gt_mask,
            coord_weight=1.0, aa_weight=1.0, match_metric="ca_l1",
        )
        total_total += float(out["total"].item())
        total_coord += float(out["coord"].item())
        total_aa += float(out["aa"].item())
        n_batches += 1

        # Diagnostic: mean CA-coord L2 error on matched anchors
        from scipy.optimize import linear_sum_assignment
        B = gt_mask.shape[0]
        for b in range(B):
            vi = gt_mask[b].nonzero(as_tuple=True)[0]
            if vi.numel() == 0:
                continue
            pred_ca = pred_coords[b, :, 1, :]
            gt_ca = gt_coords[b, vi, 1, :]
            cost = (pred_ca.unsqueeze(1) - gt_ca.unsqueeze(0)).abs().sum(-1)
            ri, ci = linear_sum_assignment(cost.detach().cpu().numpy())
            for rr, cc in zip(ri, ci):
                d = (pred_ca[rr] - gt_ca[cc]).norm().item()
                ca_err_sum += d
                pred_aa_idx = pred_aa[b, rr].argmax(-1).item()
                gt_aa_idx = int(gt_aa[b, vi[cc]].clamp(0, 19).item())
                if pred_aa_idx == gt_aa_idx:
                    aa_correct += 1
                n_matched_total += 1

    n = max(n_batches, 1)
    nm = max(n_matched_total, 1)
    return {
        "loss_total": total_total / n,
        "loss_coord": total_coord / n,
        "loss_aa": total_aa / n,
        "mean_ca_err": ca_err_sum / nm,
        "aa_acc": aa_correct / nm,
        "n_matched": n_matched_total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="hotflow/configs/train_hotspot.yaml")
    parser.add_argument("--logdir", type=str, default="logs_hotspot")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--batch_size", type=int, default=None, help="Override config batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override config lr")
    parser.add_argument("--max_iters", type=int, default=None, help="Override config max_iters")
    parser.add_argument("--num_workers", type=int, default=None, help="Override config num_workers")
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.safe_load(f))
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.lr is not None:
        config.train.optimizer.lr = args.lr
    if args.max_iters is not None:
        config.train.max_iters = args.max_iters
    if args.num_workers is not None:
        config.train.num_workers = args.num_workers
    torch.manual_seed(config.train.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ts = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    tag = ("_" + args.tag) if args.tag else ""
    run_dir = Path(args.logdir) / f"train_hotspot_{ts}{tag}"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(yaml.safe_load(open(args.config)), f)
    print(f"[INFO] run_dir = {run_dir}")

    train_loader, val_loader = build_loaders(config)
    print(f"[INFO] train batches/epoch = {len(train_loader)}")
    if val_loader is not None:
        print(f"[INFO] val batches/epoch = {len(val_loader)}")

    model_type = config.model.get("type", "plain")
    model_kwargs = dict(
        d_model=config.model.d_model,
        n_encoder_layers=config.model.n_encoder_layers,
        n_decoder_layers=config.model.n_decoder_layers,
        n_heads=config.model.n_heads,
        K=config.hotspot.num_anchors,
        dropout=config.model.dropout,
    )
    if model_type == "se3":
        model = HotspotPredictorSE3(**model_kwargs).to(device)
    elif model_type == "plain":
        model = HotspotPredictor(**model_kwargs).to(device)
    elif model_type == "plain_v2":
        model = HotspotPredictorV2(**model_kwargs).to(device)
    else:
        raise ValueError(f"Unknown model.type: {model_type}")
    print(f"[INFO] model.type={model_type}, params = {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.optimizer.lr,
        betas=(config.train.optimizer.beta1, config.train.optimizer.beta2),
        weight_decay=config.train.optimizer.weight_decay,
    )

    start_iter = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
        start_iter = ck.get("iteration", 0)
        print(f"[INFO] resumed from {args.resume} @ iter {start_iter}")

    rot_aug = bool(config.train.get("rotation_aug", False))
    min_rec_len = int(config.train.get("min_rec_length", 0))
    best_metric = float("inf")
    best_ckpt_path = ckpt_dir / "best.pt"

    train_iter = iter(train_loader)
    t0 = time.time()
    for it in range(start_iter + 1, config.train.max_iters + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        model.train()
        mp = float(config.train.get("mask_prob", 0.0))
        rec_coords, rec_aa, rec_mask = extract_receptor(batch, device, mask_prob=mp)
        gt_coords, gt_aa, gt_mask = extract_gt_anchors(batch, device)

        # Filter samples with too-short receptors (drop entire batch entries)
        if min_rec_len > 0:
            keep = rec_mask.sum(dim=-1) >= min_rec_len
            if not keep.any():
                continue
            if not keep.all():
                rec_coords = rec_coords[keep]
                rec_aa = rec_aa[keep]
                rec_mask = rec_mask[keep]
                gt_coords = gt_coords[keep]
                gt_aa = gt_aa[keep]
                gt_mask = gt_mask[keep]

        # Random SO(3) rotation augmentation
        if rot_aug:
            B_ = rec_coords.shape[0]
            R_g = _random_rotation(B_, device)
            rec_coords = _rotate_coords_batched(rec_coords, R_g)
            gt_coords = _rotate_coords_batched(gt_coords, R_g)

        pred_coords, pred_aa = model(rec_coords, rec_aa, rec_mask)
        out = hungarian_match_and_loss(
            pred_coords, pred_aa, gt_coords, gt_aa, gt_mask,
            coord_weight=config.train.loss.coord_weight,
            aa_weight=config.train.loss.aa_weight,
            match_metric=config.train.loss.match_metric,
        )
        loss = out["total"]

        optimizer.zero_grad()
        loss.backward()
        if config.train.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        if it % config.train.log_freq == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(
                f"[iter {it:6d}] total={float(loss):.3f} "
                f"coord={float(out['coord']):.3f} aa={float(out['aa']):.3f} "
                f"n_matched={out['n_matched']:4d}/{out['n_valid']:4d} "
                f"({config.train.log_freq}it / {dt:.1f}s)"
            )

        if val_loader is not None and it % config.train.val_freq == 0:
            ev = evaluate(model, val_loader, device, max_batches=50)
            print(
                f"[val   {it:6d}] loss_total={ev['loss_total']:.3f} "
                f"coord={ev['loss_coord']:.3f} aa={ev['loss_aa']:.3f} "
                f"mean_CA_err={ev['mean_ca_err']:.3f} A "
                f"aa_acc={ev['aa_acc']*100:.1f}% "
                f"(n_matched={ev['n_matched']})"
            )
            # Save best-by-CA-error checkpoint
            metric = ev["mean_ca_err"]
            if metric < best_metric:
                best_metric = metric
                torch.save({
                    "iteration": it,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_metric": metric,
                }, best_ckpt_path)
                print(f"[INFO] new best mean_CA_err={metric:.3f} A → {best_ckpt_path}")

        if it % config.train.ckpt_freq == 0:
            p = ckpt_dir / f"{it:06d}.pt"
            torch.save({
                "iteration": it,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, p)
            print(f"[INFO] saved {p}")


if __name__ == "__main__":
    main()
