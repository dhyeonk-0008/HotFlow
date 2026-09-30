"""Unified evaluation script: PepFlow vs Approach A vs Approach B.

Runs the same test set through up to three models and produces per-model
metrics CSVs plus a side-by-side comparison table.

Models:
  1. PepFlow   — vanilla FlowModel (no hotspot conditioning)
  2. Approach A — FlowModel + inpainting with GT hotspot anchors
  3. Approach B — FlowModelB with cross-attention hotspot conditioning

Usage:
    # Compare all three
    python hotflow/eval_compare.py \
        --config hotflow/configs/train_b.yaml \
        --pepflow_ckpt PepFlowww/model2.pt \
        --approach_b_ckpt logs_b/.../checkpoints/280000.pt \
        --outdir results/compare \
        --device cuda:0 \
        --num_samples 100

    # Compare PepFlow vs Approach B only (skip Approach A)
    python hotflow/eval_compare.py \
        --config hotflow/configs/train_b.yaml \
        --pepflow_ckpt PepFlowww/model2.pt \
        --approach_b_ckpt logs_b/.../checkpoints/280000.pt \
        --skip_approach_a \
        --outdir results/compare \
        --device cuda:0

    # Approach B with multiple guidance scales
    python hotflow/eval_compare.py \
        --config hotflow/configs/train_b.yaml \
        --pepflow_ckpt PepFlowww/model2.pt \
        --approach_b_ckpt logs_b/.../checkpoints/280000.pt \
        --outdir results/compare \
        --guidance_scales 1.0 1.5 2.0

    # With Rosetta FastRelax scoring (slow; requires pyrosetta).
    # GT is scored once per sample and shared across all methods so that
    # rosetta_*_delta columns are directly comparable.
    python hotflow/eval_compare.py ... \
        --rosetta --rosetta_iterations 2 --rosetta_score_gt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
PEPFLOW_ROOT = REPO_ROOT / "PepFlowww"
_script_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _script_dir]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PEPFLOW_ROOT))

from pepflow.utils.data import PaddingCollate, DEFAULT_PAD_VALUES, DEFAULT_NO_PADDING
from pepflow.utils.misc import load_config, seed_all
from pepflow.utils.train import recursive_to
from models_con.flow_model import FlowModel
from models_con.pep_dataloader import PepDataset
from models_con.utils import process_dic
from data import residue_constants

from hotflow.models.flow_model_b import FlowModelB
from hotflow.data.dataset_b import PepDatasetB, HOTSPOT_PAD_VALUES, HOTSPOT_NO_PADDING
from hotflow.data.hotspot_labeling import label_hotspots_from_batch, select_top_k_hotspots
from hotflow.sampling.inpainting import anchors_to_condition, prepare_inpainting_batch
from hotflow.data_types import HotspotAnchor

# Reuse metrics and PDB helpers from eval_b
from hotflow.eval_b import (
    compute_ca_rmsd,
    compute_tm_score,
    compute_aar,
    compute_anchor_contact_rate,
    compute_anchor_min_dist,
    compute_interface_rmsd,
    compute_binding_site_overlap,
    extract_ca_coords,
    save_generated_pdb,
    save_gt_pdb,
    idx_to_seq,
    IDX_TO_AA,
    _detect_peptide_chain,
)

# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_pepflow(config, ckpt_path, device):
    """Load vanilla PepFlow FlowModel."""
    model = FlowModel(config.model).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(process_dic(state_dict))
    model.eval()
    iteration = ckpt.get("iteration", "pretrained")
    print(f"[PepFlow] Loaded: {ckpt_path} (iter {iteration})")
    return model, iteration


def build_pephar_sampler(args, device):
    """Load PepHAR's AnchorBasedSamplerDenovo (density + prediction nets).

    Returns the sampler instance; raises if PepHAR weights are unreachable.
    """
    PEPHAR_ROOT = REPO_ROOT / "PepHAR"
    if str(PEPHAR_ROOT) not in sys.path:
        sys.path.insert(0, str(PEPHAR_ROOT))
    from evaluate.sample_revised import AnchorBasedSamplerDenovo

    class _Args:
        pass

    pephar_args = _Args()
    pephar_args.density_config_path = str(REPO_ROOT / args.pephar_density_config)
    pephar_args.density_param_path = str(REPO_ROOT / args.pephar_density_weights)
    pephar_args.prediction_config_path = str(REPO_ROOT / args.pephar_prediction_config)
    pephar_args.prediction_param_path = str(REPO_ROOT / args.pephar_prediction_weights)
    pephar_args.device = str(device)

    sampler = AnchorBasedSamplerDenovo(pephar_args)
    print(f"[PepHAR] Loaded density + prediction models")
    return sampler


def build_detr_sampler(ckpt_path: str, model_type: str, K: int, device):
    """Load DETR-style HotspotPredictor from a checkpoint.

    Returns the model in eval mode.
    """
    from hotflow.models.hotspot_predictor import HotspotPredictor
    from hotflow.models.hotspot_predictor_v2 import HotspotPredictorV2
    from hotflow.models.hotspot_predictor_se3 import HotspotPredictorSE3

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    model_cfg = cfg.get("model", {}) if cfg else {}

    kwargs = dict(
        d_model=model_cfg.get("d_model", 128),
        n_encoder_layers=model_cfg.get("n_encoder_layers", 4),
        n_decoder_layers=model_cfg.get("n_decoder_layers", 2),
        n_heads=model_cfg.get("n_heads", 4),
        K=K,
        dropout=0.0,
    )
    if model_type == "se3":
        model = HotspotPredictorSE3(**kwargs)
    elif model_type == "plain_v2":
        model = HotspotPredictorV2(**kwargs)
    else:
        model = HotspotPredictor(**kwargs)

    model.load_state_dict(ck["model"])
    model = model.to(device).eval()
    iteration = ck.get("iteration", "?")
    print(f"[DETR] Loaded {model_type} from {ckpt_path} (iter {iteration})")
    return model


def _build_detr_anchors_from_batch(
    batch, detr_model, device, num_anchors: int = 5,
):
    """Run DETR hotspot predictor on a batch receptor to predict K anchors.

    Returns (anchor_coords (1,K,3,3) [CA,C,N], anchor_types (1,K), anchor_mask (1,K))
    on CPU.
    """
    from pepflow.modules.protein.constants import BBHeavyAtom

    gen_mask = batch["generate_mask"][0].cpu().bool()
    valid = batch["res_mask"][0].cpu().bool() if "res_mask" in batch else torch.ones_like(gen_mask)
    rec_idx = (~gen_mask) & valid

    pos = batch["pos_heavyatom"][0].cpu()   # (L, A, 3)
    aa = batch["aa"][0].cpu()               # (L,)

    N_i = int(BBHeavyAtom.N); CA_i = int(BBHeavyAtom.CA); C_i = int(BBHeavyAtom.C)
    rec_coords = torch.stack(
        [pos[rec_idx, N_i], pos[rec_idx, CA_i], pos[rec_idx, C_i]], dim=-2,
    ).unsqueeze(0)  # (1, R, 3, 3) [N, CA, C]
    rec_aa = aa[rec_idx].clamp(0, 20).unsqueeze(0)       # (1, R)
    rec_mask = torch.ones(1, rec_idx.sum(), dtype=torch.bool)  # (1, R)

    rec_coords = rec_coords.to(device)
    rec_aa = rec_aa.to(device)
    rec_mask = rec_mask.to(device)

    with torch.no_grad():
        pred_coords, pred_aa_logits = detr_model(rec_coords, rec_aa, rec_mask)
    # pred_coords: (1, K, 3, 3) [N=0, CA=1, C=2]

    # Convert [N, CA, C] → [CA, C, N] (FlowModelB/HotspotEncoder convention)
    anchor_coords = pred_coords[..., [1, 2, 0], :].cpu()   # (1, K, 3, 3)
    anchor_types = pred_aa_logits.argmax(dim=-1).cpu()      # (1, K)
    anchor_mask = torch.ones(1, num_anchors, dtype=torch.bool)

    return anchor_coords, anchor_types, anchor_mask


def _split_batch_to_pephar(batch, device):
    """Convert one eval_compare batch (PepDatasetB collated) into PepHAR
    input format + a receptor dict suitable for save_pephar_pdb.

    Returns:
        pephar_data: dict(rec_coord, rec_aa, pep_coord, pep_aa) on `device`
        receptor: dict(pos_heavyatom, mask_heavyatom, aa) on cpu — receptor
            slice from the batch, for PDB writing.
    """
    from pepflow.modules.protein.constants import BBHeavyAtom

    gen_mask = batch["generate_mask"][0].bool().cpu()
    if "res_mask" in batch:
        valid = batch["res_mask"][0].bool().cpu()
    else:
        valid = torch.ones_like(gen_mask)

    pep_idx = gen_mask & valid
    rec_idx = (~gen_mask) & valid

    pos_heavyatom = batch["pos_heavyatom"][0].cpu()
    mask_heavyatom = batch["mask_heavyatom"][0].cpu()
    aa = batch["aa"][0].cpu()

    bb_slots = [BBHeavyAtom.CA, BBHeavyAtom.C, BBHeavyAtom.N]
    pephar_data = {
        "rec_coord": pos_heavyatom[rec_idx][:, bb_slots],
        "rec_aa": aa[rec_idx],
        "pep_coord": pos_heavyatom[pep_idx][:, bb_slots],
        "pep_aa": aa[pep_idx],
    }
    pephar_data = recursive_to(pephar_data, device)

    receptor = {
        "pos_heavyatom": pos_heavyatom[rec_idx],
        "mask_heavyatom": mask_heavyatom[rec_idx],
        "aa": aa[rec_idx],
    }
    return pephar_data, receptor


def _detect_receptor_chain(batch, batch_idx, pep_chain):
    """Find the dominant receptor chain ID in the batch (anything that's
    not the peptide chain). Falls back to 'A'."""
    gen_mask = batch["generate_mask"][batch_idx].bool().cpu()
    chain_ids = _extract_chain_id_safe(batch, batch_idx, gen_mask)
    rec_chains = []
    for i, is_pep in enumerate(gen_mask.tolist()):
        if not is_pep and chain_ids[i] not in rec_chains:
            rec_chains.append(chain_ids[i])
    if not rec_chains:
        return "A"
    return rec_chains[0]


def _extract_chain_id_safe(batch, batch_idx, gen_mask):
    """Same as eval_b._extract_chain_id but local copy to avoid name shadow."""
    if "chain_id" not in batch:
        return ["A" if not g else "B" for g in gen_mask.tolist()]
    raw = batch["chain_id"]
    try:
        per_sample = [list(item) for item in zip(*raw)]
        return per_sample[batch_idx]
    except Exception:
        return ["A" if not g else "B" for g in gen_mask.tolist()]


def sample_pephar(sampler, batch, device, anchor_steps=100, finetune_steps=100):
    """Run PepHAR on a single eval_compare batch.

    NOT wrapped in @torch.no_grad — PepHAR's anchor sampling does gradient
    descent on the EBM energy (`obj.backward()` inside `_sample_anchor`), so
    the autograd graph must be enabled. PepHAR handles its own no_grad
    contexts internally where applicable.

    Returns (gen, receptor_dict). `gen` has `pep_coord`(L,3,3 — CA/C/N) and
    `pep_aa`(L,). `receptor_dict` is the receptor slice for PDB writing.
    """
    pephar_data, receptor = _split_batch_to_pephar(batch, device)
    gen, _ = sampler.sample(
        pephar_data,
        anchor_steps=anchor_steps,
        finetune_steps=finetune_steps,
        anchor_strategy="ebm",
        extend_strategy="sto",
        anchor_nums=1,
    )
    return gen, receptor


def _compute_pephar_metrics_skeleton(batch, gen):
    """Tensor-based metrics for PepHAR output. When --rosetta is on, these
    are later overwritten by post-relax values from compute_metrics_from_pdb
    (which sees the full FastRelax-completed structure). When --no_rosetta,
    these are what ends up in the CSV — so we compute the basics directly
    from the backbone tensors PepHAR returns.
    """
    import numpy as np
    from pepflow.modules.protein.constants import BBHeavyAtom

    gen_mask = batch["generate_mask"][0].bool().cpu()
    pep_len = int(gen_mask.sum())

    # GT backbone CA from the batch
    pos_heavyatom = batch["pos_heavyatom"][0].cpu()
    aa_gt_full = batch["aa"][0].cpu()
    gt_pep_ca = pos_heavyatom[gen_mask, BBHeavyAtom.CA].numpy()  # (Lp, 3)
    gt_aa = aa_gt_full[gen_mask].numpy()
    # Receptor CA for interface metrics
    rec_mask = ~gen_mask
    if "res_mask" in batch:
        rec_mask = rec_mask & batch["res_mask"][0].bool().cpu()
    rec_ca = pos_heavyatom[rec_mask, BBHeavyAtom.CA].numpy()

    # PepHAR generated peptide
    pred_ca_t = gen["pep_coord"][:, 0].detach().cpu()  # (Lp, 3) — CA index 0
    pred_ca = pred_ca_t.numpy()
    pred_aa = gen["pep_aa"].detach().cpu().numpy()

    if len(pred_ca) >= 3 and len(gt_pep_ca) >= 3 and len(pred_ca) == len(gt_pep_ca):
        rmsd_unaligned, rmsd_aligned = compute_ca_rmsd(pred_ca, gt_pep_ca)
        try:
            tm_score = compute_tm_score(pred_ca, gt_pep_ca)
        except Exception:
            tm_score = float("nan")
        irmsd = compute_interface_rmsd(pred_ca, gt_pep_ca, rec_ca, interface_cutoff=8.0)
        bs_overlap = compute_binding_site_overlap(pred_ca, gt_pep_ca, rec_ca,
                                                   contact_cutoff=10.0)
    else:
        rmsd_unaligned = rmsd_aligned = tm_score = float("nan")
        irmsd = bs_overlap = float("nan")

    if len(pred_aa) == len(gt_aa) and len(pred_aa) > 0:
        aar = float((pred_aa == gt_aa).mean())
    else:
        aar = float("nan")

    # Anchor contacts (require batch's anchor fields)
    anchor_contact = {"anchor_contact_4A": float("nan"),
                      "anchor_contact_6A": float("nan"),
                      "anchor_contact_8A": float("nan"),
                      "anchor_min_dist": float("nan")}
    if "anchor_coords" in batch and "anchor_mask" in batch:
        anc_coords = batch["anchor_coords"][0].cpu().numpy()
        anc_mask = batch["anchor_mask"][0].cpu().numpy()
        try:
            anchor_contact = {
                "anchor_contact_4A": compute_anchor_contact_rate(
                    pred_ca, anc_coords, anc_mask, cutoff=4.0),
                "anchor_contact_6A": compute_anchor_contact_rate(
                    pred_ca, anc_coords, anc_mask, cutoff=6.0),
                "anchor_contact_8A": compute_anchor_contact_rate(
                    pred_ca, anc_coords, anc_mask, cutoff=8.0),
                "anchor_min_dist": compute_anchor_min_dist(
                    pred_ca, anc_coords, anc_mask),
            }
        except Exception:
            pass

    return {
        "rmsd_unaligned": rmsd_unaligned,
        "rmsd_aligned": rmsd_aligned,
        "tm_score": tm_score,
        "aar": aar,
        **anchor_contact,
        "interface_rmsd": irmsd,
        "binding_site_overlap": bs_overlap,
        "pred_seq": idx_to_seq(torch.tensor(pred_aa)),
        "gt_seq": idx_to_seq(torch.tensor(gt_aa)),
        "pep_len": pep_len,
    }


def build_approach_b(config, ckpt_path, device):
    """Load FlowModelB (Approach B)."""
    hs_cfg = config.hotspot
    model = FlowModelB(
        config.model,
        d_hotspot=hs_cfg.d_hotspot,
        cross_attn_heads=hs_cfg.cross_attn_heads,
        cross_attn_blocks=tuple(hs_cfg.cross_attn_blocks),
        hotspot_dropout_p=0.0,
        num_hotspot_anchors=hs_cfg.num_anchors,
        contact_distance_cutoff=hs_cfg.contact_distance_cutoff,
        contact_loss_target=hs_cfg.contact_loss_target,
        contact_loss_margin=hs_cfg.contact_loss_margin,
        use_se3_invariant=getattr(hs_cfg, "use_se3_invariant", True),
        use_anchor_self_attn=getattr(hs_cfg, "use_anchor_self_attn", True),
        num_self_attn_layers=getattr(hs_cfg, "num_self_attn_layers", 1),
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    cleaned = {}
    for k, v in state_dict.items():
        key = k.replace("module.", "") if k.startswith("module.") else k
        cleaned[key] = v
    model.load_state_dict(cleaned, strict=True)
    model = model.to(device)
    model.eval()
    iteration = ckpt.get("iteration", "unknown")
    print(f"[Approach B] Loaded: {ckpt_path} (iter {iteration})")
    return model, iteration


# ---------------------------------------------------------------------------
# Data loader (shared across all models)
# ---------------------------------------------------------------------------

def build_eval_dataloader(config, num_workers=4):
    """Build DataLoader with hotspot anchor fields (needed for A and B)."""
    collate_pad = {**DEFAULT_PAD_VALUES, **HOTSPOT_PAD_VALUES}
    collate_nopad = set(DEFAULT_NO_PADDING) | HOTSPOT_NO_PADDING
    collate_fn = PaddingCollate(eight=False, pad_values=collate_pad, no_padding=collate_nopad)

    val_cfg = config.dataset.get("val", None)
    if val_cfg and os.path.exists(val_cfg.structure_dir):
        print(f"[Data] Using validation set: {val_cfg.structure_dir}")
        base_ds = PepDataset(
            structure_dir=val_cfg.structure_dir,
            dataset_dir=val_cfg.dataset_dir,
            name=val_cfg.name,
            transform=None,
            reset=False,
        )
    else:
        print("[Data] Validation set not found, using train set.")
        train_cfg = config.dataset.train
        base_ds = PepDataset(
            structure_dir=train_cfg.structure_dir,
            dataset_dir=train_cfg.dataset_dir,
            name=train_cfg.name,
            transform=None,
            reset=False,
        )

    dataset = PepDatasetB(
        base_ds,
        num_anchors=config.hotspot.num_anchors,
        distance_cutoff=config.hotspot.contact_distance_cutoff,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return loader, dataset


# ---------------------------------------------------------------------------
# Per-model sampling functions
# ---------------------------------------------------------------------------

def _compute_metrics(batch, final, batch_idx=0, device="cpu"):
    """Compute all metrics from a sampling result. Shared across models."""
    pred_pep_ca, gt_pep_ca, rec_ca, gen_mask_np = extract_ca_coords(
        batch, final, batch_idx=batch_idx, device=device,
    )

    # Structural
    if len(pred_pep_ca) >= 3 and len(gt_pep_ca) >= 3:
        rmsd_unaligned, rmsd_aligned = compute_ca_rmsd(pred_pep_ca, gt_pep_ca)
        tm_score = compute_tm_score(pred_pep_ca, gt_pep_ca)
    else:
        rmsd_unaligned = rmsd_aligned = tm_score = float("nan")

    # Sequence
    gen_mask_t = batch["generate_mask"][0].bool().cpu()
    pred_aa = final["seqs"][0].cpu()
    gt_aa = batch["aa"][0].cpu()
    aar = compute_aar(pred_aa, gt_aa, gen_mask_t)
    pred_seq = idx_to_seq(pred_aa[gen_mask_t])
    gt_seq = idx_to_seq(gt_aa[gen_mask_t])

    # Binding (anchor-based)
    anchor_coords = batch["anchor_coords"][0].cpu().numpy()
    anchor_mask = batch["anchor_mask"][0].cpu().numpy()

    anchor_contact_4 = compute_anchor_contact_rate(pred_pep_ca, anchor_coords, anchor_mask, cutoff=4.0)
    anchor_contact_6 = compute_anchor_contact_rate(pred_pep_ca, anchor_coords, anchor_mask, cutoff=6.0)
    anchor_contact_8 = compute_anchor_contact_rate(pred_pep_ca, anchor_coords, anchor_mask, cutoff=8.0)
    anchor_min_dist = compute_anchor_min_dist(pred_pep_ca, anchor_coords, anchor_mask)

    irmsd = compute_interface_rmsd(pred_pep_ca, gt_pep_ca, rec_ca, interface_cutoff=8.0)
    bs_overlap = compute_binding_site_overlap(pred_pep_ca, gt_pep_ca, rec_ca, contact_cutoff=10.0)

    return {
        "rmsd_unaligned": rmsd_unaligned,
        "rmsd_aligned": rmsd_aligned,
        "tm_score": tm_score,
        "aar": aar,
        "anchor_contact_4A": anchor_contact_4,
        "anchor_contact_6A": anchor_contact_6,
        "anchor_contact_8A": anchor_contact_8,
        "anchor_min_dist": anchor_min_dist,
        "interface_rmsd": irmsd,
        "binding_site_overlap": bs_overlap,
        "pred_seq": pred_seq,
        "gt_seq": gt_seq,
        "pep_len": int(gen_mask_t.sum()),
    }, final


@torch.no_grad()
def sample_pepflow(model, batch, device, num_steps=100):
    """Sample with vanilla PepFlow (no hotspot conditioning)."""
    batch_dev = recursive_to(batch, device)
    traj = model.sample(
        batch_dev,
        num_steps=num_steps,
        sample_bb=True,
        sample_ang=True,
        sample_seq=True,
    )
    final = traj[-1]
    return _compute_metrics(batch, final, device=device)


def _build_pephar_anchors_from_batch(batch, pephar_sampler, num_anchors=5,
                                       anchor_steps=100):
    """Run PepHAR EBM on a batch's receptor to predict K hotspot anchors.

    Per Proposal §3.1, inference-time anchors should come from PepHAR's
    pretrained EBM density model — NOT from GT contacts. This helper
    extracts the receptor slice from a PepDatasetB batch and delegates
    to `build_denovo_anchors` (which wraps PepHAR's `sample_denovo`).

    Returns (anchor_coords (1,K,3,3), anchor_types (1,K), anchor_mask (1,K))
    on CPU. Caller is responsible for moving to device.
    """
    if pephar_sampler is None:
        raise ValueError("pephar_sampler required for PepHAR-anchor inference")

    from hotflow.benchmark_targets import build_denovo_anchors

    gen_mask = batch["generate_mask"][0].cpu().bool()
    if "res_mask" in batch:
        valid = batch["res_mask"][0].cpu().bool()
    else:
        valid = torch.ones_like(gen_mask)
    rec_idx = (~gen_mask) & valid

    receptor = {
        "pos_heavyatom": batch["pos_heavyatom"][0].cpu()[rec_idx],
        "aa": batch["aa"][0].cpu()[rec_idx],
        "mask_heavyatom": batch["mask_heavyatom"][0].cpu()[rec_idx],
    }
    return build_denovo_anchors(
        receptor,
        num_anchors=num_anchors,
        pephar_sampler=pephar_sampler,
        anchor_steps=anchor_steps,
    )


def _build_pephar_gt_seeded_anchors_from_batch(
    batch, pephar_sampler, num_anchors=5, anchor_steps=100,
    contact_cutoff=4.0,
):
    """Build A/B anchors via PepHAR baseline's GT-seeded EBM optimization.

    Mirrors `AnchorBasedSamplerDenovo.sample(anchor_strategy='ebm')`
    (`sample_revised.py:541-558`):

        1. Pick top-K peptide residues by GT contact frequency
           (same selection as training-time hotspot labeling — top-K
           by # heavy-atom pairs within `contact_cutoff` of receptor).
        2. `_rand_anchors(coord, aa)` → Gaussian noise on coord
           (`x_sig=1.0`, `o_sig=0.2`); residue type is *re-drawn at random*.
        3. `_sample_anchor(...)` → `anchor_steps` of Adam(lr=3e-2) on the
           EBM density log-prob plus per-step Langevin perturbation
           (`noise_eps=1e-2`).

    Differs from `_build_pephar_anchors_from_batch` (which uses
    `sample_denovo` and inits from receptor surface — true de novo). This
    path *peeks at GT* for the seed positions, so it is NOT proposal-§3.1
    compliant for true de novo evaluation; it exists to mirror the PepHAR
    baseline's internal anchor init for apples-to-apples comparison.

    Not wrapped in `@torch.no_grad` — `_sample_anchor` calls `.backward()`
    on the EBM objective, so autograd must be enabled.

    Returns: (anchor_coords (1,K,3,3) [CA,C,N], anchor_types (1,K),
              anchor_mask (1,K)) on CPU.
    """
    from hotflow.data.hotspot_labeling import (
        label_hotspots_from_batch, select_top_k_hotspots,
    )

    if pephar_sampler is None:
        raise ValueError(
            "pephar_sampler required for GT-seeded EBM anchor inference"
        )

    pephar_data, _ = _split_batch_to_pephar(batch, pephar_sampler.device)

    hotspot_labels = label_hotspots_from_batch(
        batch, distance_cutoff=contact_cutoff,
    )
    _, gt_anchor_coords, gt_anchor_types, gt_anchor_mask = select_top_k_hotspots(
        hotspot_labels, batch, k=num_anchors, distance_cutoff=contact_cutoff,
    )

    K = num_anchors
    out_coords = torch.zeros(1, K, 3, 3)
    out_types = torch.full((1, K), 20, dtype=torch.long)
    out_mask = torch.zeros(1, K, dtype=torch.bool)

    for j in range(K):
        if not bool(gt_anchor_mask[0, j]):
            continue
        coord = gt_anchor_coords[0, j].to(pephar_sampler.device)
        aa = gt_anchor_types[0, j].to(pephar_sampler.device)
        coord, aa = pephar_sampler._rand_anchors(coord, aa)
        if anchor_steps > 0:
            coord, aa = pephar_sampler._sample_anchor(
                pephar_data, coord, aa, n_steps=anchor_steps,
            )
        out_coords[0, j] = coord.detach().cpu()
        out_types[0, j] = int(aa.detach().cpu().item())
        out_mask[0, j] = True

    return out_coords, out_types, out_mask


@torch.no_grad()
def sample_approach_a(model, batch, device, num_steps=100, num_anchors=5,
                      contact_cutoff=4.0,
                      pephar_anchor_coords=None, pephar_anchor_types=None,
                      pephar_anchor_mask=None):
    """Sample with Approach A: hotspot anchors → inpainting.

    Per Proposal §3.1, anchors at inference come from PepHAR EBM
    (passed in via `pephar_anchor_*` tensors). The K anchors are
    distributed evenly across the target peptide indices and inpainted
    (those positions are frozen, rest is generated).

    Falls back to GT contact-based anchors when `pephar_anchor_coords`
    is None (ablation / smoke-test path).
    """
    batch_dev = recursive_to(batch, device)
    gen_mask = batch_dev["generate_mask"][0].bool()
    pep_indices = gen_mask.nonzero(as_tuple=True)[0]
    pep_length = len(pep_indices)

    use_pephar = pephar_anchor_coords is not None

    if use_pephar:
        a_coords = pephar_anchor_coords.to(device)
        a_types = pephar_anchor_types.to(device)
        a_mask = pephar_anchor_mask.to(device)

        # Distribute the K anchors evenly across the target peptide length.
        anchor_positions = torch.linspace(0, pep_length - 1, num_anchors).long()
        anchors = []
        for j in range(num_anchors):
            if not bool(a_mask[0, j]):
                continue
            anchors.append(HotspotAnchor(
                residue_index=int(anchor_positions[j].item()),
                backbone_coords=a_coords[0, j].cpu(),
                residue_type=int(a_types[0, j].item()),
                source="pephar",
            ))
    else:
        # Fallback: GT contact (only for ablation / when PepHAR is skipped)
        hotspot_labels = label_hotspots_from_batch(
            batch_dev, distance_cutoff=contact_cutoff,
        )
        _, anchor_coords_t, anchor_types_t, anchor_mask_t = select_top_k_hotspots(
            hotspot_labels, batch_dev, k=num_anchors,
            distance_cutoff=contact_cutoff,
        )
        anchors = []
        for j in range(num_anchors):
            if not anchor_mask_t[0, j]:
                continue
            g_idx = int(torch.argmin(
                torch.sum((batch_dev["pos_heavyatom"][0, pep_indices, 1, :] -
                            anchor_coords_t[0, j, 0, :].unsqueeze(0)) ** 2, dim=-1)
            ))
            anchors.append(HotspotAnchor(
                residue_index=g_idx,
                backbone_coords=anchor_coords_t[0, j].cpu(),
                residue_type=int(anchor_types_t[0, j].item()),
                source="gt_contact",
            ))

    if not anchors:
        # No anchors — fall back to vanilla sampling
        traj = model.sample(batch_dev, num_steps=num_steps,
                            sample_bb=True, sample_ang=True, sample_seq=True)
        final = traj[-1]
        return _compute_metrics(batch, final, device=device)

    condition = anchors_to_condition(batch_dev, anchors)
    condition.freeze_backbone = True
    condition.freeze_sequence = True
    condition.freeze_torsions = True
    prepared = prepare_inpainting_batch(batch_dev, condition)

    traj = model.sample(
        prepared,
        num_steps=num_steps,
        sample_bb=True,
        sample_ang=True,
        sample_seq=True,
    )
    final = traj[-1]
    return _compute_metrics(batch, final, device=device)


@torch.no_grad()
def sample_approach_b(model, batch, device, num_steps=100, guidance_scale=1.0,
                      pephar_anchor_coords=None, pephar_anchor_types=None,
                      pephar_anchor_mask=None):
    """Sample with Approach B (cross-attention conditioning).

    Per Proposal §3.1, anchors at inference come from PepHAR EBM
    (passed in via `pephar_anchor_*` tensors). When provided, those
    override the batch's GT-derived anchors.

    Falls back to the batch's GT-derived anchors (via FlowModelB's
    internal `_encode_hotspots_from_batch`) when `pephar_anchor_coords`
    is None.
    """
    batch_dev = recursive_to(batch, device)
    kwargs = dict(
        num_steps=num_steps,
        sample_bb=True,
        sample_ang=True,
        sample_seq=True,
        guidance_scale=guidance_scale,
    )
    if pephar_anchor_coords is not None:
        kwargs["anchor_coords"] = pephar_anchor_coords.to(device)
        kwargs["anchor_types"] = pephar_anchor_types.to(device)
        kwargs["anchor_mask"] = pephar_anchor_mask.to(device)

    traj = model.sample(batch_dev, **kwargs)
    final = traj[-1]
    return _compute_metrics(batch, final, device=device)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

METRIC_COLS = [
    "rmsd_unaligned", "rmsd_aligned", "tm_score", "aar",
    "anchor_contact_4A", "anchor_contact_6A", "anchor_contact_8A",
    "anchor_min_dist", "interface_rmsd", "binding_site_overlap",
]

ROSETTA_COLS = ["rosetta_stab", "rosetta_bind"]
ROSETTA_GT_COLS = [
    "rosetta_stab_gt", "rosetta_bind_gt",
    "rosetta_stab_delta", "rosetta_bind_delta",
]


def _metric_cols_for(args):
    """Return the list of numeric metric columns for the current run."""
    cols = list(METRIC_COLS)
    if args.rosetta:
        cols += ROSETTA_COLS
        if args.rosetta_score_gt:
            cols += ROSETTA_GT_COLS
    return cols


def _add_rosetta_metrics(metrics, gen_pdb_path, gt_pdb_path,
                          pep_chain, gt_rosetta, args):
    """FastRelax the generated PDB, then recompute every structural /
    sequence / interface / clash metric from the **relaxed** PDB.

    Mutates `metrics` in place with:
      - the Rosetta energy columns (stab, bind, optional gt + delta)
      - post-relax overrides of rmsd_*/tm_score/aar/interface_rmsd/
        binding_site_overlap (these replace the tensor-based values
        produced by _compute_metrics on the raw generation).

    A dumped `<gen_pdb>_relaxed.pdb` is written next to the raw generation
    so that case-study selection can later inspect the relaxed structure.
    `gt_rosetta` is computed once per sample and reused across methods.
    """
    from hotflow.utils.rosetta_score import fast_relax_score
    from hotflow.utils.pdb_metrics import compute_metrics_from_pdb

    metrics["rosetta_stab"] = float("nan")
    metrics["rosetta_bind"] = float("nan")
    if args.rosetta_score_gt:
        metrics["rosetta_stab_gt"] = float("nan")
        metrics["rosetta_bind_gt"] = float("nan")
        metrics["rosetta_stab_delta"] = float("nan")
        metrics["rosetta_bind_delta"] = float("nan")

    if not gen_pdb_path.exists():
        return
    if pep_chain is None:
        return

    relaxed_pdb_path = gen_pdb_path.with_name(
        gen_pdb_path.stem + "_relaxed.pdb"
    )
    gen_score = fast_relax_score(
        str(gen_pdb_path),
        peptide_chain=pep_chain,
        num_iterations=args.rosetta_iterations,
        out_pdb=str(relaxed_pdb_path),
    )
    if gen_score["success"]:
        metrics["rosetta_stab"] = gen_score["stab"]
        metrics["rosetta_bind"] = gen_score["bind"]

        try:
            pdb_metrics = compute_metrics_from_pdb(
                gen_pdb=str(relaxed_pdb_path),
                gt_pdb=str(gt_pdb_path) if gt_pdb_path and Path(gt_pdb_path).exists() else None,
                peptide_chain=pep_chain,
            )
            # Only overwrite the structural / sequence / interface / clash
            # columns. Anchor-contact metrics (computed from the model's
            # anchor inputs, not from a PDB) stay as set by _compute_metrics.
            for k in ("rmsd_unaligned", "rmsd_aligned", "tm_score", "aar",
                      "interface_rmsd", "binding_site_overlap",
                      "valid", "mean_ca_dist", "max_ca_dist_val", "min_ca_dist_val",
                      "num_clashes", "worst_clash_dist", "clash_free",
                      "internal_clashes", "internal_worst_dist", "internal_clash_free",
                      "pred_seq", "gt_seq", "pep_len"):
                if k in pdb_metrics:
                    metrics[k] = pdb_metrics[k]
        except Exception as e:
            print(f"  [post-relax metric failed] {relaxed_pdb_path.name}: {e}")
    else:
        print(f"  [Rosetta gen failed] {gen_pdb_path.name}: {gen_score['error']}")

    if args.rosetta_score_gt and gt_rosetta is not None:
        if gt_rosetta["success"]:
            metrics["rosetta_stab_gt"] = gt_rosetta["stab"]
            metrics["rosetta_bind_gt"] = gt_rosetta["bind"]
            if gen_score["success"]:
                metrics["rosetta_stab_delta"] = gen_score["stab"] - gt_rosetta["stab"]
                metrics["rosetta_bind_delta"] = gen_score["bind"] - gt_rosetta["bind"]


def _summarize(df, label, args):
    """Print and return aggregate summary dict."""
    summary = {"method": label, "num_samples": len(df)}
    cols = _metric_cols_for(args)
    print(f"\n  === {label} (n={len(df)}) ===")
    print(f"  {'Metric':<30} {'Mean':>10} {'Std':>10} {'Median':>10}")
    print(f"  {'-' * 62}")
    for col in cols:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) > 0:
            m, s, med = float(vals.mean()), float(vals.std()), float(vals.median())
        else:
            m = s = med = float("nan")
        summary[f"{col}_mean"] = m
        summary[f"{col}_std"] = s
        summary[f"{col}_median"] = med
        print(f"  {col:<30} {m:>10.4f} {s:>10.4f} {med:>10.4f}")
    return summary


def run_comparison(args):
    config, _ = load_config(args.config)
    seed_all(args.seed)
    device = torch.device(args.device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Fail fast if --rosetta is requested but pyrosetta is not installed.
    if args.rosetta:
        from hotflow.utils.rosetta_score import init_pyrosetta, fast_relax_score  # noqa: F401
        init_pyrosetta(silent=True)
        print(f"[Eval] Rosetta FastRelax scoring enabled "
              f"(iterations={args.rosetta_iterations}, "
              f"score_gt={args.rosetta_score_gt})")

    # ------------------------------------------------------------------
    # Build models
    # ------------------------------------------------------------------
    models = {}

    if args.pepflow_ckpt:
        models["PepFlow"] = build_pepflow(config, args.pepflow_ckpt, device)

    if not args.skip_approach_a and args.pepflow_ckpt:
        # Approach A reuses the same PepFlow weights
        models["Approach_A"] = models["PepFlow"]  # same model, different sampling

    if args.approach_b_ckpt:
        models["Approach_B"] = build_approach_b(config, args.approach_b_ckpt, device)

    pephar_sampler = None
    if not args.skip_pephar:
        try:
            pephar_sampler = build_pephar_sampler(args, device)
            models["PepHAR"] = (pephar_sampler, "pretrained")
        except Exception as e:
            print(f"[PepHAR] Skipping (load failed): {e}")
            pephar_sampler = None

    detr_model = None
    if args.detr_ckpt:
        try:
            hs_cfg_tmp = load_config(args.config)[0].hotspot
            detr_model = build_detr_sampler(
                args.detr_ckpt, args.detr_model_type,
                K=hs_cfg_tmp.num_anchors, device=device,
            )
        except Exception as e:
            print(f"[DETR] Skipping (load failed): {e}")
            detr_model = None

    if not models:
        print("No models specified. Use --pepflow_ckpt and/or --approach_b_ckpt.")
        return

    # ------------------------------------------------------------------
    # Build dataloader
    # ------------------------------------------------------------------
    loader, dataset = build_eval_dataloader(config, num_workers=args.num_workers)
    num_samples = min(args.num_samples, len(dataset))
    print(f"\n[Eval] {num_samples} samples, methods: {list(models.keys())}")

    hs_cfg = config.hotspot
    guidance_scales = args.guidance_scales

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    all_results = {name: [] for name in models}
    # Approach B evaluated per guidance scale
    if "Approach_B" in models and len(guidance_scales) > 1:
        for gs in guidance_scales[1:]:
            tag = f"Approach_B_gs{gs:.1f}"
            all_results[tag] = []
    # PepHAR uses the same tag — already registered if loaded
    # (models["PepHAR"] check above)

    for i, batch in enumerate(tqdm(loader, total=num_samples, desc="Evaluating")):
        if i >= num_samples:
            break

        sample_id = batch.get("id", [f"sample_{i:04d}"])[0] if "id" in batch else f"sample_{i:04d}"

        # Save GT PDB once at the top of the iteration so it can be scored
        # by Rosetta exactly once and shared across all methods.
        gt_dir = outdir / "gt_pdbs"
        gt_dir.mkdir(parents=True, exist_ok=True)
        gt_pdb_path = gt_dir / f"{sample_id}.pdb"
        try:
            save_gt_pdb(batch, 0, gt_pdb_path)
        except Exception:
            pass

        # Detect peptide chain ID once (same for all methods on this sample)
        # and optionally score the GT complex once.
        try:
            pep_chain = _detect_peptide_chain(batch, 0)
        except Exception as e:
            print(f"  [pep chain detect failed] {sample_id}: {e}")
            pep_chain = None
        gt_rosetta = None
        if args.rosetta:
            from hotflow.utils.rosetta_score import fast_relax_score
            if args.rosetta_score_gt and pep_chain is not None and gt_pdb_path.exists():
                gt_relaxed_path = gt_dir / f"{sample_id}_relaxed.pdb"
                gt_rosetta = fast_relax_score(
                    str(gt_pdb_path),
                    peptide_chain=pep_chain,
                    num_iterations=args.rosetta_iterations,
                    out_pdb=str(gt_relaxed_path),
                )
                if not gt_rosetta["success"]:
                    print(f"  [Rosetta gt failed] {sample_id}: {gt_rosetta['error']}")

        def _save_and_score(tag, batch, final, metrics):
            """Save the generated PDB for `tag`. When --rosetta is on, also
            FastRelax it, dump the relaxed PDB, and overwrite every
            structural / sequence / interface / clash column with values
            computed from the relaxed structure (see _add_rosetta_metrics)."""
            pdb_dir = outdir / tag / "pdbs"
            pdb_dir.mkdir(parents=True, exist_ok=True)
            gen_pdb_path = pdb_dir / f"{sample_id}_gen.pdb"
            try:
                save_generated_pdb(batch, final, 0, gen_pdb_path)
            except Exception:
                pass
            if args.rosetta:
                _add_rosetta_metrics(
                    metrics, gen_pdb_path, gt_pdb_path,
                    pep_chain, gt_rosetta, args,
                )

        def _save_and_score_pephar(tag, batch, gen, receptor, metrics):
            """Same flow as _save_and_score but writes PepHAR-style PDB
            (backbone-only peptide). Chain IDs are aligned with the GT so
            the post-relax metric comparison shares one coordinate frame."""
            from hotflow.benchmark import save_pephar_pdb
            pdb_dir = outdir / tag / "pdbs"
            pdb_dir.mkdir(parents=True, exist_ok=True)
            gen_pdb_path = pdb_dir / f"{sample_id}_gen.pdb"
            pep_chain_local = pep_chain if pep_chain is not None else "B"
            rec_chain_local = _detect_receptor_chain(batch, 0, pep_chain_local)
            try:
                save_pephar_pdb(
                    gen, receptor, gen_pdb_path,
                    rec_chain=rec_chain_local,
                    pep_chain=pep_chain_local,
                )
            except Exception as e:
                print(f"  [PepHAR PDB save failed] {sample_id}: {e}")
                return
            if args.rosetta:
                _add_rosetta_metrics(
                    metrics, gen_pdb_path, gt_pdb_path,
                    pep_chain_local, gt_rosetta, args,
                )

        # --- PepFlow ---
        if "PepFlow" in models:
            try:
                model_pf, _ = models["PepFlow"]
                metrics, final = sample_pepflow(model_pf, batch, device, num_steps=args.num_steps)
                metrics["sample_id"] = sample_id
                metrics["method"] = "PepFlow"
                _save_and_score("PepFlow", batch, final, metrics)
                all_results["PepFlow"].append(metrics)
            except Exception as e:
                print(f"  [PepFlow SKIP] {sample_id}: {e}")

        # --- Anchor source for Approach A and B ---
        # Selected by `--ab_anchor_source`. DETR (`--detr_ckpt`), when
        # provided, takes priority over the PepHAR paths.
        #   pephar_denovo    (default): PepHAR EBM, receptor-surface init
        #   pephar_gt_seeded: PepHAR EBM, init from GT contact-based hotspots
        #                     (matches the PepHAR baseline's internal sample()
        #                      with anchor_strategy='ebm' — peeks at GT!)
        #   gt_contact       : GT contact freeze, no EBM. Leaves anchor tensors
        #                      None so sample_approach_a/b take their internal
        #                      GT fallback paths.
        ab_anchor_coords = ab_anchor_types = ab_anchor_mask = None
        need_ab_anchors = (
            "Approach_A" in models or "Approach_B" in models
        )
        if detr_model is not None and need_ab_anchors:
            try:
                ab_anchor_coords, ab_anchor_types, ab_anchor_mask = (
                    _build_detr_anchors_from_batch(
                        batch, detr_model, device,
                        num_anchors=hs_cfg.num_anchors,
                    )
                )
            except Exception as e:
                print(f"  [DETR anchor build failed] {sample_id}: {e} "
                      f"— falling back to GT contacts for A/B")
        elif (
            args.ab_anchor_source == "pephar_gt_seeded"
            and pephar_sampler is not None
            and need_ab_anchors
        ):
            try:
                ab_anchor_coords, ab_anchor_types, ab_anchor_mask = (
                    _build_pephar_gt_seeded_anchors_from_batch(
                        batch, pephar_sampler,
                        num_anchors=hs_cfg.num_anchors,
                        anchor_steps=args.pephar_anchor_steps,
                        contact_cutoff=hs_cfg.contact_distance_cutoff,
                    )
                )
            except Exception as e:
                print(f"  [PepHAR GT-seeded anchor build failed] "
                      f"{sample_id}: {e} — falling back to GT contacts for A/B")
        elif (
            args.ab_anchor_source == "pephar_denovo"
            and pephar_sampler is not None
            and need_ab_anchors
        ):
            try:
                ab_anchor_coords, ab_anchor_types, ab_anchor_mask = (
                    _build_pephar_anchors_from_batch(
                        batch, pephar_sampler,
                        num_anchors=hs_cfg.num_anchors,
                        anchor_steps=args.pephar_anchor_steps,
                    )
                )
            except Exception as e:
                print(f"  [PepHAR anchor build failed] {sample_id}: {e} "
                      f"— falling back to GT contacts for A/B")

        # --- Approach A ---
        if "Approach_A" in models:
            try:
                model_a, _ = models["Approach_A"]
                metrics, final = sample_approach_a(
                    model_a, batch, device,
                    num_steps=args.num_steps,
                    num_anchors=hs_cfg.num_anchors,
                    contact_cutoff=hs_cfg.contact_distance_cutoff,
                    pephar_anchor_coords=ab_anchor_coords,
                    pephar_anchor_types=ab_anchor_types,
                    pephar_anchor_mask=ab_anchor_mask,
                )
                metrics["sample_id"] = sample_id
                metrics["method"] = "Approach_A"
                _save_and_score("Approach_A", batch, final, metrics)
                all_results["Approach_A"].append(metrics)
            except Exception as e:
                print(f"  [Approach A SKIP] {sample_id}: {e}")

        # --- Approach B ---
        if "Approach_B" in models:
            model_b, _ = models["Approach_B"]
            for gs in guidance_scales:
                tag = "Approach_B" if gs == guidance_scales[0] else f"Approach_B_gs{gs:.1f}"
                try:
                    metrics, final = sample_approach_b(
                        model_b, batch, device,
                        num_steps=args.num_steps,
                        guidance_scale=gs,
                        pephar_anchor_coords=ab_anchor_coords,
                        pephar_anchor_types=ab_anchor_types,
                        pephar_anchor_mask=ab_anchor_mask,
                    )
                    metrics["sample_id"] = sample_id
                    metrics["method"] = tag
                    metrics["guidance_scale"] = gs
                    _save_and_score(tag, batch, final, metrics)
                    all_results[tag].append(metrics)
                except Exception as e:
                    print(f"  [{tag} SKIP] {sample_id}: {e}")

        # --- PepHAR ---
        if "PepHAR" in models:
            try:
                sampler, _ = models["PepHAR"]
                gen, receptor = sample_pephar(
                    sampler, batch, device,
                    anchor_steps=args.pephar_anchor_steps,
                    finetune_steps=args.pephar_finetune_steps,
                )
                metrics = _compute_pephar_metrics_skeleton(batch, gen)
                metrics["sample_id"] = sample_id
                metrics["method"] = "PepHAR"
                _save_and_score_pephar("PepHAR", batch, gen, receptor, metrics)
                all_results["PepHAR"].append(metrics)
            except Exception as e:
                print(f"  [PepHAR SKIP] {sample_id}: {e}")
                import traceback
                traceback.print_exc()

    # ------------------------------------------------------------------
    # Per-method CSV + summary
    # ------------------------------------------------------------------
    summaries = []
    for name, records in all_results.items():
        if not records:
            continue
        df = pd.DataFrame(records)
        csv_path = outdir / f"{name}_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  Saved {csv_path}")
        summary = _summarize(df, name, args)
        summaries.append(summary)

    # ------------------------------------------------------------------
    # Side-by-side comparison table
    # ------------------------------------------------------------------
    if len(summaries) >= 2:
        print(f"\n{'=' * 70}")
        print("  COMPARISON TABLE")
        print(f"{'=' * 70}")

        cols_for_compare = _metric_cols_for(args)
        methods = [s["method"] for s in summaries]
        header = f"  {'Metric':<28}" + "".join(f"{m:>16}" for m in methods)
        print(header)
        print(f"  {'-' * (28 + 16 * len(methods))}")

        for col in cols_for_compare:
            row = f"  {col:<28}"
            for s in summaries:
                val = s.get(f"{col}_mean", float("nan"))
                row += f"{val:>16.4f}"
            print(row)

        # Also save as CSV
        compare_rows = []
        for col in cols_for_compare:
            row_data = {"metric": col}
            for s in summaries:
                row_data[f"{s['method']}_mean"] = s.get(f"{col}_mean", float("nan"))
                row_data[f"{s['method']}_std"] = s.get(f"{col}_std", float("nan"))
            compare_rows.append(row_data)

        compare_df = pd.DataFrame(compare_rows)
        compare_path = outdir / "comparison.csv"
        compare_df.to_csv(compare_path, index=False)
        print(f"\n  Comparison saved to {compare_path}")

    # Save all summaries as JSON
    json_path = outdir / "summaries.json"
    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"  Summaries saved to {json_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare PepFlow vs Approach A vs Approach B",
    )
    parser.add_argument("--config", type=str, default="hotflow/configs/train_b.yaml")
    parser.add_argument("--pepflow_ckpt", type=str, default=None,
                        help="PepFlow pretrained checkpoint (used for PepFlow & Approach A)")
    parser.add_argument("--approach_b_ckpt", type=str, default=None,
                        help="Approach B (FlowModelB) trained checkpoint")
    parser.add_argument("--skip_approach_a", action="store_true",
                        help="Skip Approach A evaluation")
    parser.add_argument("--skip_pephar", action="store_true",
                        help="Skip PepHAR (default loads PepHAR if weights "
                             "are reachable)")
    parser.add_argument("--pephar_density_config", type=str,
                        default="PepHAR/ckpts/density_v4_x5o2_2024_09_08__11_25_36/density_v4_x5o2.yml")
    parser.add_argument("--pephar_density_weights", type=str,
                        default="PepHAR/ckpts/density_v4_x5o2_2024_09_08__11_25_36/checkpoints/1400.pt")
    parser.add_argument("--pephar_prediction_config", type=str,
                        default="PepHAR/ckpts/prediction_d2_x2o1_2024_09_08__11_21_33/prediction_d2_x2o1.yml")
    parser.add_argument("--pephar_prediction_weights", type=str,
                        default="PepHAR/ckpts/prediction_d2_x2o1_2024_09_08__11_21_33/checkpoints/2400.pt")
    parser.add_argument("--pephar_anchor_steps", type=int, default=100)
    parser.add_argument("--pephar_finetune_steps", type=int, default=100)
    parser.add_argument(
        "--ab_anchor_source", type=str, default="pephar_denovo",
        choices=["pephar_denovo", "pephar_gt_seeded", "gt_contact"],
        help=(
            "Anchor source for Approach A/B (DETR overrides this when "
            "--detr_ckpt is given). "
            "pephar_denovo (default): PepHAR sample_denovo() — receptor "
            "surface init + EBM. True de novo (proposal §3.1). "
            "pephar_gt_seeded: PepHAR _sample_anchor() — GT contact-based "
            "top-K seed + _rand_anchors noise + EBM. Mirrors the PepHAR "
            "baseline's internal sample(anchor_strategy='ebm') exactly; "
            "peeks at GT for seeding. "
            "gt_contact: no EBM — A/B fall back to their internal GT "
            "contact-freeze paths."
        ),
    )
    parser.add_argument("--detr_ckpt", type=str, default=None,
                        help="DETR hotspot predictor checkpoint (best.pt). "
                             "When provided, replaces PepHAR as anchor source for A/B.")
    parser.add_argument("--detr_model_type", type=str, default="plain_v2",
                        choices=["plain", "plain_v2", "se3"],
                        help="DETR model variant matching the checkpoint.")
    parser.add_argument("--outdir", type=str, default="results/compare")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--guidance_scales", type=float, nargs="+", default=[1.0],
                        help="Guidance scales for Approach B (first is default)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rosetta", dest="rosetta", action="store_true",
                        default=True,
                        help="Run Rosetta FastRelax on each generation and "
                             "compute every metric from the relaxed PDB. "
                             "Default ON; requires pyrosetta.")
    parser.add_argument("--no_rosetta", dest="rosetta", action="store_false",
                        help="Skip Rosetta; report tensor-based metrics from "
                             "the raw generation (smoke-test fallback).")
    parser.add_argument("--rosetta_iterations", type=int, default=2,
                        help="Number of FastRelax + score iterations to mean "
                             "over. PepHAR uses 2, PepFlow uses 5. Default 2.")
    parser.add_argument("--rosetta_score_gt", action="store_true",
                        help="Also score the ground-truth complex with Rosetta "
                             "(once per sample, shared across methods) and "
                             "report (gen - gt) deltas.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_all(args.seed)
    run_comparison(args)
