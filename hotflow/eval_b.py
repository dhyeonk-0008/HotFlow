"""Evaluation script for FlowModelB (Approach B).

Loads a trained checkpoint, runs sampling on a test set, and computes:
  Phase 1 — Structural: CA-RMSD (unaligned / aligned), TM-score
  Phase 2 — Sequence:   Amino Acid Recovery (AAR)
  Phase 3 — Binding:    Contact Preservation Rate, Interface RMSD,
                         Binding-Site Overlap
  Phase 4 — Energy (optional, --rosetta):
            Rosetta FastRelax + InterfaceAnalyzerMover applied to each saved
            PDB. Reports mean full-atom REU (stab) and mean dG_separated
            (bind) over `--rosetta_iterations` repeats. This mirrors the
            post-generation scoring protocol used in PepFlow (5 iter) and
            PepHAR (2 iter).

Outputs:
  - Per-sample PDB files  (generated + ground-truth)
  - Per-sample CSV with all metrics
  - Aggregate summary printed to stdout / saved as JSON

Usage:
    # No Rosetta (fast)
    python hotflow/eval_b.py \
        --config  hotflow/configs/train_b.yaml \
        --ckpt    logs_b/train_b_.../checkpoints/280000.pt \
        --outdir  results/eval_280k \
        --device  cuda:0 \
        --num_samples 100 \
        --guidance_scales 1.0 1.5 2.0

    # With Rosetta scoring (slow; requires pyrosetta)
    python hotflow/eval_b.py ... --rosetta --rosetta_iterations 2 --rosetta_score_gt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
PEPFLOW_ROOT = REPO_ROOT / "PepFlowww"
_script_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _script_dir]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PEPFLOW_ROOT))

from pepflow.utils.data import PaddingCollate, DEFAULT_PAD_VALUES, DEFAULT_NO_PADDING
from pepflow.utils.misc import load_config, seed_all
from pepflow.utils.train import recursive_to
from pepflow.modules.common.geometry import reconstruct_backbone
from pepflow.modules.protein.writers import save_pdb
from models_con.pep_dataloader import PepDataset
from models_con.torsion import full_atom_reconstruction, get_heavyatom_mask
from data import residue_constants

from hotflow.models.flow_model_b import FlowModelB
from hotflow.data.dataset_b import PepDatasetB, HOTSPOT_PAD_VALUES, HOTSPOT_NO_PADDING

# ---------------------------------------------------------------------------
# Residue index -> 1-letter code
# ---------------------------------------------------------------------------
IDX_TO_AA = residue_constants.restypes_with_x  # len=21, index 20 = 'X'


def idx_to_seq(aa_indices: torch.Tensor) -> str:
    """Convert (L,) int tensor of residue indices to 1-letter string."""
    return "".join(IDX_TO_AA[min(i, 20)] for i in aa_indices.tolist())


# ===========================================================================
# Metrics
# ===========================================================================

# --- Structural -----------------------------------------------------------

def compute_ca_rmsd(pos_pred: np.ndarray, pos_gt: np.ndarray):
    """CA-RMSD without alignment (unaligned) and with Kabsch alignment.

    Args:
        pos_pred, pos_gt: (L, 3) CA coordinates.

    Returns:
        (rmsd_unaligned, rmsd_aligned)
    """
    assert pos_pred.shape == pos_gt.shape
    # Unaligned
    diff = pos_pred - pos_gt
    rmsd_unaligned = np.sqrt(np.mean(np.sum(diff ** 2, axis=-1)))

    # Kabsch alignment
    p = pos_pred - pos_pred.mean(axis=0)
    q = pos_gt - pos_gt.mean(axis=0)
    H = p.T @ q
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, d])
    R = Vt.T @ sign_matrix @ U.T
    p_aligned = p @ R.T
    diff_aligned = p_aligned - q
    rmsd_aligned = np.sqrt(np.mean(np.sum(diff_aligned ** 2, axis=-1)))
    return float(rmsd_unaligned), float(rmsd_aligned)


def compute_tm_score(pos_pred: np.ndarray, pos_gt: np.ndarray):
    """TM-score using tmtools (if available), else NaN."""
    try:
        import tmtools
        L = len(pos_gt)
        seq_dummy = "A" * L
        result = tmtools.tm_align(pos_pred, pos_gt, seq_dummy, seq_dummy)
        return float(result.tm_norm_chain2)
    except ImportError:
        return float("nan")


# --- Sequence -------------------------------------------------------------

def compute_aar(pred_aa: torch.Tensor, gt_aa: torch.Tensor, mask: torch.Tensor):
    """Amino Acid Recovery: fraction of matching residues in masked region.

    Args:
        pred_aa, gt_aa: (L,) integer tensors.
        mask: (L,) boolean, True for residues to compare (peptide region).

    Returns:
        AAR as float in [0, 1].
    """
    mask = mask.bool()
    if mask.sum() == 0:
        return float("nan")
    match = (pred_aa[mask] == gt_aa[mask]).float().mean()
    return float(match)


# --- Binding quality ------------------------------------------------------

def compute_anchor_contact_rate(
    pred_pep_ca: np.ndarray,
    anchor_coords: np.ndarray,
    anchor_mask: np.ndarray,
    cutoff: float = 6.0,
):
    """Fraction of hotspot anchors (receptor side) contacted by any generated
    peptide residue.

    For each valid anchor, check if ANY predicted peptide CA is within cutoff.
    This measures whether the model successfully placed the peptide near the
    receptor hotspots, regardless of which peptide residue makes the contact.

    Args:
        pred_pep_ca: (P, 3) predicted peptide CA positions.
        anchor_coords: (K, 3, 3) anchor backbone coords (CA at index 0).
        anchor_mask: (K,) boolean validity mask.
        cutoff: distance cutoff in Angstroms.

    Returns:
        contact_rate in [0, 1] or NaN if no valid anchors.
    """
    valid = anchor_mask.astype(bool)
    if not valid.any() or len(pred_pep_ca) == 0:
        return float("nan")

    anchor_ca = anchor_coords[valid, 0, :]  # (K', 3) — CA is index 0
    # Min distance from each anchor CA to any peptide CA
    diff = anchor_ca[:, None, :] - pred_pep_ca[None, :, :]  # (K', P, 3)
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))  # (K', P)
    min_dist = dist.min(axis=1)  # (K',)
    return float((min_dist <= cutoff).mean())


def compute_anchor_min_dist(
    pred_pep_ca: np.ndarray,
    anchor_coords: np.ndarray,
    anchor_mask: np.ndarray,
):
    """Mean minimum distance from each hotspot anchor to the nearest generated
    peptide CA. Lower is better.

    Returns:
        mean_min_dist in Angstroms, or NaN.
    """
    valid = anchor_mask.astype(bool)
    if not valid.any() or len(pred_pep_ca) == 0:
        return float("nan")

    anchor_ca = anchor_coords[valid, 0, :]
    diff = anchor_ca[:, None, :] - pred_pep_ca[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    min_dist = dist.min(axis=1)
    return float(min_dist.mean())


def compute_interface_rmsd(
    pred_ca: np.ndarray,
    gt_ca: np.ndarray,
    rec_ca: np.ndarray,
    interface_cutoff: float = 8.0,
):
    """RMSD of peptide residues at the binding interface.

    Interface residues = peptide residues whose GT CA is within
    interface_cutoff of any receptor CA.

    Returns:
        iRMSD (aligned) or NaN if no interface residues.
    """
    if len(rec_ca) == 0:
        return float("nan")

    # Identify interface residues from GT
    diff = gt_ca[:, None, :] - rec_ca[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    min_dist = dist.min(axis=1)
    interface_mask = min_dist <= interface_cutoff

    if interface_mask.sum() == 0:
        return float("nan")

    pred_intf = pred_ca[interface_mask]
    gt_intf = gt_ca[interface_mask]
    _, rmsd_aligned = compute_ca_rmsd(pred_intf, gt_intf)
    return float(rmsd_aligned)


def compute_binding_site_overlap(
    pred_ca: np.ndarray,
    gt_ca: np.ndarray,
    rec_ca: np.ndarray,
    contact_cutoff: float = 10.0,
):
    """Overlap of receptor residues contacted by predicted vs GT peptide.

    A receptor residue is 'contacted' if any peptide CA is within
    contact_cutoff. Returns Jaccard similarity of the two contact sets.
    """
    if len(rec_ca) == 0:
        return float("nan")

    def _contact_set(pep_ca):
        diff = pep_ca[:, None, :] - rec_ca[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        min_per_rec = dist.min(axis=0)
        return set(np.where(min_per_rec <= contact_cutoff)[0].tolist())

    set_pred = _contact_set(pred_ca)
    set_gt = _contact_set(gt_ca)
    if len(set_pred | set_gt) == 0:
        return float("nan")
    return float(len(set_pred & set_gt) / len(set_pred | set_gt))


# ===========================================================================
# Batch field extraction helpers
# ===========================================================================

def _extract_chain_id(batch, batch_idx, gen_mask):
    """Extract per-residue chain_id list from a collated batch.

    default_collate turns a list-of-str field into a length-L tuple of
    length-B tuples, e.g. (('A','A'), ('A','A'), ('B','B'), ...).
    We unpack by transposing with zip(*...) then picking batch_idx.
    """
    if "chain_id" not in batch:
        return ["A" if g else "B" for g in gen_mask.tolist()]

    raw = batch["chain_id"]
    try:
        # Typical collated form: tuple/list of tuples, length = L
        # Each inner tuple has length B (one entry per batch element)
        per_sample = [list(item) for item in zip(*raw)]  # B lists of length L
        return per_sample[batch_idx]
    except Exception:
        # Fallback
        return ["A" if g else "B" for g in gen_mask.tolist()]


def _extract_icode(batch, batch_idx, num_res):
    """Extract icode list, handling collated tuple-of-tuples."""
    if "icode" not in batch:
        return [" "] * num_res

    raw = batch["icode"]
    try:
        per_sample = [list(item) for item in zip(*raw)]
        return per_sample[batch_idx]
    except Exception:
        return [" "] * num_res


def _detect_peptide_chain(batch, batch_idx):
    """Return the chain ID of the (generated) peptide region for this sample.

    Falls back to 'A' if no chain_id field is available, matching the fallback
    in `_extract_chain_id` and `save_generated_pdb`.
    """
    gen_mask = batch["generate_mask"][batch_idx].bool().cpu()
    chain_ids = _extract_chain_id(batch, batch_idx, gen_mask)
    pep_chains = []
    for i, is_pep in enumerate(gen_mask.tolist()):
        if is_pep and chain_ids[i] not in pep_chains:
            pep_chains.append(chain_ids[i])
    if not pep_chains:
        return "A"
    return pep_chains[0]


# ===========================================================================
# PDB saving helpers
# ===========================================================================

def _deduplicate_resseq(chain_nb, resseq, icode):
    """Re-number residues per chain to avoid duplicate (resseq, icode) pairs
    that cause BioPython StructureBuilder errors."""
    new_resseq = resseq.clone()
    for ch in chain_nb.unique().tolist():
        mask = (chain_nb == ch)
        idx = mask.nonzero(as_tuple=True)[0]
        new_resseq[idx] = torch.arange(1, len(idx) + 1, dtype=resseq.dtype)
    new_icode = [" "] * len(resseq)
    return new_resseq, new_icode


def save_generated_pdb(batch, final_state, batch_idx, path, device="cpu"):
    """Save generated peptide + receptor as a single PDB file.

    Uses full_atom_reconstruction for the generated peptide region,
    and keeps original receptor coordinates.
    """
    # Unpack single item from batch
    B_idx = batch_idx

    rotmats = final_state["rotmats"][B_idx: B_idx + 1].to(device)
    trans = final_state["trans"][B_idx: B_idx + 1].to(device)
    angles = final_state["angles"][B_idx: B_idx + 1].to(device)
    seqs = final_state["seqs"][B_idx: B_idx + 1].to(device)

    # Full atom reconstruction for generated region
    pos14, _, _ = full_atom_reconstruction(R_bb=rotmats, t_bb=trans, angles=angles, aa=seqs)
    pos15 = F.pad(pos14, pad=(0, 0, 0, 15 - 14), value=0.0)[0].cpu()
    mask15 = get_heavyatom_mask(seqs)[0].cpu()

    gen_mask = batch["generate_mask"][B_idx].bool().cpu()
    pos_orig = batch["pos_heavyatom"][B_idx].cpu()
    mask_orig = batch["mask_heavyatom"][B_idx].cpu()
    aa_orig = batch["aa"][B_idx].cpu()

    pos_new = torch.where(gen_mask[:, None, None], pos15, pos_orig)
    mask_new = torch.where(gen_mask[:, None], mask15, mask_orig)
    aa_new = torch.where(gen_mask, seqs[0].cpu(), aa_orig)

    chain_nb = batch["chain_nb"][B_idx].cpu()
    chain_id = _extract_chain_id(batch, B_idx, gen_mask)
    resseq_orig = batch["resseq"][B_idx].cpu() if "resseq" in batch else torch.arange(len(aa_new))
    resseq, icode = _deduplicate_resseq(chain_nb, resseq_orig, None)

    data = {
        "chain_nb": chain_nb,
        "chain_id": chain_id,
        "resseq": resseq,
        "icode": icode,
        "aa": aa_new,
        "mask_heavyatom": mask_new,
        "pos_heavyatom": pos_new,
    }
    save_pdb(data, path=str(path))


def save_gt_pdb(batch, batch_idx, path):
    """Save ground-truth complex as PDB."""
    chain_nb = batch["chain_nb"][batch_idx].cpu()
    aa = batch["aa"][batch_idx].cpu()
    mask_ha = batch["mask_heavyatom"][batch_idx].cpu()
    pos_ha = batch["pos_heavyatom"][batch_idx].cpu()
    resseq = batch["resseq"][batch_idx].cpu() if "resseq" in batch else torch.arange(len(aa))

    gen_mask = batch["generate_mask"][batch_idx].bool().cpu()
    chain_id = _extract_chain_id(batch, batch_idx, gen_mask)
    resseq, icode = _deduplicate_resseq(chain_nb, resseq, None)

    data = {
        "chain_nb": chain_nb,
        "chain_id": chain_id,
        "resseq": resseq,
        "icode": icode,
        "aa": aa,
        "mask_heavyatom": mask_ha,
        "pos_heavyatom": pos_ha,
    }
    save_pdb(data, path=str(path))


# ===========================================================================
# Extract coordinates from sample trajectory
# ===========================================================================

def extract_ca_coords(batch, final_state, batch_idx, device="cpu"):
    """Extract predicted and GT CA coordinates for peptide and receptor.

    Returns:
        pred_pep_ca: (P, 3) np.ndarray — predicted peptide CA
        gt_pep_ca:   (P, 3) np.ndarray — ground-truth peptide CA
        rec_ca:      (R, 3) np.ndarray — receptor CA (fixed)
        gen_mask:    (L,) bool np.ndarray
    """
    gen_mask = batch["generate_mask"][batch_idx].bool().cpu()
    res_mask = batch["res_mask"][batch_idx].bool().cpu()

    CA_IDX = 1  # N=0, CA=1, C=2 in PepFlow atom ordering

    # GT CA
    gt_pos = batch["pos_heavyatom"][batch_idx].cpu()  # (L, A, 3)
    gt_ca_all = gt_pos[:, CA_IDX, :]  # (L, 3)

    # Predicted CA = trans from flow model (trans == CA position)
    pred_ca_all = final_state["trans"][batch_idx].cpu()  # (L, 3)

    pep_idx = (gen_mask & res_mask).numpy()
    rec_idx = (~gen_mask & res_mask).numpy()

    pred_pep_ca = pred_ca_all[pep_idx].numpy()
    gt_pep_ca = gt_ca_all[pep_idx].numpy()
    rec_ca = gt_ca_all[rec_idx].numpy()

    return pred_pep_ca, gt_pep_ca, rec_ca, gen_mask.numpy()


# ===========================================================================
# Main evaluation
# ===========================================================================

def build_model_from_ckpt(config, ckpt_path, device):
    """Build FlowModelB and load checkpoint weights."""
    hs_cfg = config.hotspot
    model = FlowModelB(
        config.model,
        d_hotspot=hs_cfg.d_hotspot,
        cross_attn_heads=hs_cfg.cross_attn_heads,
        cross_attn_blocks=tuple(hs_cfg.cross_attn_blocks),
        hotspot_dropout_p=0.0,  # no dropout at eval time
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
    # Handle DDP prefix
    cleaned = {}
    for k, v in state_dict.items():
        key = k.replace("module.", "") if k.startswith("module.") else k
        cleaned[key] = v
    model.load_state_dict(cleaned, strict=True)
    model = model.to(device)
    model.eval()

    iteration = ckpt.get("iteration", "unknown")
    print(f"[Eval] Loaded checkpoint: {ckpt_path} (iter {iteration})")
    return model, iteration


def build_eval_dataloader(config, num_workers=4):
    """Build evaluation DataLoader from the validation split."""
    collate_fn_pad = {**DEFAULT_PAD_VALUES, **HOTSPOT_PAD_VALUES}
    collate_fn_nopad = set(DEFAULT_NO_PADDING) | HOTSPOT_NO_PADDING
    collate_fn = PaddingCollate(eight=False, pad_values=collate_fn_pad, no_padding=collate_fn_nopad)

    # Try validation set first; fall back to train set
    val_cfg = config.dataset.get("val", None)
    if val_cfg and os.path.exists(val_cfg.structure_dir):
        print(f"[Eval] Using validation set: {val_cfg.structure_dir}")
        base_ds = PepDataset(
            structure_dir=val_cfg.structure_dir,
            dataset_dir=val_cfg.dataset_dir,
            name=val_cfg.name,
            transform=None,
            reset=False,
        )
    else:
        print("[Eval] Validation set not found, falling back to train set.")
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
        batch_size=1,  # per-sample evaluation
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return loader, dataset


def evaluate_single_sample(
    model,
    batch,
    device,
    guidance_scale=1.0,
    num_steps=100,
    contact_cutoff=4.0,
    interface_cutoff=8.0,
    binding_cutoff=10.0,
):
    """Run sampling and compute all metrics for a single batch (B=1).

    Returns dict of metrics.
    """
    batch_dev = recursive_to(batch, device)

    # Sample
    traj = model.sample(
        batch_dev,
        num_steps=num_steps,
        sample_bb=True,
        sample_ang=True,
        sample_seq=True,
        guidance_scale=guidance_scale,
    )
    final = traj[-1]  # last step, already on CPU

    # Extract coordinates
    pred_pep_ca, gt_pep_ca, rec_ca, gen_mask_np = extract_ca_coords(
        batch, final, batch_idx=0, device=device
    )

    # --- Structural metrics ---
    if len(pred_pep_ca) >= 3 and len(gt_pep_ca) >= 3:
        rmsd_unaligned, rmsd_aligned = compute_ca_rmsd(pred_pep_ca, gt_pep_ca)
        tm_score = compute_tm_score(pred_pep_ca, gt_pep_ca)
    else:
        rmsd_unaligned = rmsd_aligned = tm_score = float("nan")

    # --- Sequence metrics ---
    gen_mask_t = batch["generate_mask"][0].bool().cpu()
    pred_aa = final["seqs"][0].cpu()
    gt_aa = batch["aa"][0].cpu()
    aar = compute_aar(pred_aa, gt_aa, gen_mask_t)
    pred_seq = idx_to_seq(pred_aa[gen_mask_t])
    gt_seq = idx_to_seq(gt_aa[gen_mask_t])

    # --- Binding metrics ---
    # Anchor-based: how many receptor hotspot anchors are contacted by
    # any generated peptide residue?
    anchor_coords = batch["anchor_coords"][0].cpu().numpy()  # (K, 3, 3)
    anchor_mask = batch["anchor_mask"][0].cpu().numpy()       # (K,)

    anchor_contact_4 = compute_anchor_contact_rate(
        pred_pep_ca, anchor_coords, anchor_mask, cutoff=4.0
    )
    anchor_contact_6 = compute_anchor_contact_rate(
        pred_pep_ca, anchor_coords, anchor_mask, cutoff=6.0
    )
    anchor_contact_8 = compute_anchor_contact_rate(
        pred_pep_ca, anchor_coords, anchor_mask, cutoff=8.0
    )
    anchor_min_dist = compute_anchor_min_dist(
        pred_pep_ca, anchor_coords, anchor_mask
    )

    irmsd = compute_interface_rmsd(
        pred_pep_ca, gt_pep_ca, rec_ca, interface_cutoff=interface_cutoff
    )
    bs_overlap = compute_binding_site_overlap(
        pred_pep_ca, gt_pep_ca, rec_ca, contact_cutoff=binding_cutoff
    )

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


def run_evaluation(args):
    """Main evaluation loop."""
    config, _ = load_config(args.config)
    seed_all(config.train.seed)
    device = torch.device(args.device)

    # Fail fast if --rosetta is requested but pyrosetta is not installed.
    if args.rosetta:
        from hotflow.utils.rosetta_score import init_pyrosetta
        init_pyrosetta(silent=True)
        print(f"[Eval] Rosetta FastRelax scoring enabled "
              f"(iterations={args.rosetta_iterations}, "
              f"score_gt={args.rosetta_score_gt})")

    # Build model
    model, iteration = build_model_from_ckpt(config, args.ckpt, device)

    # Build dataloader
    loader, dataset = build_eval_dataloader(config, num_workers=args.num_workers)
    num_samples = min(args.num_samples, len(dataset))
    print(f"[Eval] Evaluating {num_samples} samples from {len(dataset)} total")

    guidance_scales = args.guidance_scales

    for gs in guidance_scales:
        gs_tag = f"gs{gs:.1f}"
        outdir = Path(args.outdir) / gs_tag
        pdb_dir = outdir / "pdbs"
        pdb_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Guidance scale: {gs}")
        print(f"  Output: {outdir}")
        print(f"{'='*60}")

        all_metrics = []

        for i, batch in enumerate(tqdm(loader, total=num_samples, desc=f"Sampling (gs={gs})")):
            if i >= num_samples:
                break

            sample_id = batch.get("id", [f"sample_{i:04d}"])[0] if "id" in batch else f"sample_{i:04d}"

            try:
                metrics, final = evaluate_single_sample(
                    model, batch, device,
                    guidance_scale=gs,
                    num_steps=args.num_steps,
                    contact_cutoff=config.hotspot.contact_distance_cutoff,
                )
            except Exception as e:
                print(f"  [SKIP] {sample_id}: {e}")
                continue

            metrics["sample_id"] = sample_id
            metrics["guidance_scale"] = gs
            all_metrics.append(metrics)

            # Save PDBs
            gen_pdb_path = pdb_dir / f"{sample_id}_gen.pdb"
            gt_pdb_path = pdb_dir / f"{sample_id}_gt.pdb"
            pdbs_saved = False
            try:
                save_generated_pdb(batch, final, 0, gen_pdb_path)
                save_gt_pdb(batch, 0, gt_pdb_path)
                pdbs_saved = True
            except Exception as e:
                print(f"  [PDB save failed] {sample_id}: {e}")

            # --- Rosetta FastRelax scoring (opt-in) ---
            # Mirrors PepFlow / PepHAR's `get_rosetta_score_base`: run FastRelax
            # and InterfaceAnalyzerMover for `rosetta_iterations` repeats and
            # report mean stability / dG_separated. Applied to the freshly
            # saved PDB so the structure passed to Rosetta matches what we
            # store on disk.
            if args.rosetta:
                metrics["rosetta_stab"] = float("nan")
                metrics["rosetta_bind"] = float("nan")
                if args.rosetta_score_gt:
                    metrics["rosetta_stab_gt"] = float("nan")
                    metrics["rosetta_bind_gt"] = float("nan")
                    metrics["rosetta_stab_delta"] = float("nan")
                    metrics["rosetta_bind_delta"] = float("nan")

                if pdbs_saved:
                    from hotflow.utils.rosetta_score import fast_relax_score
                    pep_chain = _detect_peptide_chain(batch, 0)
                    gen_score = fast_relax_score(
                        gen_pdb_path,
                        peptide_chain=pep_chain,
                        num_iterations=args.rosetta_iterations,
                    )
                    if gen_score["success"]:
                        metrics["rosetta_stab"] = gen_score["stab"]
                        metrics["rosetta_bind"] = gen_score["bind"]
                    else:
                        print(f"  [Rosetta gen failed] {sample_id}: {gen_score['error']}")

                    if args.rosetta_score_gt:
                        gt_score = fast_relax_score(
                            gt_pdb_path,
                            peptide_chain=pep_chain,
                            num_iterations=args.rosetta_iterations,
                        )
                        if gt_score["success"]:
                            metrics["rosetta_stab_gt"] = gt_score["stab"]
                            metrics["rosetta_bind_gt"] = gt_score["bind"]
                        else:
                            print(f"  [Rosetta gt failed] {sample_id}: {gt_score['error']}")
                        if gen_score["success"] and gt_score["success"]:
                            metrics["rosetta_stab_delta"] = gen_score["stab"] - gt_score["stab"]
                            metrics["rosetta_bind_delta"] = gen_score["bind"] - gt_score["bind"]

        if not all_metrics:
            print(f"  No valid samples for gs={gs}")
            continue

        # --- Save per-sample CSV ---
        df = pd.DataFrame(all_metrics)
        csv_path = outdir / "metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  Per-sample metrics saved to {csv_path}")

        # --- Aggregate summary ---
        numeric_cols = [
            "rmsd_unaligned", "rmsd_aligned", "tm_score", "aar",
            "anchor_contact_4A", "anchor_contact_6A", "anchor_contact_8A",
            "anchor_min_dist", "interface_rmsd", "binding_site_overlap",
        ]
        if args.rosetta:
            numeric_cols += ["rosetta_stab", "rosetta_bind"]
            if args.rosetta_score_gt:
                numeric_cols += [
                    "rosetta_stab_gt", "rosetta_bind_gt",
                    "rosetta_stab_delta", "rosetta_bind_delta",
                ]
        summary = {"guidance_scale": gs, "num_samples": len(df), "checkpoint_iter": iteration}
        for col in numeric_cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                summary[f"{col}_mean"] = float(vals.mean())
                summary[f"{col}_std"] = float(vals.std())
                summary[f"{col}_median"] = float(vals.median())
            else:
                summary[f"{col}_mean"] = float("nan")

        # Print summary
        print(f"\n  === Summary (gs={gs}, n={len(df)}) ===")
        print(f"  {'Metric':<30} {'Mean':>8} {'Std':>8} {'Median':>8}")
        print(f"  {'-'*56}")
        for col in numeric_cols:
            mean = summary.get(f"{col}_mean", float("nan"))
            std = summary.get(f"{col}_std", float("nan"))
            med = summary.get(f"{col}_median", float("nan"))
            print(f"  {col:<30} {mean:>8.4f} {std:>8.4f} {med:>8.4f}")

        # Save JSON
        json_path = outdir / "summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary saved to {json_path}")

    # --- Cross-guidance comparison ---
    if len(guidance_scales) > 1:
        print(f"\n{'='*60}")
        print("  Guidance Scale Comparison")
        print(f"{'='*60}")
        all_dfs = []
        for gs in guidance_scales:
            csv_p = Path(args.outdir) / f"gs{gs:.1f}" / "metrics.csv"
            if csv_p.exists():
                tmp = pd.read_csv(csv_p)
                all_dfs.append(tmp)
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            numeric_cols = [
                "rmsd_aligned", "tm_score", "aar",
                "anchor_contact_4A", "anchor_contact_6A", "anchor_contact_8A",
                "anchor_min_dist", "interface_rmsd", "binding_site_overlap",
            ]
            if args.rosetta:
                numeric_cols += ["rosetta_stab", "rosetta_bind"]
                if args.rosetta_score_gt:
                    numeric_cols += ["rosetta_stab_delta", "rosetta_bind_delta"]
            numeric_cols = [c for c in numeric_cols if c in combined.columns]
            pivot = combined.groupby("guidance_scale")[numeric_cols].mean()
            print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

            combined_path = Path(args.outdir) / "comparison.csv"
            pivot.to_csv(combined_path)
            print(f"\n  Comparison saved to {combined_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FlowModelB (Approach B)")
    parser.add_argument("--config", type=str, default="hotflow/configs/train_b.yaml",
                        help="Config YAML file")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Checkpoint path (.pt)")
    parser.add_argument("--outdir", type=str, default="results/eval_b",
                        help="Output directory for PDBs and metrics")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to evaluate")
    parser.add_argument("--num_steps", type=int, default=100,
                        help="Denoising steps for sampling")
    parser.add_argument("--guidance_scales", type=float, nargs="+", default=[1.0],
                        help="Classifier-free guidance scales to evaluate")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rosetta", action="store_true",
                        help="Run Rosetta FastRelax + InterfaceAnalyzer scoring "
                             "on each generated PDB (matches PepFlow/PepHAR "
                             "evaluation protocol). Requires pyrosetta.")
    parser.add_argument("--rosetta_iterations", type=int, default=2,
                        help="Number of FastRelax + score iterations to mean "
                             "over. PepHAR uses 2, PepFlow uses 5. Default 2.")
    parser.add_argument("--rosetta_score_gt", action="store_true",
                        help="Also run Rosetta scoring on the ground-truth "
                             "complex and report (gen - gt) deltas.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_all(args.seed)
    run_evaluation(args)
