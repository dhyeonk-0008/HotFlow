"""Unified peptide binder design benchmark.

Generates 50 peptide binders per target per method (4 methods x 5 targets),
saves PDB outputs, and evaluates with structural/sequence metrics.

Usage:
    python hotflow/benchmark.py \
        --targets 9CDZ 7UXO 6YVR 4Y5U 8TF5 \
        --methods PepFlow PepHAR Approach_A Approach_B \
        --device cuda:0 \
        --num_samples 50

    # Quick test with 1 sample
    python hotflow/benchmark.py \
        --targets 9CDZ \
        --methods PepFlow \
        --num_samples 1 \
        --device cpu
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
PEPHAR_ROOT = REPO_ROOT / "PepHAR"

_script_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _script_dir]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PEPFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(PEPFLOW_ROOT))
if str(PEPHAR_ROOT) not in sys.path:
    sys.path.insert(0, str(PEPHAR_ROOT))

from pepflow.utils.data import PaddingCollate  # noqa: E402
from pepflow.utils.misc import load_config, seed_all  # noqa: E402
from pepflow.utils.train import recursive_to  # noqa: E402
from pepflow.modules.protein.writers import save_pdb  # noqa: E402
from models_con.flow_model import FlowModel  # noqa: E402
from models_con.utils import process_dic  # noqa: E402
from models_con.torsion import full_atom_reconstruction, get_heavyatom_mask  # noqa: E402

from hotflow.models.flow_model_b import FlowModelB  # noqa: E402
from hotflow.data.hotspot_labeling import label_hotspots_from_batch, select_top_k_hotspots  # noqa: E402
from hotflow.sampling.inpainting import anchors_to_condition, prepare_inpainting_batch  # noqa: E402
from hotflow.data_types import HotspotAnchor  # noqa: E402

from hotflow.eval_b import (  # noqa: E402
    compute_ca_rmsd,
    compute_tm_score,
    compute_aar,
    compute_interface_rmsd,
    compute_binding_site_overlap,
    save_generated_pdb,
    save_gt_pdb,
    extract_ca_coords,
    idx_to_seq,
)

from hotflow.benchmark_targets import (  # noqa: E402
    TARGET_CONFIGS,
    prepare_target,
    build_pepflow_batch_gt,
    build_pepflow_batch_denovo,
    build_pephar_data_gt,
    build_pephar_data_denovo,
    build_denovo_anchors,
    get_denovo_pep_length,
)

from hotflow.utils.rosetta_score import fast_relax_score, init_pyrosetta  # noqa: E402
from hotflow.utils.pdb_metrics import compute_metrics_from_pdb  # noqa: E402


# ---------------------------------------------------------------------------
# Rosetta helpers
# ---------------------------------------------------------------------------

ROSETTA_NAN_METRICS = {
    "rosetta_stab": float("nan"),
    "rosetta_bind": float("nan"),
}

ROSETTA_GT_NAN_METRICS = {
    "rosetta_stab_gt": float("nan"),
    "rosetta_bind_gt": float("nan"),
    "rosetta_stab_delta": float("nan"),
    "rosetta_bind_delta": float("nan"),
}


def _relax_generated_pdb(
    raw_pdb_path: str | Path,
    relaxed_pdb_path: str | Path,
    args,
    peptide_chain: str = "B",
) -> dict:
    """FastRelax the generated complex and dump the relaxed pose.

    Returns the raw fast_relax_score dict (with `relaxed_pdb` filled in on
    success). The caller is responsible for translating this into Rosetta-
    score columns and for invoking compute_metrics_from_pdb on
    `relaxed_pdb_path` so that all downstream metrics see the post-relax
    structure.
    """
    return fast_relax_score(
        str(raw_pdb_path),
        peptide_chain=peptide_chain,
        num_iterations=args.rosetta_iterations,
        out_pdb=str(relaxed_pdb_path),
    )


def _relax_gt_once(
    gt_pdb_path: str | Path,
    gt_relaxed_path: str | Path,
    args,
    peptide_chain: str = "B",
) -> dict | None:
    """Relax the GT complex one time per target (cached on disk).

    Returns the fast_relax_score dict, or None if `--rosetta_score_gt` is
    disabled. If `gt_relaxed_path` already exists from a previous run we
    still re-relax to recover the score, but the dumped PDB is overwritten
    deterministically.
    """
    if not args.rosetta_score_gt or gt_pdb_path is None:
        return None
    if not Path(gt_pdb_path).exists():
        return None
    return fast_relax_score(
        str(gt_pdb_path),
        peptide_chain=peptide_chain,
        num_iterations=args.rosetta_iterations,
        out_pdb=str(gt_relaxed_path),
    )


def _rosetta_score_columns(
    gen_score: dict,
    gt_score: dict | None,
    args,
) -> dict:
    """Pack stab / bind / delta columns from raw fast_relax_score dicts."""
    cols = dict(ROSETTA_NAN_METRICS)
    if args.rosetta_score_gt:
        cols.update(ROSETTA_GT_NAN_METRICS)

    if gen_score is not None and gen_score.get("success"):
        cols["rosetta_stab"] = gen_score["stab"]
        cols["rosetta_bind"] = gen_score["bind"]

    if args.rosetta_score_gt and gt_score is not None and gt_score.get("success"):
        cols["rosetta_stab_gt"] = gt_score["stab"]
        cols["rosetta_bind_gt"] = gt_score["bind"]
        if gen_score is not None and gen_score.get("success"):
            cols["rosetta_stab_delta"] = gen_score["stab"] - gt_score["stab"]
            cols["rosetta_bind_delta"] = gen_score["bind"] - gt_score["bind"]

    return cols


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_pepflow(config, ckpt_path, device):
    """Load vanilla PepFlow FlowModel."""
    model = FlowModel(config.model).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(process_dic(state_dict))
    model.eval()
    print(f"[PepFlow] Loaded: {ckpt_path}")
    return model


def load_approach_b(config, ckpt_path, device):
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
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)
    model = model.to(device)
    model.eval()
    print(f"[Approach B] Loaded: {ckpt_path}")
    return model


def load_pephar_sampler(args, device):
    """Load PepHAR AnchorBasedSamplerDenovo."""
    from evaluate.sample_revised import AnchorBasedSamplerDenovo

    # Build an args-like object for the sampler
    class PepHARArgs:
        pass

    pephar_args = PepHARArgs()
    pephar_args.density_config_path = str(REPO_ROOT / args.pephar_density_config)
    pephar_args.density_param_path = str(REPO_ROOT / args.pephar_density_weights)
    pephar_args.prediction_config_path = str(REPO_ROOT / args.pephar_prediction_config)
    pephar_args.prediction_param_path = str(REPO_ROOT / args.pephar_prediction_weights)
    pephar_args.device = str(device)

    sampler = AnchorBasedSamplerDenovo(pephar_args)
    print(f"[PepHAR] Loaded density + prediction models")
    return sampler


# ---------------------------------------------------------------------------
# Sampling functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_pepflow(model, batch, device, num_steps=200):
    """Sample with vanilla PepFlow."""
    batch_dev = recursive_to(batch, device)
    traj = model.sample(
        batch_dev, num_steps=num_steps,
        sample_bb=True, sample_ang=True, sample_seq=True,
    )
    return traj[-1]


@torch.no_grad()
def sample_approach_a_gt(model, batch, device, num_steps=100, num_anchors=5,
                         contact_cutoff=4.0):
    """Sample with Approach A using GT hotspot anchors → inpainting."""
    batch_dev = recursive_to(batch, device)

    # Label hotspots from GT contacts
    hotspot_labels = label_hotspots_from_batch(batch_dev, distance_cutoff=contact_cutoff)
    _, anchor_coords_t, anchor_types_t, anchor_mask_t = select_top_k_hotspots(
        hotspot_labels, batch_dev, k=num_anchors, distance_cutoff=contact_cutoff,
    )

    # Convert to HotspotAnchor list
    gen_mask = batch_dev["generate_mask"][0].bool()
    pep_indices = gen_mask.nonzero(as_tuple=True)[0]
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
        traj = model.sample(batch_dev, num_steps=num_steps,
                            sample_bb=True, sample_ang=True, sample_seq=True)
        return traj[-1]

    condition = anchors_to_condition(batch_dev, anchors)
    condition.freeze_backbone = True
    condition.freeze_sequence = True
    condition.freeze_torsions = True
    prepared = prepare_inpainting_batch(batch_dev, condition)

    traj = model.sample(prepared, num_steps=num_steps,
                        sample_bb=True, sample_ang=True, sample_seq=True)
    return traj[-1]


@torch.no_grad()
def sample_approach_b(model, batch, device, num_steps=100, guidance_scale=1.0,
                      anchor_coords=None, anchor_types=None, anchor_mask=None):
    """Sample with Approach B (cross-attention conditioning)."""
    batch_dev = recursive_to(batch, device)
    kwargs = dict(
        num_steps=num_steps,
        sample_bb=True, sample_ang=True, sample_seq=True,
        guidance_scale=guidance_scale,
    )
    if anchor_coords is not None:
        kwargs["anchor_coords"] = anchor_coords.to(device)
        kwargs["anchor_types"] = anchor_types.to(device)
        kwargs["anchor_mask"] = anchor_mask.to(device)
    traj = model.sample(batch_dev, **kwargs)
    return traj[-1]


def sample_pephar_gt(sampler, pephar_data, device, anchor_steps=100,
                     finetune_steps=100, anchor_nums=1):
    """Sample with PepHAR using GT peptide."""
    from utils.train import recursive_to as pephar_recursive_to

    data = pephar_recursive_to(pephar_data, device)
    gen, metrics = sampler.sample(
        data, anchor_steps=anchor_steps, finetune_steps=finetune_steps,
        anchor_strategy='ebm', extend_strategy='sto',
        anchor_nums=anchor_nums,
    )
    return gen, metrics


def sample_pephar_denovo(sampler, rec_coord, rec_aa, pep_length, device,
                         anchor_steps=100, finetune_steps=100, anchor_nums=1):
    """Sample with PepHAR de novo (no GT peptide)."""
    gen, metrics = sampler.sample_denovo(
        rec_coord=rec_coord.to(device),
        rec_aa=rec_aa.to(device),
        pep_length=pep_length,
        anchor_steps=anchor_steps,
        finetune_steps=finetune_steps,
        anchor_nums=anchor_nums,
    )
    return gen, metrics


# ---------------------------------------------------------------------------
# PDB saving for PepHAR results
# ---------------------------------------------------------------------------

def save_pephar_pdb(
    gen: dict,
    receptor: dict,
    path: str | Path,
    rec_chain: str = "A",
    pep_chain: str = "B",
) -> None:
    """Save PepHAR-generated peptide + receptor as PDB.

    Chain IDs default to A=receptor / B=peptide (the convention used by
    benchmark.py). When called from eval_compare.py, pass the GT-detected
    chain IDs so the saved PDB matches the GT for relax + metric comparison.
    """
    from pepflow.modules.protein.constants import BBHeavyAtom, max_num_heavyatoms
    from pepflow.modules.common.geometry import construct_3d_basis

    pep_coord = gen["pep_coord"]  # (L, 3, 3): CA, C, N
    pep_aa = gen["pep_aa"]  # (L,)
    pep_len = pep_aa.shape[0]

    # Build minimal pos_heavyatom for peptide (backbone only)
    pep_pos = torch.zeros(pep_len, max_num_heavyatoms, 3)
    pep_mask = torch.zeros(pep_len, max_num_heavyatoms, dtype=torch.bool)
    pep_pos[:, BBHeavyAtom.CA] = pep_coord[:, 0]
    pep_pos[:, BBHeavyAtom.C] = pep_coord[:, 1]
    pep_pos[:, BBHeavyAtom.N] = pep_coord[:, 2]
    pep_mask[:, BBHeavyAtom.CA] = True
    pep_mask[:, BBHeavyAtom.C] = True
    pep_mask[:, BBHeavyAtom.N] = True

    rec_n = receptor["aa"].shape[0]
    rec_pos = receptor["pos_heavyatom"]
    rec_mask = receptor["mask_heavyatom"]
    rec_aa = receptor["aa"]

    total_n = rec_n + pep_len
    data = {
        "chain_nb": torch.cat([torch.zeros(rec_n, dtype=torch.long),
                               torch.ones(pep_len, dtype=torch.long)]),
        "chain_id": [rec_chain] * rec_n + [pep_chain] * pep_len,
        "resseq": torch.arange(1, total_n + 1, dtype=torch.long),
        "icode": [" "] * total_n,
        "aa": torch.cat([rec_aa, pep_aa.long()]),
        "mask_heavyatom": torch.cat([rec_mask, pep_mask]),
        "pos_heavyatom": torch.cat([rec_pos, pep_pos]),
    }
    save_pdb(data, path=str(path))


# ---------------------------------------------------------------------------
# Validity & steric clash checks
# ---------------------------------------------------------------------------

def check_validity(pep_ca: np.ndarray, max_ca_dist: float = 4.5) -> dict:
    """Check backbone validity from CA coordinates.

    Args:
        pep_ca: (L, 3) peptide CA coordinates.
        max_ca_dist: maximum allowed consecutive CA distance.

    Returns:
        dict with 'valid', 'mean_ca_dist', 'max_ca_dist', 'min_ca_dist'.
    """
    if len(pep_ca) < 2:
        return {"valid": False, "mean_ca_dist": float("nan"),
                "max_ca_dist_val": float("nan"), "min_ca_dist_val": float("nan")}

    ca_dists = np.linalg.norm(pep_ca[:-1] - pep_ca[1:], axis=-1)
    return {
        "valid": bool(np.all(ca_dists < max_ca_dist)),
        "mean_ca_dist": float(np.mean(ca_dists)),
        "max_ca_dist_val": float(np.max(ca_dists)),
        "min_ca_dist_val": float(np.min(ca_dists)),
    }


def check_steric_clash(
    pep_pos: np.ndarray,
    pep_mask: np.ndarray,
    rec_pos: np.ndarray,
    rec_mask: np.ndarray,
    clash_cutoff: float = 2.0,
) -> dict:
    """Check steric clashes between peptide and receptor heavy atoms.

    Args:
        pep_pos: (P, A, 3) peptide heavy atom positions.
        pep_mask: (P, A) boolean mask for valid atoms.
        rec_pos: (R, A, 3) receptor heavy atom positions.
        rec_mask: (R, A) boolean mask for valid atoms.
        clash_cutoff: minimum allowed distance (Angstroms).

    Returns:
        dict with 'num_clashes', 'worst_clash_dist', 'clash_free'.
    """
    # Flatten to valid atoms only
    pep_atoms = pep_pos[pep_mask]  # (M, 3)
    rec_atoms = rec_pos[rec_mask]  # (N, 3)

    if len(pep_atoms) == 0 or len(rec_atoms) == 0:
        return {"num_clashes": 0, "worst_clash_dist": float("nan"), "clash_free": True}

    # Pairwise distances
    diff = pep_atoms[:, None, :] - rec_atoms[None, :, :]  # (M, N, 3)
    dists = np.linalg.norm(diff, axis=-1)  # (M, N)

    clashes = dists < clash_cutoff
    num_clashes = int(clashes.sum())
    worst_dist = float(dists.min()) if dists.size > 0 else float("nan")

    return {
        "num_clashes": num_clashes,
        "worst_clash_dist": worst_dist,
        "clash_free": num_clashes == 0,
    }


def check_internal_clash(
    pep_pos: np.ndarray,
    pep_mask: np.ndarray,
    clash_cutoff: float = 1.5,
    bond_sep: int = 2,
) -> dict:
    """Check intra-molecular steric clashes within the peptide.

    Atoms on the same residue or on residues within ``bond_sep`` positions
    are excluded (they are bonded neighbours).

    Args:
        pep_pos: (P, A, 3) peptide heavy atom positions.
        pep_mask: (P, A) boolean mask for valid atoms.
        clash_cutoff: minimum allowed distance (Angstroms).
        bond_sep: residue separation below which atom pairs are skipped.

    Returns:
        dict with 'internal_clashes', 'internal_worst_dist',
        'internal_clash_free'.
    """
    P, A = pep_mask.shape

    # Build flat arrays with residue index tracking
    coords_list = []
    res_idx_list = []
    for r in range(P):
        for a in range(A):
            if pep_mask[r, a]:
                coords_list.append(pep_pos[r, a])
                res_idx_list.append(r)

    if len(coords_list) < 2:
        return {"internal_clashes": 0, "internal_worst_dist": float("nan"),
                "internal_clash_free": True}

    coords = np.array(coords_list)  # (M, 3)
    res_ids = np.array(res_idx_list)  # (M,)

    # Pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]  # (M, M, 3)
    dists = np.linalg.norm(diff, axis=-1)  # (M, M)

    # Mask: exclude self, same residue, and bonded neighbours
    res_sep = np.abs(res_ids[:, None] - res_ids[None, :])
    exclude = res_sep <= bond_sep
    np.fill_diagonal(exclude, True)

    # Find clashes among non-bonded pairs
    clash_mask = (dists < clash_cutoff) & ~exclude
    # Count unique pairs (upper triangle only)
    num_clashes = int(clash_mask[np.triu_indices_from(clash_mask, k=1)].sum())

    valid_dists = dists.copy()
    valid_dists[exclude] = float("inf")
    worst_dist = float(valid_dists.min()) if valid_dists.size > 0 else float("nan")

    return {
        "internal_clashes": num_clashes,
        "internal_worst_dist": worst_dist if num_clashes > 0 else float("nan"),
        "internal_clash_free": num_clashes == 0,
    }


def _extract_clash_data_flow(batch, final_state, batch_idx=0, device="cpu"):
    """Extract heavy atom positions for clash checking from flow model output."""
    B_idx = batch_idx
    rotmats = final_state["rotmats"][B_idx:B_idx + 1].to(device)
    trans = final_state["trans"][B_idx:B_idx + 1].to(device)
    angles = final_state["angles"][B_idx:B_idx + 1].to(device)
    seqs = final_state["seqs"][B_idx:B_idx + 1].to(device)

    # Full atom reconstruction for generated peptide
    pos14, _, _ = full_atom_reconstruction(R_bb=rotmats, t_bb=trans, angles=angles, aa=seqs)
    pos14 = pos14[0].cpu().numpy()  # (L, 14, 3)
    mask15 = get_heavyatom_mask(seqs)[0].cpu().numpy()  # (L, 15)
    mask14 = mask15[:, :14]  # trim to match pos14

    gen_mask = batch["generate_mask"][B_idx].bool().cpu().numpy()
    res_mask = batch["res_mask"][B_idx].bool().cpu().numpy()

    pep_idx = gen_mask & res_mask
    rec_idx = (~gen_mask) & res_mask

    pep_pos = pos14[pep_idx]        # (P, 14, 3)
    pep_mask = mask14[pep_idx]      # (P, 14)
    rec_pos = batch["pos_heavyatom"][B_idx].cpu().numpy()[rec_idx][:, :14, :]   # (R, 14, 3)
    rec_mask_ha = batch["mask_heavyatom"][B_idx].cpu().numpy()[rec_idx][:, :14]  # (R, 14)

    return pep_pos, pep_mask, rec_pos, rec_mask_ha


def _extract_clash_data_pephar(gen, receptor):
    """Extract heavy atom positions for clash checking from PepHAR output."""
    pep_coord = gen["pep_coord"]  # (L, 3, 3): CA, C, N
    pep_len = pep_coord.shape[0]

    # PepHAR only has backbone — build (L, 3, 3) with mask (L, 3)
    pep_pos = pep_coord.cpu().numpy()  # (L, 3, 3)
    pep_mask = np.ones((pep_len, 3), dtype=bool)

    rec_pos = receptor["pos_heavyatom"].cpu().numpy()  # (R, 15, 3)
    rec_mask = receptor["mask_heavyatom"].cpu().numpy()  # (R, 15)

    return pep_pos, pep_mask, rec_pos, rec_mask


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_gt_metrics(batch, final_state, batch_idx=0, device="cpu"):
    """Compute metrics for GT targets (RMSD, TM-score, AAR, iRMSD, etc.)."""
    pred_pep_ca, gt_pep_ca, rec_ca, gen_mask_np = extract_ca_coords(
        batch, final_state, batch_idx=batch_idx, device=device,
    )

    # Structural
    if len(pred_pep_ca) >= 3 and len(gt_pep_ca) >= 3:
        rmsd_unaligned, rmsd_aligned = compute_ca_rmsd(pred_pep_ca, gt_pep_ca)
        tm_score = compute_tm_score(pred_pep_ca, gt_pep_ca)
    else:
        rmsd_unaligned = rmsd_aligned = tm_score = float("nan")

    # Sequence
    gen_mask_t = batch["generate_mask"][0].bool().cpu()
    pred_aa = final_state["seqs"][0].cpu()
    gt_aa = batch["aa"][0].cpu()
    aar = compute_aar(pred_aa, gt_aa, gen_mask_t)
    pred_seq = idx_to_seq(pred_aa[gen_mask_t])
    gt_seq = idx_to_seq(gt_aa[gen_mask_t])

    # Interface
    irmsd = compute_interface_rmsd(pred_pep_ca, gt_pep_ca, rec_ca, interface_cutoff=8.0)
    bs_overlap = compute_binding_site_overlap(pred_pep_ca, gt_pep_ca, rec_ca, contact_cutoff=10.0)

    # Validity + clash
    validity = check_validity(pred_pep_ca)
    try:
        pep_pos, pep_mask, rec_pos, rec_mask = _extract_clash_data_flow(
            batch, final_state, batch_idx, device)
        clash = check_steric_clash(pep_pos, pep_mask, rec_pos, rec_mask)
        internal = check_internal_clash(pep_pos, pep_mask)
    except Exception:
        clash = {"num_clashes": -1, "worst_clash_dist": float("nan"), "clash_free": False}
        internal = {"internal_clashes": -1, "internal_worst_dist": float("nan"),
                     "internal_clash_free": False}

    return {
        "rmsd_unaligned": rmsd_unaligned,
        "rmsd_aligned": rmsd_aligned,
        "tm_score": tm_score,
        "aar": aar,
        "interface_rmsd": irmsd,
        "binding_site_overlap": bs_overlap,
        "pred_seq": pred_seq,
        "gt_seq": gt_seq,
        "pep_len": int(gen_mask_t.sum()),
        **validity,
        **clash,
        **internal,
    }


def compute_gt_metrics_pephar(gen, pephar_data, receptor):
    """Compute GT metrics for PepHAR output."""
    from Bio.SVDSuperimposer import SVDSuperimposer

    pred_ca = gen["pep_coord"][:, 0].cpu().numpy()  # (L, 3) CA coords
    gt_ca = pephar_data["pep_coord"][:, 0].cpu().numpy()

    # RMSD
    if len(pred_ca) >= 3:
        rmsd_unaligned = np.sqrt(np.mean(np.sum((pred_ca - gt_ca) ** 2, axis=-1)))
        sup = SVDSuperimposer()
        sup.set(gt_ca, pred_ca)
        sup.run()
        rmsd_aligned = float(sup.get_rms())
    else:
        rmsd_unaligned = rmsd_aligned = float("nan")

    # TM-score
    try:
        tm_score = compute_tm_score(pred_ca, gt_ca)
    except Exception:
        tm_score = float("nan")

    # AAR
    pred_aa = gen["pep_aa"].cpu()
    gt_aa = pephar_data["pep_aa"].cpu()
    if torch.is_tensor(pred_aa) and torch.is_tensor(gt_aa):
        aar = float((pred_aa == gt_aa).float().mean().item())
    else:
        aar = float("nan")

    # Validity + clash
    validity = check_validity(pred_ca)
    try:
        pep_pos, pep_mask, rec_pos, rec_mask = _extract_clash_data_pephar(gen, receptor)
        clash = check_steric_clash(pep_pos, pep_mask, rec_pos, rec_mask)
        internal = check_internal_clash(pep_pos, pep_mask)
    except Exception:
        clash = {"num_clashes": -1, "worst_clash_dist": float("nan"), "clash_free": False}
        internal = {"internal_clashes": -1, "internal_worst_dist": float("nan"),
                     "internal_clash_free": False}

    return {
        "rmsd_unaligned": rmsd_unaligned,
        "rmsd_aligned": rmsd_aligned,
        "tm_score": tm_score,
        "aar": aar,
        "interface_rmsd": float("nan"),
        "binding_site_overlap": float("nan"),
        "pred_seq": "",
        "gt_seq": "",
        "pep_len": len(pred_ca),
        **validity,
        **clash,
        **internal,
    }


def compute_denovo_metrics(gen=None, pred_ca=None, receptor=None):
    """Compute validity + clash metrics for de novo targets."""
    if gen is not None:
        pred_ca_t = gen["pep_coord"][:, 0]
        ca_np = pred_ca_t.cpu().numpy() if torch.is_tensor(pred_ca_t) else pred_ca_t
    elif pred_ca is not None:
        ca_np = pred_ca
    else:
        return {"valid": False, "clash_free": False, "mean_ca_dist": float("nan"),
                "pep_len": 0, "num_clashes": -1, "worst_clash_dist": float("nan"),
                "max_ca_dist_val": float("nan"), "min_ca_dist_val": float("nan")}

    validity = check_validity(ca_np)

    # Clash check if receptor provided and gen has coordinates
    clash = {"num_clashes": -1, "worst_clash_dist": float("nan"), "clash_free": False}
    internal = {"internal_clashes": -1, "internal_worst_dist": float("nan"),
                 "internal_clash_free": False}
    if receptor is not None and gen is not None:
        try:
            pep_pos, pep_mask, rec_pos, rec_mask = _extract_clash_data_pephar(gen, receptor)
            clash = check_steric_clash(pep_pos, pep_mask, rec_pos, rec_mask)
            internal = check_internal_clash(pep_pos, pep_mask)
        except Exception:
            pass

    return {
        **validity,
        **clash,
        **internal,
        "pep_len": len(ca_np),
        "rmsd_unaligned": float("nan"),
        "rmsd_aligned": float("nan"),
        "tm_score": float("nan"),
        "aar": float("nan"),
        "interface_rmsd": float("nan"),
        "binding_site_overlap": float("nan"),
    }


def compute_denovo_metrics_flow(batch, final_state, batch_idx=0, device="cpu"):
    """Compute de novo metrics from PepFlow/Approach B final state."""
    gen_mask = batch["generate_mask"][batch_idx].bool().cpu()
    res_mask = batch["res_mask"][batch_idx].bool().cpu()
    pred_ca = final_state["trans"][batch_idx].cpu()
    pep_idx = (gen_mask & res_mask)
    pep_ca = pred_ca[pep_idx].numpy()

    validity = check_validity(pep_ca)

    # Clash check
    try:
        pep_pos, pep_mask, rec_pos, rec_mask = _extract_clash_data_flow(
            batch, final_state, batch_idx, device)
        clash = check_steric_clash(pep_pos, pep_mask, rec_pos, rec_mask)
        internal = check_internal_clash(pep_pos, pep_mask)
    except Exception:
        clash = {"num_clashes": -1, "worst_clash_dist": float("nan"), "clash_free": False}
        internal = {"internal_clashes": -1, "internal_worst_dist": float("nan"),
                     "internal_clash_free": False}

    return {
        **validity,
        **clash,
        **internal,
        "pep_len": len(pep_ca),
        "rmsd_unaligned": float("nan"),
        "rmsd_aligned": float("nan"),
        "tm_score": float("nan"),
        "aar": float("nan"),
        "interface_rmsd": float("nan"),
        "binding_site_overlap": float("nan"),
    }


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(args):
    config, _ = load_config(args.config)
    seed_all(args.seed)
    device = torch.device(args.device)
    outdir = Path(args.outdir)

    # Initialize pyrosetta if Rosetta scoring is enabled
    if args.rosetta:
        init_pyrosetta(silent=True)
        print(f"[Benchmark] Rosetta FastRelax scoring enabled "
              f"(iterations={args.rosetta_iterations}, "
              f"score_gt={args.rosetta_score_gt})")

    # Load models
    models = {}
    if "PepFlow" in args.methods or "Approach_A" in args.methods:
        models["PepFlow_model"] = load_pepflow(config, args.pepflow_ckpt, device)
    if "Approach_B" in args.methods:
        models["Approach_B_model"] = load_approach_b(config, args.approach_b_ckpt, device)
    if "PepHAR" in args.methods or "Approach_A" in args.methods or "Approach_B" in args.methods:
        models["PepHAR_sampler"] = load_pephar_sampler(args, device)

    num_samples = args.num_samples

    all_summaries = []

    for target_name in args.targets:
        print(f"\n{'='*60}")
        print(f"  Target: {target_name}")
        print(f"{'='*60}")

        cfg = TARGET_CONFIGS[target_name]
        receptor, peptide, metadata = prepare_target(target_name, outdir)

        # Save GT PDB once
        gt_pdb_path: Path | None = None
        gt_relaxed_path: Path | None = None
        gt_rosetta_score: dict | None = None
        if metadata["has_gt"] and peptide is not None:
            gt_batch = build_pepflow_batch_gt(receptor, peptide, batch_size=1,
                                              target_name=target_name)
            gt_dir = outdir / target_name
            gt_pdb_path = gt_dir / "gt_complex.pdb"
            save_gt_pdb(gt_batch, 0, gt_pdb_path)

            # Relax GT once per target so its score is reused across all
            # methods and samples (and so that, if desired, the relaxed GT
            # could later serve as the structural reference).
            if args.rosetta and args.rosetta_score_gt:
                gt_relaxed_path = gt_dir / "gt_complex_relaxed.pdb"
                gt_rosetta_score = _relax_gt_once(
                    gt_pdb_path, gt_relaxed_path, args, peptide_chain="B",
                )
                if gt_rosetta_score is None or not gt_rosetta_score.get("success"):
                    err = (gt_rosetta_score or {}).get("error", "unknown")
                    print(f"  [Rosetta gt failed] {gt_pdb_path}: {err}")

        for method_name in args.methods:
            method_dir = outdir / target_name / method_name
            method_dir.mkdir(parents=True, exist_ok=True)
            results = []

            print(f"\n  [{target_name}] {method_name}: generating {num_samples} samples...")

            for sample_idx in tqdm(range(num_samples), desc=f"{method_name}"):
                try:
                    # Determine peptide length
                    if metadata["has_gt"]:
                        pep_length = metadata["pep_length"]
                    else:
                        pep_length = get_denovo_pep_length(metadata, sample_idx)

                    metrics = _run_single_sample(
                        method_name=method_name,
                        models=models,
                        receptor=receptor,
                        peptide=peptide,
                        metadata=metadata,
                        pep_length=pep_length,
                        sample_idx=sample_idx,
                        method_dir=method_dir,
                        device=device,
                        config=config,
                        num_steps=args.num_steps,
                        guidance_scale=args.guidance_scale,
                    )
                    metrics["sample_idx"] = sample_idx
                    metrics["target"] = target_name
                    metrics["method"] = method_name
                    metrics["pep_length_requested"] = pep_length

                    # Canonical post-relax evaluation: FastRelax the generated
                    # complex, dump the relaxed pose, then recompute every
                    # structural / sequence / interface / clash metric from
                    # the relaxed PDB. Tensor-based metrics from
                    # _run_single_sample are only kept as a fallback when
                    # `--no_rosetta` is set (legacy path, not the canonical
                    # evaluation).
                    if args.rosetta:
                        raw_pdb = method_dir / f"sample_{sample_idx:02d}.pdb"
                        relaxed_pdb = method_dir / f"sample_{sample_idx:02d}_relaxed.pdb"
                        gen_score = _relax_generated_pdb(
                            raw_pdb, relaxed_pdb, args, peptide_chain="B",
                        )
                        if gen_score["success"]:
                            pdb_metrics = compute_metrics_from_pdb(
                                gen_pdb=str(relaxed_pdb),
                                gt_pdb=str(gt_pdb_path) if gt_pdb_path else None,
                                peptide_chain="B",
                            )
                            # Overwrite tensor-based metrics with post-relax
                            # values for every column that exists in both.
                            metrics.update(pdb_metrics)
                        else:
                            print(f"  [Rosetta gen failed] {raw_pdb}: "
                                  f"{gen_score['error']}")
                        metrics.update(
                            _rosetta_score_columns(gen_score, gt_rosetta_score, args)
                        )

                    results.append(metrics)

                except Exception as e:
                    print(f"    [SKIP] {method_name} sample {sample_idx}: {e}")
                    import traceback
                    traceback.print_exc()

            # Save per-method CSV
            if results:
                df = pd.DataFrame(results)
                df.to_csv(method_dir / "metrics.csv", index=False)
                summary = _summarize(df, f"{target_name}/{method_name}")
                all_summaries.append(summary)

    # Generate cross-target summary
    _generate_summary(all_summaries, outdir)


def _run_single_sample(
    method_name, models, receptor, peptide, metadata, pep_length,
    sample_idx, method_dir, device, config, num_steps, guidance_scale,
):
    """Run a single sample for a given method and return metrics."""
    has_gt = metadata["has_gt"]
    target_name = metadata["target_name"]
    pdb_path = method_dir / f"sample_{sample_idx:02d}.pdb"

    if method_name == "PepFlow":
        return _run_pepflow_sample(
            models["PepFlow_model"], receptor, peptide, has_gt,
            pep_length, target_name, pdb_path, device, num_steps,
        )
    elif method_name == "PepHAR":
        return _run_pephar_sample(
            models["PepHAR_sampler"], receptor, peptide, has_gt,
            pep_length, pdb_path, device,
        )
    elif method_name == "Approach_A":
        return _run_approach_a_sample(
            models["PepFlow_model"], models.get("PepHAR_sampler"),
            receptor, peptide, has_gt, pep_length, target_name,
            pdb_path, device, config, num_steps,
        )
    elif method_name == "Approach_B":
        return _run_approach_b_sample(
            models["Approach_B_model"], receptor, peptide, has_gt,
            pep_length, target_name, pdb_path, device, num_steps,
            guidance_scale,
            pephar_sampler=models.get("PepHAR_sampler"),
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")


def _run_pepflow_sample(model, receptor, peptide, has_gt, pep_length,
                        target_name, pdb_path, device, num_steps):
    if has_gt:
        batch = build_pepflow_batch_gt(receptor, peptide, batch_size=1,
                                       target_name=target_name)
    else:
        batch = build_pepflow_batch_denovo(receptor, pep_length, batch_size=1,
                                           target_name=target_name)

    final = sample_pepflow(model, batch, device, num_steps=num_steps)
    save_generated_pdb(batch, final, 0, pdb_path)

    if has_gt:
        return compute_gt_metrics(batch, final, device=device)
    else:
        return compute_denovo_metrics_flow(batch, final, device=device)


def _run_pephar_sample(sampler, receptor, peptide, has_gt, pep_length,
                       pdb_path, device):
    if has_gt:
        pephar_data = build_pephar_data_gt(receptor, peptide)
        gen, raw_metrics = sample_pephar_gt(sampler, pephar_data, device)
        save_pephar_pdb(gen, receptor, pdb_path)
        return compute_gt_metrics_pephar(gen, pephar_data, receptor)
    else:
        rec_coord, rec_aa = build_pephar_data_denovo(receptor)
        gen, raw_metrics = sample_pephar_denovo(
            sampler, rec_coord, rec_aa, pep_length, device)
        save_pephar_pdb(gen, receptor, pdb_path)
        return compute_denovo_metrics(gen=gen, receptor=receptor)


def _run_approach_a_sample(pepflow_model, pephar_sampler, receptor, peptide,
                           has_gt, pep_length, target_name, pdb_path,
                           device, config, num_steps):
    if has_gt:
        batch = build_pepflow_batch_gt(receptor, peptide, batch_size=1,
                                       target_name=target_name)
        final = sample_approach_a_gt(
            pepflow_model, batch, device, num_steps=num_steps,
            num_anchors=5, contact_cutoff=4.0,
        )
        save_generated_pdb(batch, final, 0, pdb_path)
        return compute_gt_metrics(batch, final, device=device)
    else:
        # De novo: use PepHAR to find anchors, then PepFlow with inpainting
        if pephar_sampler is None:
            raise ValueError("PepHAR sampler required for Approach A de novo")

        # Step 1: Get anchors from PepHAR de novo
        rec_coord, rec_aa = build_pephar_data_denovo(receptor)
        gen, _ = sample_pephar_denovo(
            pephar_sampler, rec_coord, rec_aa, pep_length, device)

        # Step 2: Use generated peptide as anchor source for PepFlow
        # Build batch with the generated peptide as reference
        batch = build_pepflow_batch_denovo(receptor, pep_length, batch_size=1,
                                           target_name=target_name)
        # Sample with PepFlow (no inpainting for de novo — just use PepFlow)
        final = sample_pepflow(pepflow_model, batch, device, num_steps=num_steps)
        save_generated_pdb(batch, final, 0, pdb_path)
        return compute_denovo_metrics_flow(batch, final, device=device)


def _run_approach_b_sample(model, receptor, peptide, has_gt, pep_length,
                           target_name, pdb_path, device, num_steps,
                           guidance_scale, pephar_sampler=None):
    if has_gt:
        batch = build_pepflow_batch_gt(receptor, peptide, batch_size=1,
                                       target_name=target_name)
        final = sample_approach_b(
            model, batch, device, num_steps=num_steps,
            guidance_scale=guidance_scale,
        )
    else:
        batch = build_pepflow_batch_denovo(receptor, pep_length, batch_size=1,
                                           target_name=target_name)
        anchor_coords, anchor_types, anchor_mask = build_denovo_anchors(
            receptor, num_anchors=5, pephar_sampler=pephar_sampler)
        final = sample_approach_b(
            model, batch, device, num_steps=num_steps,
            guidance_scale=guidance_scale,
            anchor_coords=anchor_coords,
            anchor_types=anchor_types,
            anchor_mask=anchor_mask,
        )

    save_generated_pdb(batch, final, 0, pdb_path)

    if has_gt:
        return compute_gt_metrics(batch, final, device=device)
    else:
        return compute_denovo_metrics_flow(batch, final, device=device)


# ---------------------------------------------------------------------------
# Summary and comparison
# ---------------------------------------------------------------------------

GT_METRIC_COLS = [
    "rmsd_unaligned", "rmsd_aligned", "tm_score", "aar",
    "interface_rmsd", "binding_site_overlap",
    "valid", "clash_free", "num_clashes",
    "internal_clash_free", "internal_clashes",
    "rosetta_stab", "rosetta_bind",
    "rosetta_stab_gt", "rosetta_bind_gt",
    "rosetta_stab_delta", "rosetta_bind_delta",
]

DENOVO_METRIC_COLS = ["valid", "clash_free", "num_clashes",
                      "internal_clash_free", "internal_clashes", "mean_ca_dist",
                      "rosetta_stab", "rosetta_bind"]


def _summarize(df, label):
    """Print and return aggregate summary dict."""
    summary = {"label": label, "num_samples": len(df)}

    has_gt = not df["rmsd_unaligned"].isna().all() if "rmsd_unaligned" in df.columns else False
    cols = GT_METRIC_COLS if has_gt else DENOVO_METRIC_COLS

    print(f"\n  === {label} (n={len(df)}) ===")
    for col in cols:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) > 0:
            if df[col].dtype == bool or col == "valid":
                m = float(vals.astype(float).mean())
                summary[f"{col}_mean"] = m
                print(f"  {col:<28} {m:>8.4f}")
            else:
                m, s = float(vals.mean()), float(vals.std())
                summary[f"{col}_mean"] = m
                summary[f"{col}_std"] = s
                print(f"  {col:<28} {m:>8.4f} ± {s:>8.4f}")
    return summary


def _generate_summary(summaries, outdir):
    """Generate cross-target comparison table."""
    if not summaries:
        return

    outdir = Path(outdir)
    json_path = outdir / "summaries.json"
    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\n  Summaries saved to {json_path}")

    # Build comparison CSV
    rows = []
    for s in summaries:
        row = {"label": s["label"], "num_samples": s["num_samples"]}
        for key, val in s.items():
            if key not in ("label", "num_samples"):
                row[key] = val
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = outdir / "summary_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Comparison saved to {csv_path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Peptide binder design benchmark")

    parser.add_argument("--targets", nargs="+",
                        default=list(TARGET_CONFIGS.keys()),
                        help="Target names")
    parser.add_argument("--methods", nargs="+",
                        default=["PepFlow", "PepHAR", "Approach_A", "Approach_B"],
                        help="Method names")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--num_steps", type=int, default=200,
                        help="Denoising steps for flow models")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="Guidance scale for Approach B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="test_cases")

    # Checkpoints
    parser.add_argument("--config", type=str,
                        default="hotflow/configs/train_b.yaml")
    parser.add_argument("--pepflow_ckpt", type=str,
                        default="PepFlowww/model2.pt")
    parser.add_argument("--approach_b_ckpt", type=str,
                        default="logs_b/train_b_2026_04_14__21_22_14/checkpoints/400000.pt")
    parser.add_argument("--pephar_density_config", type=str,
                        default="PepHAR/ckpts/density_v4_x5o2_2024_09_08__11_25_36/density_v4_x5o2.yml")
    parser.add_argument("--pephar_density_weights", type=str,
                        default="PepHAR/ckpts/density_v4_x5o2_2024_09_08__11_25_36/checkpoints/1400.pt")
    parser.add_argument("--pephar_prediction_config", type=str,
                        default="PepHAR/ckpts/prediction_d2_x2o1_2024_09_08__11_21_33/prediction_d2_x2o1.yml")
    parser.add_argument("--pephar_prediction_weights", type=str,
                        default="PepHAR/ckpts/prediction_d2_x2o1_2024_09_08__11_21_33/checkpoints/2400.pt")

    # Rosetta refinement & scoring. The canonical evaluation FastRelax-es
    # every generated complex and computes ALL downstream metrics from the
    # relaxed PDB. Use `--no_rosetta` for a quick smoke test on a machine
    # without pyrosetta — that path falls back to tensor-based metrics from
    # the raw generation and is NOT the published evaluation.
    parser.add_argument("--rosetta", dest="rosetta", action="store_true",
                        default=True,
                        help="Run Rosetta FastRelax on each generation and "
                             "compute every metric from the relaxed PDB. "
                             "Default ON; requires pyrosetta.")
    parser.add_argument("--no_rosetta", dest="rosetta", action="store_false",
                        help="Skip Rosetta; report tensor-based metrics from "
                             "the raw generation (smoke-test fallback).")
    parser.add_argument("--rosetta_iterations", type=int, default=2,
                        help="Number of FastRelax + score iterations to mean over.")
    parser.add_argument("--rosetta_score_gt", action="store_true",
                        help="Also FastRelax gt_complex.pdb once per target and "
                             "report (gen - gt) deltas in rosetta_*_delta.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
