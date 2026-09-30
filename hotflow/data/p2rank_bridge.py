"""P2Rank integration for de novo peptide sampling.

Converts P2Rank pocket predictions on receptor surfaces into
HotspotAnchor objects that FlowModelB can use for conditioning.

P2Rank predicts binding pocket residues on a receptor.  This module:
  1. Parses P2Rank residue-level CSV output.
  2. Extracts backbone CA/C/N coordinates from the receptor PDB.
  3. Selects top-K pocket residues by P2Rank score.
  4. Packages them as HotspotAnchor objects for FlowModelB.sample().

Usage (de novo sampling):
    anchors = p2rank_to_anchors(receptor_pdb, p2rank_csv, top_k=5)
    coords, types, mask = anchors_to_tensor(anchors, K=5)
    traj = model.sample(batch, anchor_coords=coords, anchor_types=types,
                        anchor_mask=mask)
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PEPFLOW_ROOT = REPO_ROOT / "PepFlowww"
if str(PEPFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(PEPFLOW_ROOT))

from pepflow.modules.protein.constants import BBHeavyAtom  # noqa: E402
from pepflow.modules.protein.parsers import parse_pdb  # noqa: E402

from hotflow.data_types import HotspotAnchor  # noqa: E402

# Standard 3-letter → index mapping (PepFlow convention)
AA_3TO1 = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
}
UNKNOWN_AA = 20


@dataclass(frozen=True)
class P2RankResidue:
    """Single residue from P2Rank prediction."""
    chain: str
    residue_label: str       # e.g. "45" or "45A" (resseq + icode)
    residue_name: str        # 3-letter code, e.g. "ALA"
    score: float             # P2Rank pocket probability
    pocket: Optional[int]    # pocket ID (1-based), None if not in pocket


def parse_p2rank_residues(csv_path: str | Path) -> list[P2RankResidue]:
    """Parse P2Rank *_residues.csv output file.

    Expected CSV columns (P2Rank v2.x format):
        chain, residue_label, residue_name, score, zscore, probability, pocket

    Also handles older format:
        chain, residue_label, residue_name, score, pocket

    Returns list of P2RankResidue sorted by score descending.
    """
    residues: list[P2RankResidue] = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        # Normalize column names (P2Rank sometimes uses leading spaces)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}

            chain = row.get('chain', 'A')
            residue_label = row.get('residue_label', row.get('label', ''))
            residue_name = row.get('residue_name', row.get('name', 'UNK'))

            # P2Rank v2.4+ uses 'probability', older uses 'score'
            score_str = row.get('probability', row.get('score', '0.0'))
            try:
                score = float(score_str)
            except ValueError:
                score = 0.0

            pocket_str = row.get('pocket', '')
            pocket = int(pocket_str) if pocket_str and pocket_str != '0' else None

            residues.append(P2RankResidue(
                chain=chain,
                residue_label=residue_label,
                residue_name=residue_name.upper(),
                score=score,
                pocket=pocket,
            ))

    return sorted(residues, key=lambda r: -r.score)


def parse_p2rank_pockets(csv_path: str | Path) -> list[dict]:
    """Parse P2Rank *_predictions.csv (pocket-level summary).

    Returns list of pocket dicts with keys:
        name, rank, score, probability, residue_ids, center_x/y/z
    """
    pockets: list[dict] = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            pocket = {
                'name': row.get('name', ''),
                'rank': int(row.get('rank', 0)),
                'score': float(row.get('score', 0.0)),
                'probability': float(row.get('probability', row.get('score', 0.0))),
            }
            # Parse residue IDs if present (format: "A_45 A_46 A_50")
            residue_ids_str = row.get('residue_ids', '')
            if residue_ids_str:
                pocket['residue_ids'] = residue_ids_str.split()
            else:
                pocket['residue_ids'] = []

            # Pocket center coordinates
            for coord in ('center_x', 'center_y', 'center_z'):
                if coord in row:
                    pocket[coord] = float(row[coord])

            pockets.append(pocket)

    return sorted(pockets, key=lambda p: -p['score'])


def _extract_backbone_coords(
    parsed_pdb: dict[str, torch.Tensor],
    residue_index: int,
) -> torch.Tensor:
    """Extract CA/C/N coordinates for a single residue.

    Returns: (3, 3) tensor in [CA, C, N] order.
    """
    pos = parsed_pdb['pos_heavyatom'][residue_index]
    ca = pos[BBHeavyAtom.CA]
    c = pos[BBHeavyAtom.C]
    n = pos[BBHeavyAtom.N]
    return torch.stack([ca, c, n], dim=0)


def _match_residue_to_pdb(
    parsed_pdb: dict,
    chain: str,
    residue_label: str,
) -> Optional[int]:
    """Find index in parsed PDB matching chain + residue_label.

    residue_label may be "45" (resseq) or "45A" (resseq + icode).
    """
    n_res = parsed_pdb['aa'].shape[0]

    # Parse label into resseq and optional icode
    resseq_str = ''
    icode = ' '
    for i, ch in enumerate(residue_label):
        if ch.isdigit() or (ch == '-' and i == 0):
            resseq_str += ch
        else:
            icode = residue_label[i:]
            break

    if not resseq_str:
        return None
    target_resseq = int(resseq_str)

    for idx in range(n_res):
        pdb_chain = parsed_pdb['chain_id'][idx] if 'chain_id' in parsed_pdb else 'A'
        pdb_resseq = int(parsed_pdb['resseq'][idx].item()) if torch.is_tensor(parsed_pdb['resseq'][idx]) else int(parsed_pdb['resseq'][idx])
        pdb_icode = parsed_pdb.get('icode', [' '] * n_res)[idx]

        if pdb_chain == chain and pdb_resseq == target_resseq:
            if icode == ' ' or pdb_icode == icode:
                return idx

    return None


def p2rank_to_anchors(
    receptor_pdb: str | Path,
    p2rank_residue_csv: str | Path,
    top_k: int = 5,
    pocket_id: Optional[int] = None,
    min_score: float = 0.0,
) -> list[HotspotAnchor]:
    """Convert P2Rank predictions to HotspotAnchors.

    Reads receptor PDB and P2Rank residue CSV, extracts backbone coords
    from top-scoring pocket residues.

    Args:
        receptor_pdb: path to receptor PDB file.
        p2rank_residue_csv: path to P2Rank *_residues.csv.
        top_k: maximum number of anchors to return.
        pocket_id: if set, only use residues from this pocket (1-based).
        min_score: minimum P2Rank score threshold.

    Returns:
        List of HotspotAnchor objects (up to top_k), sorted by score.
    """
    # Parse receptor PDB
    parsed_pdb, _ = parse_pdb(str(receptor_pdb))
    if parsed_pdb is None:
        raise ValueError(f"Failed to parse receptor PDB: {receptor_pdb}")

    # Parse P2Rank residue predictions
    residues = parse_p2rank_residues(p2rank_residue_csv)

    # Filter by pocket_id and min_score
    if pocket_id is not None:
        residues = [r for r in residues if r.pocket == pocket_id]
    residues = [r for r in residues if r.score >= min_score]

    # Match to PDB and build anchors
    anchors: list[HotspotAnchor] = []
    for res in residues:
        if len(anchors) >= top_k:
            break

        idx = _match_residue_to_pdb(parsed_pdb, res.chain, res.residue_label)
        if idx is None:
            continue

        # Check backbone atoms are valid
        mask = parsed_pdb['mask_heavyatom'][idx]
        if not (mask[BBHeavyAtom.CA] and mask[BBHeavyAtom.C] and mask[BBHeavyAtom.N]):
            continue

        backbone_coords = _extract_backbone_coords(parsed_pdb, idx)
        aa_idx = AA_3TO1.get(res.residue_name, UNKNOWN_AA)

        anchors.append(HotspotAnchor(
            residue_index=idx,
            backbone_coords=backbone_coords.detach().cpu(),
            residue_type=aa_idx,
            score=res.score,
            source="p2rank",
        ))

    return anchors


def pocket_center_to_anchors(
    receptor_pdb: str | Path,
    p2rank_predictions_csv: str | Path,
    top_k: int = 5,
    pocket_rank: int = 1,
    radius: float = 8.0,
) -> list[HotspotAnchor]:
    """Use P2Rank pocket center to select nearby receptor residues as anchors.

    Alternative to residue-level CSV: uses pocket center coordinates and
    selects the closest receptor residues within a radius.

    Args:
        receptor_pdb: path to receptor PDB file.
        p2rank_predictions_csv: path to P2Rank *_predictions.csv.
        top_k: maximum number of anchors.
        pocket_rank: which pocket to use (1 = top-ranked).
        radius: max distance from pocket center to include residues (A).

    Returns:
        List of HotspotAnchor objects.
    """
    parsed_pdb, _ = parse_pdb(str(receptor_pdb))
    if parsed_pdb is None:
        raise ValueError(f"Failed to parse receptor PDB: {receptor_pdb}")

    pockets = parse_p2rank_pockets(p2rank_predictions_csv)
    if not pockets:
        return []

    # Select target pocket
    target = None
    for p in pockets:
        if p['rank'] == pocket_rank:
            target = p
            break
    if target is None:
        target = pockets[0]

    if 'center_x' not in target:
        # Fallback: use residue-level approach
        return []

    center = torch.tensor([target['center_x'], target['center_y'], target['center_z']])

    # Find residues closest to pocket center
    n_res = parsed_pdb['aa'].shape[0]
    ca_coords = parsed_pdb['pos_heavyatom'][:, BBHeavyAtom.CA]  # (N, 3)
    distances = torch.norm(ca_coords - center.unsqueeze(0), dim=-1)  # (N,)

    # Filter by radius and sort by distance
    within_radius = distances < radius
    valid_mask = parsed_pdb['mask_heavyatom'][:, BBHeavyAtom.CA] & \
                 parsed_pdb['mask_heavyatom'][:, BBHeavyAtom.C] & \
                 parsed_pdb['mask_heavyatom'][:, BBHeavyAtom.N]
    candidates = within_radius & valid_mask

    candidate_indices = torch.nonzero(candidates, as_tuple=False).squeeze(-1)
    candidate_dists = distances[candidate_indices]
    sorted_order = candidate_dists.argsort()
    selected = candidate_indices[sorted_order[:top_k]]

    anchors: list[HotspotAnchor] = []
    for idx_tensor in selected:
        idx = int(idx_tensor.item())
        backbone_coords = _extract_backbone_coords(parsed_pdb, idx)
        aa_type = int(parsed_pdb['aa'][idx].item())

        anchors.append(HotspotAnchor(
            residue_index=idx,
            backbone_coords=backbone_coords.detach().cpu(),
            residue_type=aa_type if 0 <= aa_type <= 19 else UNKNOWN_AA,
            score=float(1.0 / (distances[idx].item() + 1e-6)),
            source="p2rank_pocket_center",
        ))

    return anchors


def p2rank_pocket_to_pephar_anchors(
    receptor_pdb: str | Path,
    p2rank_predictions_csv: str | Path,
    pephar_sampler: "hotflow.models.hotspot_sampler.PepHARHotspotSampler",
    top_k: int = 5,
    pocket_rank: int = 1,
    anchor_steps: int = 100,
    spread: float = 5.0,
    verbose: bool = False,
) -> list[HotspotAnchor]:
    """Chain P2Rank pocket prediction with PepHAR EBM hotspot sampling.

    1. Parse P2Rank pocket-level CSV to get the pocket center.
    2. Parse receptor PDB for backbone coordinates.
    3. Run PepHAR's EBM from the pocket center to predict peptide
       hotspot anchor positions (backbone coords + residue types).

    This produces anchors in the same domain as the training pipeline
    (peptide-side hotspot residue coordinates), unlike the raw P2Rank
    functions which return receptor-side coordinates.

    Args:
        receptor_pdb: path to receptor PDB file.
        p2rank_predictions_csv: path to P2Rank ``*_predictions.csv``.
        pephar_sampler: initialised PepHARHotspotSampler with a loaded
            density model.
        top_k: number of hotspot anchors to produce.
        pocket_rank: which P2Rank pocket to use (1 = top-ranked).
        anchor_steps: Langevin optimisation steps per anchor.
        spread: std-dev (Angstroms) for initial position scatter.
        verbose: print per-step EBM progress.

    Returns:
        List of HotspotAnchor with peptide-side backbone coordinates.
    """
    # --- pocket center from P2Rank ---
    pockets = parse_p2rank_pockets(p2rank_predictions_csv)
    if not pockets:
        raise ValueError(f"No pockets found in {p2rank_predictions_csv}")

    target = None
    for p in pockets:
        if p['rank'] == pocket_rank:
            target = p
            break
    if target is None:
        target = pockets[0]

    if 'center_x' not in target:
        raise ValueError(
            f"Pocket rank={pocket_rank} has no center coordinates; "
            "cannot determine pocket region for EBM sampling."
        )

    pocket_center = torch.tensor(
        [target['center_x'], target['center_y'], target['center_z']]
    )

    # --- receptor structure ---
    parsed_pdb, _ = parse_pdb(str(receptor_pdb))
    if parsed_pdb is None:
        raise ValueError(f"Failed to parse receptor PDB: {receptor_pdb}")

    n_res = parsed_pdb['aa'].shape[0]
    # Build (R, 3, 3) backbone coords in [CA, C, N] order
    rec_coord = torch.stack([
        parsed_pdb['pos_heavyatom'][:, BBHeavyAtom.CA],
        parsed_pdb['pos_heavyatom'][:, BBHeavyAtom.C],
        parsed_pdb['pos_heavyatom'][:, BBHeavyAtom.N],
    ], dim=1)  # (R, 3, 3)
    rec_aa = parsed_pdb['aa'].long()

    # --- PepHAR EBM sampling ---
    return pephar_sampler.sample_de_novo(
        rec_coord=rec_coord,
        rec_aa=rec_aa,
        pocket_center=pocket_center,
        num_anchors=top_k,
        anchor_steps=anchor_steps,
        spread=spread,
        verbose=verbose,
    )


def anchors_to_tensor(
    anchors: Sequence[HotspotAnchor],
    K: int = 5,
    batch_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert HotspotAnchor list to batched tensors for FlowModelB.sample().

    Pads to K anchors if fewer are provided. Broadcasts to batch_size.

    Args:
        anchors: list of HotspotAnchor objects.
        K: target number of anchors (pads with zeros + mask=False).
        batch_size: batch dimension size.

    Returns:
        anchor_coords: (B, K, 3, 3)
        anchor_types: (B, K) long
        anchor_mask: (B, K) bool
    """
    n = min(len(anchors), K)

    coords = torch.zeros(K, 3, 3)
    types = torch.full((K,), UNKNOWN_AA, dtype=torch.long)
    mask = torch.zeros(K, dtype=torch.bool)

    for i in range(n):
        anchors[i].validate()
        coords[i] = anchors[i].backbone_coords
        types[i] = anchors[i].residue_type
        mask[i] = True

    # Broadcast to batch
    anchor_coords = coords.unsqueeze(0).expand(batch_size, -1, -1, -1).clone()
    anchor_types = types.unsqueeze(0).expand(batch_size, -1).clone()
    anchor_mask = mask.unsqueeze(0).expand(batch_size, -1).clone()

    return anchor_coords, anchor_types, anchor_mask
