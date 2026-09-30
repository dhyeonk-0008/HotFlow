"""Contact-based hotspot labeling for training data.

Labels peptide residues as hotspots if any of their heavy atoms are within
a distance cutoff (default 4.0 A) of any receptor heavy atom.
"""

from __future__ import annotations

from typing import Any

import torch


def label_hotspots_by_contact(
    receptor_pos: torch.Tensor,
    receptor_mask: torch.Tensor,
    peptide_pos: torch.Tensor,
    peptide_mask: torch.Tensor,
    distance_cutoff: float = 4.0,
) -> torch.Tensor:
    """Compute per-residue hotspot labels based on heavy-atom contacts.

    Args:
        receptor_pos: (R, A, 3) receptor heavy-atom coordinates.
        receptor_mask: (R, A) boolean mask for valid receptor atoms.
        peptide_pos: (P, A, 3) peptide heavy-atom coordinates.
        peptide_mask: (P, A) boolean mask for valid peptide atoms.
        distance_cutoff: distance threshold in Angstroms.

    Returns:
        hotspot_mask: (P,) boolean tensor. True for residues that have at
        least one heavy-atom pair within ``distance_cutoff`` of the receptor.
    """
    P = peptide_pos.shape[0]

    # Collect all valid receptor atoms: (N_rec, 3)
    rec_atoms = receptor_pos[receptor_mask.bool()]
    if rec_atoms.numel() == 0:
        return torch.zeros(P, dtype=torch.bool, device=peptide_pos.device)

    hotspot = torch.zeros(P, dtype=torch.bool, device=peptide_pos.device)
    for i in range(P):
        pep_atoms = peptide_pos[i][peptide_mask[i].bool()]  # (n_i, 3)
        if pep_atoms.numel() == 0:
            continue
        # Pairwise distances between this residue's atoms and all receptor atoms
        dist = torch.cdist(pep_atoms, rec_atoms)  # (n_i, N_rec)
        if dist.min() <= distance_cutoff:
            hotspot[i] = True

    return hotspot


def label_hotspots_from_batch(
    batch: dict[str, Any],
    distance_cutoff: float = 4.0,
) -> torch.Tensor:
    """Label hotspots for a batched PepFlow-format dict.

    Expects standard PepFlow batch keys:
        - pos_heavyatom: (B, L, A, 3)
        - mask_heavyatom: (B, L, A)
        - generate_mask: (B, L) — True for peptide residues

    Returns:
        hotspot_labels: (B, L) boolean tensor. True only for peptide residues
        (generate_mask==True) that are within ``distance_cutoff`` of the receptor.
    """
    B, L = batch["generate_mask"].shape
    device = batch["generate_mask"].device
    hotspot_labels = torch.zeros(B, L, dtype=torch.bool, device=device)

    gen_mask = batch["generate_mask"].bool()
    pos = batch["pos_heavyatom"]
    atom_mask = batch["mask_heavyatom"]

    for b in range(B):
        pep_idx = gen_mask[b].nonzero(as_tuple=True)[0]
        rec_idx = (~gen_mask[b] & batch["res_mask"][b].bool()).nonzero(as_tuple=True)[0]

        if pep_idx.numel() == 0 or rec_idx.numel() == 0:
            continue

        rec_pos = pos[b, rec_idx]  # (R, A, 3)
        rec_mask = atom_mask[b, rec_idx]  # (R, A)
        pep_pos = pos[b, pep_idx]  # (P, A, 3)
        pep_mask = atom_mask[b, pep_idx]  # (P, A)

        per_residue = label_hotspots_by_contact(
            rec_pos, rec_mask, pep_pos, pep_mask, distance_cutoff
        )
        hotspot_labels[b, pep_idx] = per_residue

    return hotspot_labels


def select_top_k_hotspots(
    hotspot_labels: torch.Tensor,
    batch: dict[str, Any],
    k: int,
    distance_cutoff: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select top-K hotspot residues per batch element, ranked by contact count.

    Args:
        hotspot_labels: (B, L) boolean hotspot labels.
        batch: PepFlow batch dict with pos_heavyatom, mask_heavyatom, generate_mask, res_mask.
        k: number of anchors to select.
        distance_cutoff: distance threshold for counting contacts.

    Returns:
        anchor_indices: (B, K) global indices into L dimension. Padded with 0.
        anchor_coords: (B, K, 3, 3) backbone coords (CA, C, N). Padded with 0.
        anchor_types: (B, K) residue types. Padded with 20 (unknown).
        anchor_mask: (B, K) boolean validity mask.
    """
    B, L = hotspot_labels.shape
    device = hotspot_labels.device

    anchor_indices = torch.zeros(B, k, dtype=torch.long, device=device)
    anchor_coords = torch.zeros(B, k, 3, 3, dtype=batch["pos_heavyatom"].dtype, device=device)
    anchor_types = torch.full((B, k), 20, dtype=torch.long, device=device)
    anchor_mask = torch.zeros(B, k, dtype=torch.bool, device=device)

    gen_mask = batch["generate_mask"].bool()
    pos = batch["pos_heavyatom"]
    atom_mask = batch["mask_heavyatom"]

    # Atom indices for backbone: N=0, CA=1, C=2 (PepFlow convention)
    CA, C, N = 1, 2, 0

    for b in range(B):
        hs_idx = hotspot_labels[b].nonzero(as_tuple=True)[0]  # global indices of hotspots
        if hs_idx.numel() == 0:
            continue

        # Rank by contact count
        rec_idx = (~gen_mask[b] & batch["res_mask"][b].bool()).nonzero(as_tuple=True)[0]
        rec_atoms = pos[b, rec_idx][atom_mask[b, rec_idx].bool()]  # (N_rec, 3)

        counts = []
        for idx in hs_idx:
            pep_atoms = pos[b, idx][atom_mask[b, idx].bool()]
            if pep_atoms.numel() == 0 or rec_atoms.numel() == 0:
                counts.append(0)
            else:
                counts.append(int((torch.cdist(pep_atoms, rec_atoms) <= distance_cutoff).sum().item()))

        counts_t = torch.tensor(counts, device=device, dtype=torch.float)
        n_select = min(k, hs_idx.numel())
        top_indices = counts_t.argsort(descending=True)[:n_select]

        for i, ti in enumerate(top_indices):
            g_idx = hs_idx[ti]
            anchor_indices[b, i] = g_idx
            anchor_coords[b, i, 0] = pos[b, g_idx, CA]  # CA
            anchor_coords[b, i, 1] = pos[b, g_idx, C]   # C
            anchor_coords[b, i, 2] = pos[b, g_idx, N]   # N
            anchor_types[b, i] = batch["aa"][b, g_idx].clamp(0, 20)
            anchor_mask[b, i] = True

    return anchor_indices, anchor_coords, anchor_types, anchor_mask
