"""Contact preservation loss for Approach B training.

Encourages generated hotspot residues to maintain proximity to the receptor
surface, measured by the distance between hotspot CA atoms and their nearest
receptor CA atoms.
"""

from __future__ import annotations

import torch


def contact_preservation_loss(
    pred_trans: torch.Tensor,
    batch: dict[str, torch.Tensor],
    hotspot_labels: torch.Tensor,
    target_distance: float = 4.0,
    margin: float = 2.0,
) -> torch.Tensor:
    """Compute contact preservation loss for hotspot residues.

    For each hotspot residue, we compute the distance from its predicted CA
    position to the nearest receptor CA. If this distance exceeds
    ``target_distance``, a squared penalty is applied (with a soft margin).

    Args:
        pred_trans: (B, L, 3) predicted CA positions from the flow model.
        batch: PepFlow batch dict containing:
            - generate_mask: (B, L)
            - res_mask: (B, L)
            - pos_heavyatom: (B, L, A, 3) — ground-truth positions (receptor
              positions are fixed and used as reference).
        hotspot_labels: (B, L) boolean tensor indicating hotspot residues.
        target_distance: desired maximum distance (Angstroms) between a
            hotspot CA and the nearest receptor CA.
        margin: soft margin beyond ``target_distance`` before penalty kicks in.
            The penalty is: max(0, dist - target_distance)^2 / margin^2.

    Returns:
        Scalar loss averaged over all hotspot residues across the batch.
    """
    B, L, _ = pred_trans.shape
    device = pred_trans.device

    gen_mask = batch["generate_mask"].bool()
    res_mask = batch["res_mask"].bool()
    pos = batch["pos_heavyatom"]
    CA_IDX = 1  # PepFlow atom ordering: N=0, CA=1, C=2

    total_loss = torch.tensor(0.0, device=device)
    count = 0

    for b in range(B):
        # Receptor CA positions (ground-truth, fixed during generation)
        rec_idx = (~gen_mask[b] & res_mask[b]).nonzero(as_tuple=True)[0]
        if rec_idx.numel() == 0:
            continue
        rec_ca = pos[b, rec_idx, CA_IDX]  # (R, 3)

        # Hotspot residue indices
        hs_idx = hotspot_labels[b].nonzero(as_tuple=True)[0]
        if hs_idx.numel() == 0:
            continue

        # Predicted CA positions for hotspot residues
        hs_ca_pred = pred_trans[b, hs_idx]  # (H, 3)

        # Distance to nearest receptor CA
        dist = torch.cdist(hs_ca_pred, rec_ca)  # (H, R)
        min_dist = dist.min(dim=1).values  # (H,)

        # Hinge loss: penalize only if distance > target_distance
        excess = torch.clamp(min_dist - target_distance, min=0.0)
        loss = (excess / margin) ** 2

        total_loss = total_loss + loss.sum()
        count += hs_idx.numel()

    if count == 0:
        # Return a graph-connected zero so DDP backward still flows through
        # `pred_trans` (and hence all upstream model params). A bare
        # `torch.tensor(0.0)` is disconnected and would leave params with
        # undefined grad → DDP reducer crash on the next iter.
        return pred_trans.sum() * 0.0

    return total_loss / count
