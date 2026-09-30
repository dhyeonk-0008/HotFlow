"""Data transforms for Approach B hotspot-conditioned training.

Provides per-item transforms that annotate PepFlow data dicts with
hotspot labels and anchor information, so these don't need to be
recomputed every training step.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Optional

import torch

from hotflow.data.hotspot_labeling import label_hotspots_by_contact


class HotspotAnnotationTransform:
    """Per-item transform: adds hotspot_labels and anchor tensors.

    Applied to a single PepFlow-format data dict (before collation).
    Adds the following keys:
        - hotspot_labels: (L,) bool — True for hotspot peptide residues.
        - anchor_indices: (K,) long — global indices of top-K anchors (0-padded).
        - anchor_coords: (K, 3, 3) float — backbone CA/C/N coords of anchors.
        - anchor_types: (K,) long — residue types of anchors (padded with 20).
        - anchor_mask: (K,) bool — validity mask for anchors.

    Args:
        num_anchors: K, number of anchor residues to select.
        distance_cutoff: contact distance threshold in Angstroms.
    """

    def __init__(self, num_anchors: int = 5, distance_cutoff: float = 4.0):
        self.num_anchors = num_anchors
        self.distance_cutoff = distance_cutoff

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        gen_mask = data["generate_mask"].bool()
        pos = data["pos_heavyatom"]      # (L, A, 3)
        atom_mask = data["mask_heavyatom"]  # (L, A)

        pep_idx = gen_mask.nonzero(as_tuple=True)[0]
        # For res_mask: in per-item data, all residues are valid (padding not yet applied)
        rec_idx = (~gen_mask).nonzero(as_tuple=True)[0]

        L = gen_mask.shape[0]
        K = self.num_anchors
        device = pos.device

        hotspot_labels = torch.zeros(L, dtype=torch.bool, device=device)

        # Atom indices for backbone: N=0, CA=1, C=2
        CA, C, N = 1, 2, 0

        anchor_indices = torch.zeros(K, dtype=torch.long, device=device)
        anchor_coords = torch.zeros(K, 3, 3, dtype=pos.dtype, device=device)
        anchor_types = torch.full((K,), 20, dtype=torch.long, device=device)
        anchor_mask = torch.zeros(K, dtype=torch.bool, device=device)

        if pep_idx.numel() == 0 or rec_idx.numel() == 0:
            data["hotspot_labels"] = hotspot_labels
            data["anchor_indices"] = anchor_indices
            data["anchor_coords"] = anchor_coords
            data["anchor_types"] = anchor_types
            data["anchor_mask"] = anchor_mask
            return data

        rec_pos = pos[rec_idx]       # (R, A, 3)
        rec_mask = atom_mask[rec_idx]  # (R, A)
        pep_pos = pos[pep_idx]       # (P, A, 3)
        pep_mask = atom_mask[pep_idx]  # (P, A)

        # Label hotspots per peptide residue
        per_residue = label_hotspots_by_contact(
            rec_pos, rec_mask, pep_pos, pep_mask, self.distance_cutoff
        )
        hotspot_labels[pep_idx] = per_residue

        # Select top-K by contact count
        hs_local = per_residue.nonzero(as_tuple=True)[0]  # indices into pep_idx
        if hs_local.numel() == 0:
            data["hotspot_labels"] = hotspot_labels
            data["anchor_indices"] = anchor_indices
            data["anchor_coords"] = anchor_coords
            data["anchor_types"] = anchor_types
            data["anchor_mask"] = anchor_mask
            return data

        # Collect all valid receptor atoms for contact counting
        rec_atoms = rec_pos[rec_mask.bool()]  # (N_rec, 3)

        counts = []
        for li in hs_local:
            pep_atoms = pep_pos[li][pep_mask[li].bool()]
            if pep_atoms.numel() == 0 or rec_atoms.numel() == 0:
                counts.append(0)
            else:
                counts.append(
                    int((torch.cdist(pep_atoms, rec_atoms) <= self.distance_cutoff).sum().item())
                )

        counts_t = torch.tensor(counts, dtype=torch.float, device=device)
        n_select = min(K, hs_local.numel())
        top_local = counts_t.argsort(descending=True)[:n_select]

        for i, tl in enumerate(top_local):
            local_idx = hs_local[tl]
            g_idx = pep_idx[local_idx]
            anchor_indices[i] = g_idx
            anchor_coords[i, 0] = pos[g_idx, CA]  # CA
            anchor_coords[i, 1] = pos[g_idx, C]   # C
            anchor_coords[i, 2] = pos[g_idx, N]   # N
            anchor_types[i] = data["aa"][g_idx].clamp(0, 20)
            anchor_mask[i] = True

        data["hotspot_labels"] = hotspot_labels
        data["anchor_indices"] = anchor_indices
        data["anchor_coords"] = anchor_coords
        data["anchor_types"] = anchor_types
        data["anchor_mask"] = anchor_mask
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_anchors={self.num_anchors}, "
            f"distance_cutoff={self.distance_cutoff})"
        )


class AnchorToGraphTransform:
    """Per-item transform: appends anchor nodes to the residue graph.

    Converts anchor_coords/types/mask (from HotspotAnnotationTransform) into
    additional residues appended to all L-dimensional batch tensors.  After
    this transform the batch has L+K residues where the last K are anchors.

    Anchors are assigned:
        - generate_mask = False  (fixed conditioning, not generated)
        - chain_nb = 2           (separate chain, zeroes cross-chain relpos)
        - node_type = 2          (new field: 0=receptor, 1=peptide, 2=anchor)
        - Only backbone atoms (N, CA, C) in pos_heavyatom

    Also adds:
        - node_type: (L+K,) long — node type indicator for the full graph
        - L_orig: int — original sequence length before anchor appending
    """

    # BBHeavyAtom indices: N=0, CA=1, C=2
    _N, _CA, _C = 0, 1, 2

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        anchor_coords = data.get("anchor_coords")  # (K, 3, 3) in CA/C/N order
        anchor_types = data.get("anchor_types")     # (K,)
        anchor_mask = data.get("anchor_mask")       # (K,)

        if anchor_coords is None or anchor_types is None or anchor_mask is None:
            return data

        K = anchor_coords.shape[0]
        L = data["aa"].shape[0]
        device = data["aa"].device
        A = data["pos_heavyatom"].shape[1]  # 15

        # --- Build node_type for original residues ---
        gen_mask = data["generate_mask"].bool()
        node_type = torch.where(gen_mask, torch.ones(L, dtype=torch.long, device=device),
                                torch.zeros(L, dtype=torch.long, device=device))

        # --- Build anchor expansions ---
        anc_pos = torch.zeros(K, A, 3, dtype=data["pos_heavyatom"].dtype, device=device)
        anc_atom_mask = torch.zeros(K, A, dtype=data["mask_heavyatom"].dtype, device=device)

        for i in range(K):
            if anchor_mask[i]:
                anc_pos[i, self._CA] = anchor_coords[i, 0]  # CA
                anc_pos[i, self._C] = anchor_coords[i, 1]   # C
                anc_pos[i, self._N] = anchor_coords[i, 2]   # N
                anc_atom_mask[i, self._N] = True
                anc_atom_mask[i, self._CA] = True
                anc_atom_mask[i, self._C] = True

        anc_aa = anchor_types.clone()
        anc_gen_mask = torch.zeros(K, dtype=torch.bool, device=device)
        anc_chain_nb = torch.full((K,), 2, dtype=data["chain_nb"].dtype, device=device)
        anc_res_nb = torch.arange(K, dtype=data["res_nb"].dtype, device=device)
        anc_torsion = torch.zeros(K, data["torsion_angle"].shape[-1],
                                  dtype=data["torsion_angle"].dtype, device=device)
        anc_torsion_mask = torch.zeros(K, data["torsion_angle_mask"].shape[-1],
                                       dtype=data["torsion_angle_mask"].dtype, device=device)
        anc_node_type = torch.full((K,), 2, dtype=torch.long, device=device)

        # --- Concatenate along L dimension ---
        data["pos_heavyatom"] = torch.cat([data["pos_heavyatom"], anc_pos], dim=0)
        data["mask_heavyatom"] = torch.cat([data["mask_heavyatom"], anc_atom_mask], dim=0)
        data["aa"] = torch.cat([data["aa"], anc_aa], dim=0)
        data["generate_mask"] = torch.cat([data["generate_mask"].bool(), anc_gen_mask], dim=0)
        data["chain_nb"] = torch.cat([data["chain_nb"], anc_chain_nb], dim=0)
        data["res_nb"] = torch.cat([data["res_nb"], anc_res_nb], dim=0)
        data["torsion_angle"] = torch.cat([data["torsion_angle"], anc_torsion], dim=0)
        data["torsion_angle_mask"] = torch.cat([data["torsion_angle_mask"], anc_torsion_mask], dim=0)
        data["node_type"] = torch.cat([node_type, anc_node_type], dim=0)
        data["L_orig"] = torch.tensor(L, dtype=torch.long, device=device)

        # Extend hotspot_labels if present
        if "hotspot_labels" in data:
            anc_hs = torch.zeros(K, dtype=torch.bool, device=device)
            data["hotspot_labels"] = torch.cat([data["hotspot_labels"], anc_hs], dim=0)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class PepHARAnchorTransform:
    """Per-item transform: replaces GT-contact anchors with PepHAR-EBM
    pre-computed anchors loaded from an LMDB cache.

    The cache is produced by `hotflow/scripts/precompute_pephar_anchors.py`
    and keyed by `data["id"]`. Each value contains N stochastic samples; one
    is drawn at random per call (training-time augmentation).

    `hotspot_labels` and `anchor_indices` are still computed from GT contacts
    via `HotspotAnnotationTransform` — these drive the `contact_loss` which
    must reference the *actual* peptide hotspot residues, not the EBM-drifted
    anchor coords. Only `anchor_coords`, `anchor_types`, `anchor_mask` (the
    fields fed to `HotspotEncoder` / cross-attention) are overwritten.

    Args:
        cache_path: path to the LMDB cache.
        num_anchors: K, must match the cache layout.
        distance_cutoff: for the GT-label fallback transform.
        augment_sigma: extra Gaussian noise (Å) on coords at access time
            for cheap extra stochasticity. 0 disables.
        fallback_gt_on_miss: if True, leave the GT-derived anchors in place
            for entries not present in the cache. If False, raise.
    """

    def __init__(
        self,
        cache_path: str,
        num_anchors: int = 5,
        distance_cutoff: float = 4.0,
        augment_sigma: float = 0.0,
        fallback_gt_on_miss: bool = True,
    ):
        self.cache_path = str(cache_path)
        self.num_anchors = num_anchors
        self.augment_sigma = float(augment_sigma)
        self.fallback_gt_on_miss = bool(fallback_gt_on_miss)
        # GT transform always runs first to set hotspot_labels (for contact_loss)
        # and to serve as a fallback anchor source when cache misses.
        self.gt_transform = HotspotAnnotationTransform(
            num_anchors=num_anchors,
            distance_cutoff=distance_cutoff,
        )
        # Open LMDB lazily per-process so DataLoader workers fork cleanly.
        self._env_pid: Optional[int] = None
        self._env = None

    def _env_lazy(self):
        import lmdb
        pid = os.getpid()
        if self._env is None or self._env_pid != pid:
            self._env = lmdb.open(
                self.cache_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                subdir=False,
                max_readers=512,
            )
            self._env_pid = pid
        return self._env

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        # Always populate hotspot_labels + GT anchor fallback first.
        data = self.gt_transform(data)

        entry_id = data.get("id", None)
        if entry_id is None:
            return data  # no key → GT fallback already in place

        key = entry_id.encode("utf-8") if isinstance(entry_id, str) else bytes(entry_id)
        env = self._env_lazy()
        with env.begin() as txn:
            raw = txn.get(key)
        if raw is None:
            if not self.fallback_gt_on_miss:
                raise KeyError(
                    f"PepHARAnchorTransform: id={entry_id!r} not in cache "
                    f"{self.cache_path}"
                )
            return data  # silent GT fallback

        cached = pickle.loads(raw)
        coords_all = cached["anchor_coords"]  # (N, K, 3, 3)
        types_all = cached["anchor_types"]    # (N, K)
        mask_all = cached["anchor_mask"]      # (N, K)

        N = coords_all.shape[0]
        idx = int(torch.randint(0, N, (1,)).item())

        coords = coords_all[idx].to(dtype=data["pos_heavyatom"].dtype)
        if self.augment_sigma > 0.0:
            coords = coords + self.augment_sigma * torch.randn_like(coords)

        data["anchor_coords"] = coords
        data["anchor_types"] = types_all[idx].long()
        data["anchor_mask"] = mask_all[idx].bool()
        # hotspot_labels and anchor_indices remain GT-derived (for contact_loss).
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"cache_path={self.cache_path!r}, "
            f"num_anchors={self.num_anchors}, "
            f"augment_sigma={self.augment_sigma})"
        )
