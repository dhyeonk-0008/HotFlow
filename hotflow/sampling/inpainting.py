from __future__ import annotations

from typing import Sequence

import torch

from hotflow.data_types import HotFlowCondition, HotspotAnchor


N_ATOM_INDEX = 0
CA_ATOM_INDEX = 1
C_ATOM_INDEX = 2
UNKNOWN_AA_INDEX = 20


def _clone_batch(batch: dict[str, object]) -> dict[str, object]:
    cloned: dict[str, object] = {}
    for key, value in batch.items():
        cloned[key] = value.clone() if torch.is_tensor(value) else value
    return cloned


def compute_effective_generate_mask(
    generate_mask: torch.Tensor,
    hotspot_mask: torch.Tensor,
) -> torch.Tensor:
    return generate_mask.bool() & ~hotspot_mask.bool()


def anchors_to_condition(
    batch: dict[str, torch.Tensor],
    anchors: Sequence[HotspotAnchor],
) -> HotFlowCondition:
    hotspot_mask = torch.zeros_like(batch["generate_mask"], dtype=torch.bool)
    residue_types = torch.full_like(batch["aa"], fill_value=UNKNOWN_AA_INDEX)
    backbone_coords = torch.zeros(
        (*batch["aa"].shape, 3, 3),
        dtype=batch["pos_heavyatom"].dtype,
        device=batch["pos_heavyatom"].device,
    )
    torsion_angles = batch["torsion_angle"].clone() if "torsion_angle" in batch else None
    torsion_mask = batch["torsion_angle_mask"].clone() if "torsion_angle_mask" in batch else None

    peptide_positions = torch.nonzero(batch["generate_mask"][0], as_tuple=False).squeeze(-1)
    if not torch.all(batch["generate_mask"] == batch["generate_mask"][0]):
        raise ValueError("anchors_to_condition expects the same generate_mask across the batch")
    seen_residue_indices: set[int] = set()
    for anchor in anchors:
        anchor.validate()
        if anchor.residue_index in seen_residue_indices:
            continue
        if anchor.residue_index >= peptide_positions.numel():
            raise IndexError(
                f"anchor residue_index {anchor.residue_index} exceeds peptide length "
                f"{peptide_positions.numel()}"
            )
        seen_residue_indices.add(anchor.residue_index)
        global_index = int(peptide_positions[anchor.residue_index].item())
        hotspot_mask[:, global_index] = True
        residue_types[:, global_index] = int(anchor.residue_type)
        backbone_coords[:, global_index] = anchor.backbone_coords.to(
            device=backbone_coords.device,
            dtype=backbone_coords.dtype,
        )

    return HotFlowCondition(
        hotspot_mask=hotspot_mask,
        residue_types=residue_types,
        backbone_coords=backbone_coords,
        torsion_angles=torsion_angles,
        torsion_mask=torsion_mask,
    )


def prepare_inpainting_batch(
    batch: dict[str, object],
    condition: HotFlowCondition,
) -> dict[str, object]:
    if "generate_mask" not in batch or "aa" not in batch:
        raise KeyError("batch must contain at least 'generate_mask' and 'aa'")

    tensor_batch = {k: v for k, v in batch.items() if torch.is_tensor(v)}
    condition.validate(tensor_batch)

    prepared = _clone_batch(batch)
    original_generate_mask = prepared["generate_mask"]
    assert torch.is_tensor(original_generate_mask)

    hotspot_mask = condition.hotspot_mask.bool()
    backbone_generate_mask = (
        compute_effective_generate_mask(original_generate_mask, hotspot_mask)
        if condition.freeze_backbone
        else original_generate_mask.bool().clone()
    )
    sequence_generate_mask = (
        compute_effective_generate_mask(original_generate_mask, hotspot_mask)
        if condition.freeze_sequence
        else original_generate_mask.bool().clone()
    )
    torsion_generate_mask = (
        compute_effective_generate_mask(original_generate_mask, hotspot_mask)
        if condition.freeze_torsions
        else original_generate_mask.bool().clone()
    )

    prepared["original_generate_mask"] = original_generate_mask.clone()
    prepared["hotspot_mask"] = hotspot_mask.clone()
    prepared["backbone_generate_mask"] = backbone_generate_mask
    prepared["sequence_generate_mask"] = sequence_generate_mask
    prepared["torsion_generate_mask"] = torsion_generate_mask
    prepared["generate_mask"] = backbone_generate_mask.clone()

    if condition.freeze_sequence and condition.residue_types is not None:
        aa = prepared["aa"]
        assert torch.is_tensor(aa)
        prepared["aa"] = torch.where(
            hotspot_mask,
            condition.residue_types.to(device=aa.device, dtype=aa.dtype),
            aa,
        )

    if condition.freeze_backbone and condition.backbone_coords is not None:
        pos_heavyatom = prepared["pos_heavyatom"]
        assert torch.is_tensor(pos_heavyatom)
        coords = condition.backbone_coords.to(
            device=pos_heavyatom.device,
            dtype=pos_heavyatom.dtype,
        )
        pos_heavyatom = pos_heavyatom.clone()
        hotspot_coord_mask = hotspot_mask.unsqueeze(-1).unsqueeze(-1)
        pos_heavyatom = torch.where(hotspot_coord_mask, torch.zeros_like(pos_heavyatom), pos_heavyatom)
        pos_heavyatom[:, :, CA_ATOM_INDEX] = torch.where(
            hotspot_mask.unsqueeze(-1),
            coords[:, :, 0],
            pos_heavyatom[:, :, CA_ATOM_INDEX],
        )
        pos_heavyatom[:, :, C_ATOM_INDEX] = torch.where(
            hotspot_mask.unsqueeze(-1),
            coords[:, :, 1],
            pos_heavyatom[:, :, C_ATOM_INDEX],
        )
        pos_heavyatom[:, :, N_ATOM_INDEX] = torch.where(
            hotspot_mask.unsqueeze(-1),
            coords[:, :, 2],
            pos_heavyatom[:, :, N_ATOM_INDEX],
        )
        prepared["pos_heavyatom"] = pos_heavyatom

        if "mask_heavyatom" in prepared:
            mask_heavyatom = prepared["mask_heavyatom"]
            assert torch.is_tensor(mask_heavyatom)
            mask_heavyatom = mask_heavyatom.clone()
            hotspot_atom_mask = hotspot_mask.unsqueeze(-1)
            mask_heavyatom = torch.where(
                hotspot_atom_mask,
                torch.zeros_like(mask_heavyatom),
                mask_heavyatom,
            )
            mask_heavyatom[:, :, N_ATOM_INDEX] = (
                mask_heavyatom[:, :, N_ATOM_INDEX] | hotspot_mask
            )
            mask_heavyatom[:, :, CA_ATOM_INDEX] = (
                mask_heavyatom[:, :, CA_ATOM_INDEX] | hotspot_mask
            )
            mask_heavyatom[:, :, C_ATOM_INDEX] = (
                mask_heavyatom[:, :, C_ATOM_INDEX] | hotspot_mask
            )
            prepared["mask_heavyatom"] = mask_heavyatom

    if condition.freeze_torsions and condition.torsion_angles is not None:
        torsion_angle = prepared["torsion_angle"]
        assert torch.is_tensor(torsion_angle)
        torsion_angle = torch.where(
            hotspot_mask.unsqueeze(-1),
            condition.torsion_angles.to(
                device=torsion_angle.device,
                dtype=torsion_angle.dtype,
            ),
            torsion_angle,
        )
        prepared["torsion_angle"] = torsion_angle

        if condition.torsion_mask is not None and "torsion_angle_mask" in prepared:
            torsion_angle_mask = prepared["torsion_angle_mask"]
            assert torch.is_tensor(torsion_angle_mask)
            torsion_angle_mask = torch.where(
                hotspot_mask.unsqueeze(-1),
                condition.torsion_mask.to(
                    device=torsion_angle_mask.device,
                    dtype=torsion_angle_mask.dtype,
                ),
                torsion_angle_mask,
            )
            prepared["torsion_angle_mask"] = torsion_angle_mask
    return prepared
