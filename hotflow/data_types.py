from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class HotspotAnchor:
    residue_index: int
    backbone_coords: torch.Tensor
    residue_type: int
    score: float | None = None
    source: str = "pephar_density"

    def validate(self) -> None:
        if self.residue_index < 0:
            raise ValueError(f"residue_index must be >= 0, got {self.residue_index}")
        if tuple(self.backbone_coords.shape) != (3, 3):
            raise ValueError(
                "backbone_coords must have shape (3, 3) in CA/C/N order, "
                f"got {tuple(self.backbone_coords.shape)}"
            )
        if not 0 <= int(self.residue_type) <= 20:
            raise ValueError(
                f"residue_type must be in [0, 20], got {self.residue_type}"
            )


@dataclass
class HotFlowCondition:
    hotspot_mask: torch.Tensor
    residue_types: torch.Tensor | None = None
    backbone_coords: torch.Tensor | None = None
    torsion_angles: torch.Tensor | None = None
    torsion_mask: torch.Tensor | None = None
    freeze_backbone: bool = True
    freeze_sequence: bool = True
    freeze_torsions: bool = True

    def validate(self, batch: dict[str, torch.Tensor]) -> None:
        if "generate_mask" not in batch:
            raise KeyError("batch must contain generate_mask")
        if self.hotspot_mask.shape != batch["generate_mask"].shape:
            raise ValueError(
                "hotspot_mask must match batch['generate_mask'] shape, "
                f"got {tuple(self.hotspot_mask.shape)} vs "
                f"{tuple(batch['generate_mask'].shape)}"
            )
        invalid = self.hotspot_mask & ~batch["generate_mask"].bool()
        if torch.any(invalid):
            raise ValueError("hotspot_mask must be a subset of generate_mask")
        if self.residue_types is not None and self.residue_types.shape != batch["aa"].shape:
            raise ValueError(
                "residue_types must match batch['aa'] shape, "
                f"got {tuple(self.residue_types.shape)} vs {tuple(batch['aa'].shape)}"
            )
        if self.backbone_coords is not None:
            expected = (*batch["aa"].shape, 3, 3)
            if tuple(self.backbone_coords.shape) != expected:
                raise ValueError(
                    "backbone_coords must match (*batch['aa'].shape, 3, 3), "
                    f"got {tuple(self.backbone_coords.shape)} vs {expected}"
                )
        if self.torsion_angles is not None and self.torsion_angles.shape != batch["torsion_angle"].shape:
            raise ValueError(
                "torsion_angles must match batch['torsion_angle'] shape, "
                f"got {tuple(self.torsion_angles.shape)} vs "
                f"{tuple(batch['torsion_angle'].shape)}"
            )
        if self.torsion_mask is not None and self.torsion_mask.shape != batch["torsion_angle_mask"].shape:
            raise ValueError(
                "torsion_mask must match batch['torsion_angle_mask'] shape, "
                f"got {tuple(self.torsion_mask.shape)} vs "
                f"{tuple(batch['torsion_angle_mask'].shape)}"
            )
