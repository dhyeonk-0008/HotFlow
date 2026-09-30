from __future__ import annotations

from typing import Any

import torch

from hotflow.sampling.inpainting import prepare_inpainting_batch
from hotflow.data_types import HotFlowCondition


class HotspotConditionedFlowGenerator:
    """Adapter that feeds hotspot-conditioned batches into PepFlow."""

    def __init__(self, flow_model: Any, supports_modality_masks: bool = True):
        self.flow_model = flow_model
        self.supports_modality_masks = supports_modality_masks

    def prepare_batch(
        self,
        batch: dict[str, object],
        condition: HotFlowCondition,
    ) -> dict[str, object]:
        return prepare_inpainting_batch(batch, condition)

    def loss(
        self,
        batch: dict[str, object],
        condition: HotFlowCondition,
    ) -> Any:
        if not condition.freeze_torsions and not self.supports_modality_masks:
            raise NotImplementedError(
                "Current PepFlow uses one generate_mask for structure, sequence, "
                "and torsions. Set freeze_torsions=True for A0, or patch PepFlow "
                "to support modality-specific masks."
            )
        prepared = self.prepare_batch(batch, condition)
        return self.flow_model(prepared)

    @torch.no_grad()
    def sample(
        self,
        batch: dict[str, object],
        condition: HotFlowCondition,
        num_steps: int = 100,
        sample_backbone: bool = True,
        sample_angles: bool = True,
        sample_sequence: bool = True,
    ) -> Any:
        if not condition.freeze_torsions and not self.supports_modality_masks:
            raise NotImplementedError(
                "Current PepFlow cannot keep hotspot backbone fixed while still "
                "sampling hotspot torsions. Start with freeze_torsions=True or "
                "extend FlowModel with modality-specific masks."
            )
        prepared = self.prepare_batch(batch, condition)
        return self.flow_model.sample(
            prepared,
            num_steps=num_steps,
            sample_bb=sample_backbone,
            sample_ang=sample_angles,
            sample_seq=sample_sequence,
        )
