from __future__ import annotations

from typing import Sequence

from hotflow.models.flow_generator import HotspotConditionedFlowGenerator
from hotflow.models.hotspot_sampler import PepHARHotspotSampler
from hotflow.sampling.inpainting import anchors_to_condition
from hotflow.data_types import HotFlowCondition, HotspotAnchor


class HotFlowModel:
    """Stage-1 hotspot sampling + stage-2 flow generation orchestration."""

    def __init__(
        self,
        flow_generator: HotspotConditionedFlowGenerator,
        hotspot_sampler: PepHARHotspotSampler | None = None,
    ):
        self.flow_generator = flow_generator
        self.hotspot_sampler = hotspot_sampler

    def sample_hotspots(
        self,
        pephar_example: dict,
        num_anchors: int = 1,
        anchor_steps: int = 100,
        anchor_strategy: str = "ebm",
        verbose: bool = False,
    ) -> Sequence[HotspotAnchor]:
        if self.hotspot_sampler is None:
            raise ValueError("hotspot_sampler is not configured")
        return self.hotspot_sampler.sample(
            pephar_example,
            num_anchors=num_anchors,
            anchor_steps=anchor_steps,
            anchor_strategy=anchor_strategy,
            verbose=verbose,
        )

    def build_condition(
        self,
        pepflow_batch: dict,
        anchors: Sequence[HotspotAnchor],
        *,
        freeze_backbone: bool = True,
        freeze_sequence: bool = True,
        freeze_torsions: bool = True,
    ) -> HotFlowCondition:
        condition = anchors_to_condition(pepflow_batch, anchors)
        condition.freeze_backbone = freeze_backbone
        condition.freeze_sequence = freeze_sequence
        condition.freeze_torsions = freeze_torsions
        return condition

    def loss(
        self,
        pepflow_batch: dict,
        condition: HotFlowCondition | None = None,
        anchors: Sequence[HotspotAnchor] | None = None,
    ):
        condition = self._resolve_condition(pepflow_batch, condition, anchors)
        return self.flow_generator.loss(pepflow_batch, condition)

    def sample(
        self,
        pepflow_batch: dict,
        num_steps: int = 100,
        condition: HotFlowCondition | None = None,
        anchors: Sequence[HotspotAnchor] | None = None,
        sample_backbone: bool = True,
        sample_angles: bool = True,
        sample_sequence: bool = True,
    ):
        condition = self._resolve_condition(pepflow_batch, condition, anchors)
        return self.flow_generator.sample(
            pepflow_batch,
            condition,
            num_steps=num_steps,
            sample_backbone=sample_backbone,
            sample_angles=sample_angles,
            sample_sequence=sample_sequence,
        )

    def _resolve_condition(
        self,
        pepflow_batch: dict,
        condition: HotFlowCondition | None,
        anchors: Sequence[HotspotAnchor] | None,
    ) -> HotFlowCondition:
        if condition is not None:
            return condition
        if anchors is not None:
            return self.build_condition(pepflow_batch, anchors)
        raise ValueError("Either condition or anchors must be provided")
