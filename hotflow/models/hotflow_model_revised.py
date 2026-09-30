"""De novo capable HotFlow orchestration model (Approach A).

Original ``HotFlowModel`` requires GT peptide for hotspot sampling.
This revised version adds ``sample_denovo()`` which uses only receptor
information to generate peptides end-to-end.
"""

from __future__ import annotations

from typing import Sequence

from hotflow.models.flow_generator import HotspotConditionedFlowGenerator
from hotflow.models.hotspot_sampler_revised import PepHARHotspotSamplerDenovo
from hotflow.sampling.inpainting import anchors_to_condition
from hotflow.data_types import HotFlowCondition, HotspotAnchor


class HotFlowModelDenovo:
    """Stage-1 hotspot sampling + stage-2 flow generation with de novo support."""

    def __init__(
        self,
        flow_generator: HotspotConditionedFlowGenerator,
        hotspot_sampler: PepHARHotspotSamplerDenovo | None = None,
    ):
        self.flow_generator = flow_generator
        self.hotspot_sampler = hotspot_sampler

    # ------------------------------------------------------------------
    # Original GT-seeded pipeline (backward compatible)
    # ------------------------------------------------------------------

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
            pepflow_batch, condition,
            num_steps=num_steps,
            sample_backbone=sample_backbone,
            sample_angles=sample_angles,
            sample_sequence=sample_sequence,
        )

    # ------------------------------------------------------------------
    # De novo pipeline — no GT peptide required
    # ------------------------------------------------------------------

    def sample_hotspots_denovo(
        self,
        rec_coord,
        rec_aa,
        pep_length: int,
        num_anchors: int = 1,
        anchor_steps: int = 100,
        verbose: bool = False,
    ) -> Sequence[HotspotAnchor]:
        """Sample hotspot anchors using only receptor information.

        Args:
            rec_coord: (R, 3, 3) receptor backbone coordinates.
            rec_aa: (R,) receptor amino acid types.
            pep_length: desired peptide length.
            num_anchors: number of anchors.
            anchor_steps: EBM optimization steps.
            verbose: print progress.

        Returns:
            List of HotspotAnchor.
        """
        if self.hotspot_sampler is None:
            raise ValueError("hotspot_sampler is not configured")
        return self.hotspot_sampler.sample_denovo(
            rec_coord=rec_coord,
            rec_aa=rec_aa,
            pep_length=pep_length,
            num_anchors=num_anchors,
            anchor_steps=anchor_steps,
            verbose=verbose,
        )

    def sample_denovo(
        self,
        pepflow_batch: dict,
        rec_coord,
        rec_aa,
        pep_length: int,
        num_steps: int = 100,
        num_anchors: int = 1,
        anchor_steps: int = 100,
        sample_backbone: bool = True,
        sample_angles: bool = True,
        sample_sequence: bool = True,
        freeze_backbone: bool = True,
        freeze_sequence: bool = True,
        freeze_torsions: bool = True,
        verbose: bool = False,
    ):
        """End-to-end de novo peptide generation.

        Stage 1: Sample hotspots from receptor surface via EBM.
        Stage 2: Generate peptide structure + sequence via flow matching,
                 conditioned on the sampled hotspots.

        Args:
            pepflow_batch: PepFlow batch dict (receptor pocket info +
                placeholder peptide residues with generate_mask=True).
            rec_coord: (R, 3, 3) receptor backbone coords for PepHAR.
            rec_aa: (R,) receptor amino acid types for PepHAR.
            pep_length: desired peptide length.
            num_steps: denoising steps for flow model.
            num_anchors: number of hotspot anchors.
            anchor_steps: EBM optimization steps per anchor.
            sample_backbone: whether to sample backbone.
            sample_angles: whether to sample torsion angles.
            sample_sequence: whether to sample sequence.
            freeze_backbone: freeze hotspot backbone in flow generation.
            freeze_sequence: freeze hotspot sequence in flow generation.
            freeze_torsions: freeze hotspot torsions in flow generation.
            verbose: print progress.

        Returns:
            Flow model trajectory (list of dicts).
        """
        # Stage 1: de novo hotspot sampling
        anchors = self.sample_hotspots_denovo(
            rec_coord=rec_coord,
            rec_aa=rec_aa,
            pep_length=pep_length,
            num_anchors=num_anchors,
            anchor_steps=anchor_steps,
            verbose=verbose,
        )

        # Stage 2: flow-based generation conditioned on hotspots
        condition = self.build_condition(
            pepflow_batch, anchors,
            freeze_backbone=freeze_backbone,
            freeze_sequence=freeze_sequence,
            freeze_torsions=freeze_torsions,
        )
        return self.flow_generator.sample(
            pepflow_batch, condition,
            num_steps=num_steps,
            sample_backbone=sample_backbone,
            sample_angles=sample_angles,
            sample_sequence=sample_sequence,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

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
