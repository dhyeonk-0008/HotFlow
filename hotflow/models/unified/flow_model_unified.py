"""UnifiedFlowModel: anchor nodes as first-class members of the IPA graph.

Instead of a separate HotspotEncoder + cross-attention conditioning, anchor
nodes are appended to the residue graph and processed by the vanilla PepFlow
GAEncoder.  The only new learnable parameter is a 3-class node type embedding
(receptor=0, peptide=1, anchor=2) added to the node features.

Training path (forward):
    1. Batch already contains L+K residues (from AnchorToGraphTransform).
    2. Encode: NodeEmbedder + EdgeEmbedder + node_type_embed → standard IPA.
    3. Flow matching: noise/denoise on generate_mask=True (peptide) positions.
    4. Loss: same as FlowModel + contact_preservation_loss.

Inference path (sample):
    1. Batch with L+K residues (receptor + noised peptide + anchor nodes).
    2. Denoising loop via GAEncoder — anchors participate as fixed context.
    3. Peptide structures emerge from the denoising process.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
PEPFLOW_ROOT = REPO_ROOT / "PepFlowww"
if str(PEPFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(PEPFLOW_ROOT))

from models_con.flow_model import FlowModel  # noqa: E402
from pepflow.modules.common.layers import sample_from as _sample_from_raw  # noqa: E402
from pepflow.modules.so3.dist import uniform_so3  # noqa: E402

from data import so3_utils  # noqa: E402
from data import all_atom  # noqa: E402
from models_con.torsion import torsions_mask  # noqa: E402
import models_con.torus as torus  # noqa: E402

from hotflow.data.hotspot_labeling import label_hotspots_from_batch  # noqa: E402
from hotflow.losses.contact_preservation import contact_preservation_loss  # noqa: E402


def _safe_softmax_sample(logits):
    """Softmax + nan guard + renormalize + sample."""
    logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    prob = F.softmax(logits, dim=-1)
    prob = torch.nan_to_num(prob, nan=1e-8, posinf=1e-8, neginf=1e-8)
    prob = torch.clamp(prob, min=1e-8)
    prob = prob / prob.sum(dim=-1, keepdim=True)
    return _sample_from_raw(prob)


def sample_from(prob):
    """Safe wrapper: nan guard + renormalize before multinomial."""
    prob = torch.nan_to_num(prob, nan=1e-8, posinf=1e-8, neginf=1e-8)
    prob = torch.clamp(prob, min=1e-8)
    prob = prob / prob.sum(dim=-1, keepdim=True)
    return _sample_from_raw(prob)


class UnifiedFlowModel(FlowModel):
    """FlowModel with anchor nodes in the IPA graph.

    Anchor nodes are appended to the residue sequence by
    ``AnchorToGraphTransform`` before this model sees the batch.  They
    have ``generate_mask=False`` (not generated) and participate in IPA
    as fixed context nodes — same treatment as receptor residues.

    The only new learnable component is a 3-class node type embedding
    that distinguishes receptor (0), peptide (1), and anchor (2) nodes.

    Args:
        cfg: model config (same format as PepFlow's learn_angle.yaml).
        hotspot_dropout_p: probability of masking anchor features during
            training (for classifier-free guidance at inference).
        contact_distance_cutoff: distance threshold for hotspot labeling.
        contact_loss_target: target distance for contact preservation.
        contact_loss_margin: margin for contact preservation hinge.
    """

    def __init__(
        self,
        cfg,
        hotspot_dropout_p: float = 0.1,
        contact_distance_cutoff: float = 4.0,
        contact_loss_target: float = 4.0,
        contact_loss_margin: float = 2.0,
    ):
        super().__init__(cfg)

        # Node type embedding: receptor=0, peptide=1, anchor=2
        self.node_type_embed = nn.Embedding(3, cfg.encoder.node_embed_size)

        self.hotspot_dropout_p = hotspot_dropout_p
        self.contact_distance_cutoff = contact_distance_cutoff
        self.contact_loss_target = contact_loss_target
        self.contact_loss_margin = contact_loss_margin

    def encode(self, batch):
        """Encode batch with node type embedding added to node features."""
        rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed = (
            super().encode(batch)
        )
        # Add node type information
        if "node_type" in batch:
            node_embed = node_embed + self.node_type_embed(batch["node_type"])
        return rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed

    def _apply_anchor_dropout(self, batch):
        """Mask anchor features for hotspot dropout (classifier-free guidance).

        Returns a modified batch where anchor nodes become invisible
        (node_type set to receptor, mask_heavyatom zeroed, aa set to UNK).
        """
        batch = {k: v.clone() if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        if "node_type" not in batch:
            return batch
        anchor_mask = batch["node_type"] == 2  # (B, L+K)
        batch["node_type"] = torch.where(
            anchor_mask,
            torch.zeros_like(batch["node_type"]),
            batch["node_type"],
        )
        batch["mask_heavyatom"] = batch["mask_heavyatom"].clone()
        batch["mask_heavyatom"][anchor_mask] = False
        batch["aa"] = batch["aa"].clone()
        batch["aa"][anchor_mask] = 20  # UNK
        return batch

    def forward(self, batch):
        """Training forward pass with anchor nodes in the graph.

        Returns loss dict with all base PepFlow losses + contact_loss.
        """
        num_batch, num_res = batch["aa"].shape
        backbone_generate_mask, sequence_generate_mask, torsion_generate_mask = (
            self._get_modality_generate_masks(batch)
        )
        backbone_gen_mask = backbone_generate_mask.long()
        sequence_gen_mask = sequence_generate_mask.long()
        res_mask = batch["res_mask"].long()

        # Hotspot dropout: mask anchor features with probability p
        # Under DDP, the drop decision must be identical across ranks.
        encode_batch = batch
        if self.training:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                decision = torch.zeros(1, device=batch["aa"].device)
                if torch.distributed.get_rank() == 0:
                    decision[0] = float(
                        torch.rand(1).item() < self.hotspot_dropout_p
                    )
                torch.distributed.broadcast(decision, src=0)
                drop = bool(decision.item())
            else:
                drop = torch.rand(1).item() < self.hotspot_dropout_p
            if drop:
                encode_batch = self._apply_anchor_dropout(batch)

        # Encode (with node_type_embed)
        rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed = (
            self.encode(encode_batch)
        )
        trans_1_c = trans_1
        seqs_1_simplex = self.seq_to_simplex(seqs_1)
        seqs_1_prob = F.softmax(seqs_1_simplex, dim=-1)

        # Corrupt (same as FlowModel — only generate_mask=True positions)
        with torch.no_grad():
            t = torch.rand((num_batch, 1), device=batch["aa"].device)
            t = (
                t * (1 - 2 * self._interpolant_cfg.min_t)
                + self._interpolant_cfg.min_t
            )

            if self.sample_structure:
                trans_0 = (
                    torch.randn((num_batch, num_res, 3), device=batch["aa"].device)
                    * self._interpolant_cfg.trans.sigma
                )
                trans_0_c, _ = self.zero_center_part(
                    trans_0, backbone_gen_mask, res_mask
                )
                trans_t = (1 - t[..., None]) * trans_0_c + t[..., None] * trans_1_c
                trans_t_c = torch.where(
                    backbone_generate_mask[..., None], trans_t, trans_1_c
                )
                rotmats_0 = uniform_so3(
                    num_batch, num_res, device=batch["aa"].device
                )
                rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
                rotmats_t = torch.where(
                    backbone_generate_mask[..., None, None], rotmats_t, rotmats_1
                )
                angles_0 = torus.tor_random_uniform(
                    angles_1.shape, device=batch["aa"].device, dtype=angles_1.dtype
                )
                angles_t = torus.tor_geodesic_t(t[..., None], angles_1, angles_0)
                angles_t = torch.where(
                    torsion_generate_mask[..., None], angles_t, angles_1
                )
            else:
                trans_t_c = trans_1_c.detach().clone()
                rotmats_t = rotmats_1.detach().clone()
                angles_t = angles_1.detach().clone()

            if self.sample_sequence:
                seqs_0_simplex = self.k * torch.randn_like(seqs_1_simplex)
                seqs_t_simplex = (
                    (1 - t[..., None]) * seqs_0_simplex
                    + t[..., None] * seqs_1_simplex
                )
                seqs_t_simplex = torch.where(
                    sequence_generate_mask[..., None],
                    seqs_t_simplex,
                    seqs_1_simplex,
                )
                seqs_t_prob = F.softmax(seqs_t_simplex, dim=-1)
                seqs_t_prob = torch.clamp(seqs_t_prob, min=1e-8)
                seqs_t_prob = seqs_t_prob / seqs_t_prob.sum(dim=-1, keepdim=True)
                seqs_t = sample_from(seqs_t_prob)
                seqs_t = torch.where(sequence_generate_mask, seqs_t, seqs_1)
            else:
                seqs_t = seqs_1.detach().clone()
                seqs_t_simplex = seqs_1_simplex.detach().clone()
                seqs_t_prob = seqs_1_prob.detach().clone()

        # Denoise via vanilla GAEncoder (no cross-attention)
        pred_rotmats_1, pred_trans_1, pred_angles_1, pred_seqs_1_prob = (
            self.ga_encoder(
                t, rotmats_t, trans_t_c, angles_t, seqs_t,
                node_embed, edge_embed, backbone_gen_mask, res_mask,
            )
        )

        # NaN guard (DDP-safe: zero loss connected to all params)
        if (
            torch.isnan(pred_trans_1).any()
            or torch.isnan(pred_rotmats_1).any()
            or torch.isnan(pred_seqs_1_prob).any()
            or torch.isnan(pred_angles_1).any()
        ):
            zero = sum(p.sum() for p in self.parameters() if p.requires_grad) * 0.0
            return {
                k: zero
                for k in [
                    "trans_loss", "rot_loss", "bb_atom_loss",
                    "seqs_loss", "angle_loss", "torsion_loss", "contact_loss",
                ]
            }

        pred_seqs_1 = _safe_softmax_sample(pred_seqs_1_prob)
        pred_seqs_1 = torch.where(
            sequence_generate_mask, pred_seqs_1, torch.clamp(seqs_1, 0, 19)
        )
        pred_trans_1_c = pred_trans_1

        norm_scale = 1 / (
            1 - torch.min(
                t[..., None],
                torch.tensor(self._interpolant_cfg.t_normalization_clip),
            )
        )

        # --- Losses (same as FlowModel) ---
        trans_loss = torch.sum(
            (pred_trans_1_c - trans_1_c) ** 2
            * backbone_gen_mask[..., None],
            dim=(-1, -2),
        ) / (torch.sum(backbone_gen_mask, dim=-1) + 1e-8)
        trans_loss = torch.mean(trans_loss)

        gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, rotmats_1)
        pred_rot_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
        rot_loss = torch.sum(
            ((gt_rot_vf - pred_rot_vf) * norm_scale) ** 2
            * backbone_gen_mask[..., None],
            dim=(-1, -2),
        ) / (torch.sum(backbone_gen_mask, dim=-1) + 1e-8)
        rot_loss = torch.mean(rot_loss)

        gt_bb_atoms = all_atom.to_atom37(trans_1_c, rotmats_1)[:, :, :3]
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1_c, pred_rotmats_1)[
            :, :, :3
        ]
        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2
            * backbone_gen_mask[..., None, None],
            dim=(-1, -2, -3),
        ) / (torch.sum(backbone_gen_mask, dim=-1) + 1e-8)
        bb_atom_loss = torch.mean(bb_atom_loss)

        seqs_loss = F.cross_entropy(
            pred_seqs_1_prob.view(-1, pred_seqs_1_prob.shape[-1]),
            torch.clamp(seqs_1, 0, 19).view(-1),
            reduction="none",
        ).view(pred_seqs_1_prob.shape[:-1])
        seqs_loss = torch.sum(seqs_loss * sequence_gen_mask, dim=-1) / (
            torch.sum(sequence_gen_mask, dim=-1) + 1e-8
        )
        seqs_loss = torch.mean(seqs_loss)

        angle_mask_loss = torsions_mask.to(batch["aa"].device)
        angle_mask_loss = angle_mask_loss[pred_seqs_1.reshape(-1)].reshape(
            num_batch, num_res, -1
        )
        angle_mask_loss = torch.cat([angle_mask_loss, angle_mask_loss], dim=-1)
        angle_mask_loss = torch.logical_and(
            torsion_generate_mask[..., None], angle_mask_loss
        )
        gt_angle_vf = torus.tor_logmap(angles_t, angles_1)
        gt_angle_vf_vec = torch.cat(
            [torch.sin(gt_angle_vf), torch.cos(gt_angle_vf)], dim=-1
        )
        pred_angle_vf = torus.tor_logmap(angles_t, pred_angles_1)
        pred_angle_vf_vec = torch.cat(
            [torch.sin(pred_angle_vf), torch.cos(pred_angle_vf)], dim=-1
        )
        angle_loss = torch.sum(
            ((gt_angle_vf_vec - pred_angle_vf_vec) * norm_scale) ** 2
            * angle_mask_loss,
            dim=(-1, -2),
        ) / (torch.sum(angle_mask_loss, dim=(-1, -2)) + 1e-8)
        angle_loss = torch.mean(angle_loss)

        angles_1_vec = torch.cat(
            [torch.sin(angles_1), torch.cos(angles_1)], dim=-1
        )
        pred_angles_1_vec = torch.cat(
            [torch.sin(pred_angles_1), torch.cos(pred_angles_1)], dim=-1
        )
        torsion_loss = torch.sum(
            (pred_angles_1_vec - angles_1_vec) ** 2 * angle_mask_loss,
            dim=(-1, -2),
        ) / (torch.sum(angle_mask_loss, dim=(-1, -2)) + 1e-8)
        torsion_loss = torch.mean(torsion_loss)

        # --- Contact preservation loss ---
        # Compute on original L positions only (exclude appended anchors)
        # so that anchors don't count as "receptor" in the distance calculation.
        L_orig = batch.get("L_orig", None)
        if L_orig is not None:
            L_orig_val = int(L_orig[0].item()) if L_orig.dim() > 0 else int(L_orig.item())
            batch_orig = {
                "generate_mask": batch["generate_mask"][:, :L_orig_val],
                "res_mask": batch["res_mask"][:, :L_orig_val],
                "pos_heavyatom": batch["pos_heavyatom"][:, :L_orig_val],
            }
            hotspot_labels = batch.get("hotspot_labels")
            if hotspot_labels is not None:
                hotspot_labels = hotspot_labels[:, :L_orig_val]
            else:
                hotspot_labels = label_hotspots_from_batch(
                    batch_orig, distance_cutoff=self.contact_distance_cutoff
                )
            contact_loss = contact_preservation_loss(
                pred_trans_1_c[:, :L_orig_val],
                batch_orig,
                hotspot_labels,
                target_distance=self.contact_loss_target,
                margin=self.contact_loss_margin,
            )
        else:
            hotspot_labels = batch.get(
                "hotspot_labels",
                label_hotspots_from_batch(
                    batch, distance_cutoff=self.contact_distance_cutoff
                ),
            )
            contact_loss = contact_preservation_loss(
                pred_trans_1_c, batch, hotspot_labels,
                target_distance=self.contact_loss_target,
                margin=self.contact_loss_margin,
            )

        return {
            "trans_loss": trans_loss,
            "rot_loss": rot_loss,
            "bb_atom_loss": bb_atom_loss,
            "seqs_loss": seqs_loss,
            "angle_loss": angle_loss,
            "torsion_loss": torsion_loss,
            "contact_loss": contact_loss,
        }

    @torch.no_grad()
    def sample(
        self,
        batch,
        num_steps: int = 100,
        sample_bb: bool = True,
        sample_ang: bool = True,
        sample_seq: bool = True,
        guidance_scale: float = 1.0,
    ):
        """Sample with anchor nodes as fixed context in the IPA graph.

        No explicit anchor_coords/types/mask arguments needed — anchors
        are already part of the batch from AnchorToGraphTransform.

        Args:
            batch: batch dict with L+K residues.
            num_steps: number of denoising steps.
            sample_bb/sample_ang/sample_seq: what modalities to sample.
            guidance_scale: classifier-free guidance scale. 1.0 = no guidance.

        Returns:
            clean_traj: list of dicts with predicted structures at each step.
        """
        num_batch, num_res = batch["aa"].shape
        backbone_generate_mask, sequence_generate_mask, torsion_generate_mask = (
            self._get_modality_generate_masks(batch)
        )
        backbone_gen_mask, res_mask = backbone_generate_mask, batch["res_mask"]
        K = self._interpolant_cfg.seqs.num_classes
        k = self._interpolant_cfg.seqs.simplex_value
        angle_mask_loss = torsions_mask.to(batch["aa"].device)

        # Encode (conditional)
        rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed = (
            self.encode(batch)
        )
        trans_1_c = trans_1
        seqs_1_simplex = self.seq_to_simplex(seqs_1)
        seqs_1_prob = F.softmax(seqs_1_simplex, dim=-1)

        # Unconditional embeddings for CFG
        use_cfg = guidance_scale > 1.0
        if use_cfg:
            uncond_batch = self._apply_anchor_dropout(batch)
            _, _, _, _, node_embed_uncond, edge_embed_uncond = (
                self.encode(uncond_batch)
            )

        # Initial noise
        if sample_bb:
            rotmats_0 = uniform_so3(num_batch, num_res, device=batch["aa"].device)
            rotmats_0 = torch.where(
                backbone_generate_mask[..., None, None], rotmats_0, rotmats_1
            )
            trans_0 = torch.randn(
                (num_batch, num_res, 3), device=batch["aa"].device
            )
            trans_0_c, center = self.zero_center_part(
                trans_0, backbone_gen_mask, res_mask
            )
            trans_0_c = torch.where(
                backbone_generate_mask[..., None], trans_0_c, trans_1_c
            )
        else:
            rotmats_0 = rotmats_1.detach().clone()
            trans_0_c = trans_1_c.detach().clone()
        if sample_ang:
            angles_0 = torus.tor_random_uniform(
                angles_1.shape, device=batch["aa"].device, dtype=angles_1.dtype
            )
            angles_0 = torch.where(
                torsion_generate_mask[..., None], angles_0, angles_1
            )
        else:
            angles_0 = angles_1.detach().clone()
        if sample_seq:
            seqs_0_simplex = k * torch.randn(
                (num_batch, num_res, K), device=batch["aa"].device
            )
            seqs_0_prob = F.softmax(seqs_0_simplex, dim=-1)
            seqs_0 = sample_from(seqs_0_prob)
            seqs_0 = torch.where(sequence_generate_mask, seqs_0, seqs_1)
            seqs_0_simplex = torch.where(
                sequence_generate_mask[..., None], seqs_0_simplex, seqs_1_simplex
            )
        else:
            seqs_0 = seqs_1.detach().clone()
            seqs_0_prob = seqs_1_prob.detach().clone()
            seqs_0_simplex = seqs_1_simplex.detach().clone()

        ts = torch.linspace(1e-2, 1.0, num_steps)
        t_1 = ts[0]
        clean_traj = []
        rotmats_t_1, trans_t_1_c, angles_t_1 = rotmats_0, trans_0_c, angles_0
        seqs_t_1, seqs_t_1_simplex = seqs_0, seqs_0_simplex

        def _denoise(t_tensor, rots, trans, angs, seqs, ne, ee):
            return self.ga_encoder(
                t_tensor, rots, trans, angs, seqs, ne, ee,
                backbone_generate_mask.long(), batch["res_mask"].long(),
            )

        def _cfg_denoise(t_tensor, rots, trans, angs, seqs):
            cond = _denoise(t_tensor, rots, trans, angs, seqs, node_embed, edge_embed)
            if not use_cfg:
                return cond
            uncond = _denoise(
                t_tensor, rots, trans, angs, seqs,
                node_embed_uncond, edge_embed_uncond,
            )
            return tuple(u + guidance_scale * (c - u) for c, u in zip(cond, uncond))

        # Denoise loop
        for t_2 in ts[1:]:
            t = torch.ones((num_batch, 1), device=batch["aa"].device) * t_1

            pred_rotmats_1, pred_trans_1, pred_angles_1, pred_seqs_1_prob = (
                _cfg_denoise(t, rotmats_t_1, trans_t_1_c, angles_t_1, seqs_t_1)
            )
            pred_rotmats_1 = torch.where(
                backbone_generate_mask[..., None, None], pred_rotmats_1, rotmats_1
            )
            pred_trans_1_c = torch.where(
                backbone_generate_mask[..., None], pred_trans_1, trans_1_c
            )
            pred_angles_1 = torch.where(
                torsion_generate_mask[..., None], pred_angles_1, angles_1
            )
            pred_seqs_1 = _safe_softmax_sample(pred_seqs_1_prob)
            pred_seqs_1 = torch.where(sequence_generate_mask, pred_seqs_1, seqs_1)
            pred_seqs_1_simplex = self.seq_to_simplex(pred_seqs_1)

            torsion_mask = angle_mask_loss[pred_seqs_1.reshape(-1)].reshape(
                num_batch, num_res, -1
            )
            pred_angles_1 = torch.where(
                torsion_mask.bool(), pred_angles_1, torch.zeros_like(pred_angles_1)
            )

            if not sample_bb:
                pred_trans_1_c = trans_1_c.detach().clone()
                pred_rotmats_1 = rotmats_1.detach().clone()
            if not sample_ang:
                pred_angles_1 = angles_1.detach().clone()
            if not sample_seq:
                pred_seqs_1 = seqs_1.detach().clone()
                pred_seqs_1_simplex = seqs_1_simplex.detach().clone()

            clean_traj.append({
                "rotmats": pred_rotmats_1.cpu(),
                "trans": pred_trans_1_c.cpu(),
                "angles": pred_angles_1.cpu(),
                "seqs": pred_seqs_1.cpu(),
                "seqs_simplex": pred_seqs_1_simplex.cpu(),
                "rotmats_1": rotmats_1.cpu(),
                "trans_1": trans_1_c.cpu(),
                "angles_1": angles_1.cpu(),
                "seqs_1": seqs_1.cpu(),
            })

            # Euler step
            d_t = (t_2 - t_1) * torch.ones(
                (num_batch, 1), device=batch["aa"].device
            )
            trans_t_2 = trans_t_1_c + (pred_trans_1_c - trans_0_c) * d_t[..., None]
            trans_t_2_c = torch.where(
                backbone_generate_mask[..., None], trans_t_2, trans_1_c
            )
            rotmats_t_2 = so3_utils.geodesic_t(
                d_t[..., None] * 10, pred_rotmats_1, rotmats_t_1
            )
            rotmats_t_2 = torch.where(
                backbone_generate_mask[..., None, None], rotmats_t_2, rotmats_1
            )
            angles_t_2 = torus.tor_geodesic_t(
                d_t[..., None], pred_angles_1, angles_t_1
            )
            angles_t_2 = torch.where(
                torsion_generate_mask[..., None], angles_t_2, angles_1
            )
            seqs_t_2_simplex = (
                seqs_t_1_simplex
                + (pred_seqs_1_simplex - seqs_0_simplex) * d_t[..., None]
            )
            seqs_t_2 = sample_from(F.softmax(seqs_t_2_simplex, dim=-1))
            seqs_t_2 = torch.where(sequence_generate_mask, seqs_t_2, seqs_1)

            torsion_mask = angle_mask_loss[seqs_t_2.reshape(-1)].reshape(
                num_batch, num_res, -1
            )
            angles_t_2 = torch.where(
                torsion_mask.bool(), angles_t_2, torch.zeros_like(angles_t_2)
            )

            if not sample_bb:
                trans_t_2_c = trans_1_c.detach().clone()
                rotmats_t_2 = rotmats_1.detach().clone()
            if not sample_ang:
                angles_t_2 = angles_1.detach().clone()
            if not sample_seq:
                seqs_t_2 = seqs_1.detach().clone()

            rotmats_t_1, trans_t_1_c, angles_t_1 = rotmats_t_2, trans_t_2_c, angles_t_2
            seqs_t_1, seqs_t_1_simplex = seqs_t_2, seqs_t_2_simplex
            t_1 = t_2

        # Final step
        t_1 = ts[-1]
        t = torch.ones((num_batch, 1), device=batch["aa"].device) * t_1
        pred_rotmats_1, pred_trans_1, pred_angles_1, pred_seqs_1_prob = (
            _cfg_denoise(t, rotmats_t_1, trans_t_1_c, angles_t_1, seqs_t_1)
        )
        pred_rotmats_1 = torch.where(
            backbone_generate_mask[..., None, None], pred_rotmats_1, rotmats_1
        )
        pred_trans_1_c = torch.where(
            backbone_generate_mask[..., None], pred_trans_1, trans_1_c
        )
        pred_angles_1 = torch.where(
            torsion_generate_mask[..., None], pred_angles_1, angles_1
        )
        pred_seqs_1 = _safe_softmax_sample(pred_seqs_1_prob)
        pred_seqs_1 = torch.where(sequence_generate_mask, pred_seqs_1, seqs_1)
        pred_seqs_1_simplex = self.seq_to_simplex(pred_seqs_1)

        torsion_mask = angle_mask_loss[pred_seqs_1.reshape(-1)].reshape(
            num_batch, num_res, -1
        )
        pred_angles_1 = torch.where(
            torsion_mask.bool(), pred_angles_1, torch.zeros_like(pred_angles_1)
        )

        if not sample_bb:
            pred_trans_1_c = trans_1_c.detach().clone()
            pred_rotmats_1 = rotmats_1.detach().clone()
        if not sample_ang:
            pred_angles_1 = angles_1.detach().clone()
        if not sample_seq:
            pred_seqs_1 = seqs_1.detach().clone()
            pred_seqs_1_simplex = seqs_1_simplex.detach().clone()

        clean_traj.append({
            "rotmats": pred_rotmats_1.cpu(),
            "trans": pred_trans_1_c.cpu(),
            "angles": pred_angles_1.cpu(),
            "seqs": pred_seqs_1.cpu(),
            "seqs_simplex": pred_seqs_1_simplex.cpu(),
            "rotmats_1": rotmats_1.cpu(),
            "trans_1": trans_1_c.cpu(),
            "angles_1": angles_1.cpu(),
            "seqs_1": seqs_1.cpu(),
        })

        return clean_traj

    @classmethod
    def from_pretrained(
        cls,
        cfg,
        checkpoint_path: str,
        strict: bool = False,
        **kwargs,
    ) -> "UnifiedFlowModel":
        """Load from a PepFlow checkpoint, initializing new params randomly.

        Args:
            cfg: model config.
            checkpoint_path: path to PepFlow checkpoint (.pt file).
            strict: if False (default), missing keys (node_type_embed)
                are initialized randomly; unexpected keys are ignored.
            **kwargs: extra args passed to UnifiedFlowModel.__init__.

        Returns:
            UnifiedFlowModel instance with pretrained base weights loaded.
        """
        model = cls(cfg, **kwargs)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model"] if "model" in ckpt else ckpt

        cleaned = {}
        for k, v in state_dict.items():
            key = k.replace("module.", "") if k.startswith("module.") else k
            cleaned[key] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=strict)
        return model
