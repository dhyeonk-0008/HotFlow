"""GAEncoder with cross-attention layers for hotspot conditioning (Approach B).

Inherits from PepFlow's GAEncoder and adds cross-attention to hotspot features
in the last N blocks (default: blocks 3, 4, 5 out of 0-5).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
PEPFLOW_ROOT = REPO_ROOT / "PepFlowww"
if str(PEPFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(PEPFLOW_ROOT))

from data import utils as du  # noqa: E402
from models_con.ga import GAEncoder  # noqa: E402


class GAEncoderCrossAttn(GAEncoder):
    """GAEncoder extended with hotspot cross-attention in selected blocks.

    Architecture per selected block (after node_transition, before bb_update):
        cross_attn: MultiheadAttention(query=node_embed, key/value=hotspot_context)
        proj: Linear projection back to c_s
        ln: LayerNorm
        -> simple addition residual connection

    Hotspot dropout: during training, with probability ``hotspot_dropout_p``,
    the hotspot context is dropped entirely (set to None). This enables
    classifier-free guidance at inference time.
    """

    def __init__(
        self,
        ipa_conf,
        d_hotspot: int = 64,
        cross_attn_heads: int = 4,
        cross_attn_blocks: Sequence[int] = (3, 4, 5),
        hotspot_dropout_p: float = 0.1,
    ):
        super().__init__(ipa_conf)
        self.cross_attn_block_set = set(cross_attn_blocks)
        self.hotspot_dropout_p = hotspot_dropout_p

        # Build cross-attention modules only for selected blocks
        self.hotspot_cross_attn = nn.ModuleDict()
        for b in cross_attn_blocks:
            self.hotspot_cross_attn[f"attn_{b}"] = nn.MultiheadAttention(
                embed_dim=ipa_conf.c_s,  # 128 — query dimension
                num_heads=cross_attn_heads,
                kdim=d_hotspot,
                vdim=d_hotspot,
                batch_first=True,
            )
            self.hotspot_cross_attn[f"proj_{b}"] = nn.Linear(
                ipa_conf.c_s, ipa_conf.c_s
            )
            self.hotspot_cross_attn[f"ln_{b}"] = nn.LayerNorm(ipa_conf.c_s)

    def forward(
        self,
        t: torch.Tensor,
        rotmats_t: torch.Tensor,
        trans_t: torch.Tensor,
        angles_t: torch.Tensor,
        seqs_t: torch.Tensor,
        node_embed: torch.Tensor,
        edge_embed: torch.Tensor,
        generate_mask: torch.Tensor,
        res_mask: torch.Tensor,
        hotspot_context: Optional[torch.Tensor] = None,
        hotspot_mask: Optional[torch.Tensor] = None,
    ):
        """Forward pass with optional hotspot cross-attention.

        Args:
            t ... res_mask: identical to GAEncoder.forward().
            hotspot_context: (B, K, d_hotspot) from HotspotEncoder, or None.
            hotspot_mask: (B, K) boolean mask for valid anchors, or None.

        Returns:
            (pred_rotmats_1, pred_trans_1, pred_angles_1, pred_seqs_1_prob)
        """
        num_batch, num_res = seqs_t.shape

        # Hotspot dropout during training. Under DDP, the drop decision MUST
        # be identical across ranks — otherwise one rank skips the cross-attn
        # subgraph and its params receive no grad, while peers do, causing a
        # silent grad-sync mismatch (DDP either hangs or trips
        # find_unused_parameters checks at the next iter).
        if self.training and hotspot_context is not None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                decision = torch.zeros(1, device=hotspot_context.device)
                if torch.distributed.get_rank() == 0:
                    decision[0] = float(torch.rand(1).item() < self.hotspot_dropout_p)
                torch.distributed.broadcast(decision, src=0)
                drop = bool(decision.item())
            else:
                drop = torch.rand(1).item() < self.hotspot_dropout_p
            if drop:
                hotspot_context = None
                hotspot_mask = None

        # --- Mixing (same as parent) ---
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        node_embed = self.res_feat_mixer(
            torch.cat(
                [
                    node_embed,
                    self.current_seq_embedder(seqs_t),
                    self.embed_t(t, node_mask),
                    self.angles_embedder(angles_t).reshape(num_batch, num_res, -1),
                ],
                dim=-1,
            )
        )
        node_embed = node_embed * node_mask[..., None]
        curr_rigids = du.create_rigid(rotmats_t, trans_t)

        # Prepare key_padding_mask for cross-attention (True = ignore)
        cross_attn_key_pad = None
        if hotspot_context is not None and hotspot_mask is not None:
            cross_attn_key_pad = ~hotspot_mask.bool()

        # --- Block loop ---
        for b in range(self._ipa_conf.num_blocks):
            # IPA
            ipa_embed = self.trunk[f"ipa_{b}"](
                node_embed, edge_embed, curr_rigids, node_mask
            )
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f"ipa_ln_{b}"](node_embed + ipa_embed)

            # Sequence transformer
            seq_tfmr_out = self.trunk[f"seq_tfmr_{b}"](
                node_embed, src_key_padding_mask=(1 - node_mask).bool()
            )
            node_embed = node_embed + self.trunk[f"post_tfmr_{b}"](seq_tfmr_out)

            # Node transition
            node_embed = self.trunk[f"node_transition_{b}"](node_embed)
            node_embed = node_embed * node_mask[..., None]

            # --- Hotspot cross-attention (blocks 3, 4, 5) ---
            if b in self.cross_attn_block_set and hotspot_context is not None:
                cross_out, _ = self.hotspot_cross_attn[f"attn_{b}"](
                    query=node_embed,
                    key=hotspot_context,
                    value=hotspot_context,
                    key_padding_mask=cross_attn_key_pad,
                )
                cross_out = self.hotspot_cross_attn[f"proj_{b}"](cross_out)
                node_embed = self.hotspot_cross_attn[f"ln_{b}"](
                    node_embed + cross_out
                )
                node_embed = node_embed * node_mask[..., None]

            # Backbone update
            rigid_update = self.trunk[f"bb_update_{b}"](
                node_embed * node_mask[..., None]
            )
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None]
            )

            # Edge transition (skip on last block)
            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f"edge_transition_{b}"](
                    node_embed, edge_embed
                )
                edge_embed *= edge_mask[..., None]

        # --- Output heads (same as parent) ---
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        pred_angles1 = self.angle_net(node_embed) % (2 * math.pi)

        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob
