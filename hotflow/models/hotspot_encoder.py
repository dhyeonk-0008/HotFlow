"""Hotspot anchor encoder for Approach B cross-attention conditioning.

Encodes K hotspot anchors into feature vectors that serve as
key/value inputs to the cross-attention layers in GAEncoderCrossAttn.

Supports two coordinate encoding modes:
  - SE(3)-invariant (default): extracts geometric invariants from backbone
    frames and receptor-relative features.
  - Legacy absolute: flattens CA/C/N coordinates directly (original behavior).

Optionally applies anchor self-attention to capture inter-anchor spatial
relationships (e.g., helix spacing constraints).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class HotspotEncoder(nn.Module):
    """Encode each hotspot anchor: coords + residue type -> feature vector.

    Input per anchor:
        - backbone_coords: (3, 3) in CA/C/N order
        - residue_type: int in [0, 20]

    Output: (B, K, d_hotspot) feature tensor.

    Args:
        d_hotspot: output feature dimension (default 64).
        d_type_embed: residue type embedding dimension (default 32).
        use_se3_invariant: if True, extract SE(3)-invariant geometric features
            instead of using raw absolute coordinates.
        use_anchor_self_attn: if True, apply self-attention across anchors
            to model inter-anchor spatial relationships.
        num_self_attn_layers: number of self-attention layers (default 1).
    """

    def __init__(
        self,
        d_hotspot: int = 64,
        d_type_embed: int = 32,
        use_se3_invariant: bool = True,
        use_anchor_self_attn: bool = True,
        num_self_attn_layers: int = 1,
    ):
        super().__init__()
        self.d_hotspot = d_hotspot
        self.use_se3_invariant = use_se3_invariant
        self.use_anchor_self_attn = use_anchor_self_attn

        # Residue type embedding (21 classes: 20 AA + unknown)
        self.type_embedder = nn.Embedding(21, d_type_embed)

        d_coord_out = d_hotspot - d_type_embed

        if use_se3_invariant:
            # Invariant features (7 scalars per anchor):
            #   CA-C distance, CA-N distance, C-CA-N angle,
            #   distance to receptor center,
            #   direction to receptor center in local frame (3 components)
            d_inv_in = 7
            self.coord_encoder = nn.Sequential(
                nn.Linear(d_inv_in, 64),
                nn.ReLU(),
                nn.Linear(64, d_coord_out),
                nn.ReLU(),
            )
        else:
            # Legacy: flatten (3, 3) -> 9
            d_coord_in = 9
            self.coord_encoder = nn.Sequential(
                nn.Linear(d_coord_in, 64),
                nn.ReLU(),
                nn.Linear(64, d_coord_out),
                nn.ReLU(),
            )

        # Final projection after concatenation
        self.output_proj = nn.Sequential(
            nn.Linear(d_hotspot, d_hotspot),
            nn.ReLU(),
            nn.Linear(d_hotspot, d_hotspot),
        )

        # Anchor self-attention layers
        if use_anchor_self_attn:
            self.self_attn_layers = nn.ModuleList()
            for _ in range(num_self_attn_layers):
                self.self_attn_layers.append(
                    nn.ModuleDict(
                        {
                            "attn": nn.MultiheadAttention(
                                embed_dim=d_hotspot,
                                num_heads=4,
                                batch_first=True,
                            ),
                            "ln1": nn.LayerNorm(d_hotspot),
                            "ffn": nn.Sequential(
                                nn.Linear(d_hotspot, d_hotspot * 4),
                                nn.ReLU(),
                                nn.Linear(d_hotspot * 4, d_hotspot),
                            ),
                            "ln2": nn.LayerNorm(d_hotspot),
                        }
                    )
                )

    def _extract_invariant_features(
        self,
        anchor_coords: torch.Tensor,
        receptor_center: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract SE(3)-invariant features from anchor backbone coordinates.

        Args:
            anchor_coords: (B, K, 3, 3) backbone coords in CA/C/N order.
            receptor_center: (B, 3) reference point (receptor center of mass).
                If None, uses centroid of anchor CA positions as fallback.

        Returns:
            (B, K, 7) invariant feature tensor.
        """
        ca = anchor_coords[:, :, 0, :]  # (B, K, 3)
        c = anchor_coords[:, :, 1, :]   # (B, K, 3)
        n = anchor_coords[:, :, 2, :]   # (B, K, 3)

        eps = 1e-8

        # --- Per-anchor internal geometry (invariant) ---
        v_ca_c = c - ca  # (B, K, 3)
        v_ca_n = n - ca  # (B, K, 3)

        ca_c_dist = torch.norm(v_ca_c, dim=-1, keepdim=True)  # (B, K, 1)
        ca_n_dist = torch.norm(v_ca_n, dim=-1, keepdim=True)  # (B, K, 1)

        # C-CA-N bond angle
        cos_angle = (v_ca_c * v_ca_n).sum(-1, keepdim=True) / (
            ca_c_dist * ca_n_dist + eps
        )
        angle = torch.acos(cos_angle.clamp(-1 + 1e-6, 1 - 1e-6))  # (B, K, 1)

        # --- Receptor-relative features (invariant) ---
        if receptor_center is None:
            receptor_center = ca.mean(dim=1)  # (B, 3)

        ca_to_center = receptor_center.unsqueeze(1) - ca  # (B, K, 3)
        dist_to_center = torch.norm(ca_to_center, dim=-1, keepdim=True)  # (B, K, 1)

        # Construct local orthonormal frame from CA/C/N
        # e1: CA->C direction, e2: orthogonal in CA-C-N plane, e3: cross product
        e1 = v_ca_c / (ca_c_dist + eps)
        u2 = v_ca_n - (v_ca_n * e1).sum(-1, keepdim=True) * e1
        e2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + eps)
        e3 = torch.cross(e1, e2, dim=-1)  # (B, K, 3)

        # Direction to receptor center expressed in local frame: R^T @ dir
        dir_unit = ca_to_center / (dist_to_center + eps)  # (B, K, 3)

        # R = [e1 | e2 | e3] as column vectors → R^T @ v = [e1·v, e2·v, e3·v]
        dir_local = torch.stack(
            [
                (e1 * dir_unit).sum(-1),
                (e2 * dir_unit).sum(-1),
                (e3 * dir_unit).sum(-1),
            ],
            dim=-1,
        )  # (B, K, 3)

        return torch.cat(
            [ca_c_dist, ca_n_dist, angle, dist_to_center, dir_local], dim=-1
        )  # (B, K, 7)

    def forward(
        self,
        anchor_coords: torch.Tensor,
        anchor_types: torch.Tensor,
        anchor_mask: torch.Tensor,
        receptor_center: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode hotspot anchors.

        Args:
            anchor_coords: (B, K, 3, 3) backbone coordinates in CA/C/N order.
            anchor_types: (B, K) residue type indices in [0, 20].
            anchor_mask: (B, K) boolean mask for valid anchors.
            receptor_center: (B, 3) receptor center of mass. Only used when
                ``use_se3_invariant=True``. If None, falls back to anchor
                CA centroid.

        Returns:
            Hotspot features (B, K, d_hotspot). Masked positions are zeroed.
        """
        B, K = anchor_types.shape

        # Coordinate features
        if self.use_se3_invariant:
            inv_feat = self._extract_invariant_features(anchor_coords, receptor_center)
            coord_feat = self.coord_encoder(inv_feat)
        else:
            coords_flat = anchor_coords.reshape(B, K, -1)  # (B, K, 9)
            coord_feat = self.coord_encoder(coords_flat)

        # Type features
        type_feat = self.type_embedder(anchor_types.clamp(0, 20))  # (B, K, d_type_embed)

        # Combine and project
        combined = torch.cat([coord_feat, type_feat], dim=-1)  # (B, K, d_hotspot)
        out = self.output_proj(combined)  # (B, K, d_hotspot)

        # Anchor self-attention
        if self.use_anchor_self_attn:
            key_padding_mask = ~anchor_mask.bool()
            for layer in self.self_attn_layers:
                attn_out, _ = layer["attn"](
                    out, out, out, key_padding_mask=key_padding_mask,
                )
                out = layer["ln1"](out + attn_out)
                ffn_out = layer["ffn"](out)
                out = layer["ln2"](out + ffn_out)

        # Zero out invalid anchors
        out = out * anchor_mask.unsqueeze(-1).float()
        return out
