"""SE(3)-equivariant HotspotPredictor.

Key design (equivariance by construction):
    1. Inputs to encoder are SE(3)-invariant features (AA embedding, pairwise
       CA distance attention bias).
    2. Each receptor residue carries a local frame (R_i, t_i) built from its
       backbone (N, CA, C) via Gram-Schmidt.
    3. Per-anchor outputs (CA, C, N positions) are produced as a
       weighted combination of receptor frames + learned local-frame offsets:
           anchor_pos[k, a] = Σ_i attn[k, i] · (t_i + R_i @ offset[k, a])
       Since Σ attn = 1 and offset is invariant, this is SE(3) equivariant.
    4. AA logits come from query scalars (invariant by construction).

Coord order (both input and output): [N, CA, C] (matching train script).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_frames(rec_coords: torch.Tensor):
    """Build per-residue rotation R (B, R, 3, 3) and translation t (B, R, 3)
    from backbone atoms in order [N, CA, C]."""
    N = rec_coords[..., 0, :]
    CA = rec_coords[..., 1, :]
    C = rec_coords[..., 2, :]
    x = F.normalize(N - CA, dim=-1)
    c = C - CA
    c = c - (c * x).sum(-1, keepdim=True) * x
    y = F.normalize(c, dim=-1)
    z = torch.cross(x, y, dim=-1)
    Rmat = torch.stack([x, y, z], dim=-1)  # columns = [x, y, z]
    return Rmat, CA


class SpatialEncoderBlock(nn.Module):
    """Self-attention block with pairwise CA-distance bias (invariant)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )
        self.dist_bias = nn.Sequential(
            nn.Linear(1, n_heads), nn.SiLU(),
            nn.Linear(n_heads, n_heads),
        )

    def forward(self, x, ca_coords, rec_mask):
        B, R = rec_mask.shape
        H = self.n_heads

        dist = torch.cdist(ca_coords, ca_coords)         # (B, R, R)
        bias = self.dist_bias(dist.unsqueeze(-1))        # (B, R, R, H)
        bias = bias.permute(0, 3, 1, 2)                  # (B, H, R, R)
        pair_mask = rec_mask.unsqueeze(1) & rec_mask.unsqueeze(2)
        bias = bias.masked_fill(~pair_mask.unsqueeze(1), -1e4)
        bias = bias.reshape(B * H, R, R)

        h = self.norm1(x)
        out, _ = self.attn(h, h, h, attn_mask=bias, key_padding_mask=~rec_mask)
        x = x + out
        x = x + self.ff(self.norm2(x))
        return x


class EquivariantOutputHead(nn.Module):
    """Produces (anchor_coords (B,K,3,3), aa_logits (B,K,20))
    equivariantly from query features and receptor frames."""

    def __init__(self, d_model: int, K: int):
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.attn_q = nn.Linear(d_model, d_model)
        self.attn_k = nn.Linear(d_model, d_model)
        self.offset_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 9),  # 3 atoms × 3 coords
        )
        self.aa_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 20),
        )

    def forward(self, h, memory, R_rec, t_rec, rec_mask):
        """
        h:       (B, K, d) — query features after cross-attention
        memory:  (B, R, d) — receptor features
        R_rec:   (B, R, 3, 3)
        t_rec:   (B, R, 3)
        rec_mask:(B, R) bool
        Returns:
            anchor_coords: (B, K, 3, 3) — [N, CA, C] in global frame
            aa_logits:     (B, K, 20)
        """
        B, K, _ = h.shape
        R = memory.shape[1]

        q = self.attn_q(h)            # (B, K, d)
        k = self.attn_k(memory)       # (B, R, d)
        scores = torch.einsum("bkd,brd->bkr", q, k) / (self.d_model ** 0.5)
        scores = scores.masked_fill(~rec_mask.unsqueeze(1), -1e4)
        attn = F.softmax(scores, dim=-1)  # (B, K, R)

        offsets = self.offset_proj(h).reshape(B, K, 3, 3)  # (B, K, 3 atoms, 3 coords)
        # rotated[b, k, r, atom, c] = Σ_j R_rec[b, r, c, j] · offsets[b, k, atom, j]
        rotated = torch.einsum("brij,bkaj->bkrai", R_rec, offsets)
        positions = t_rec.unsqueeze(1).unsqueeze(3) + rotated  # (B, K, R, 3, 3)

        anchor_coords = (attn.unsqueeze(-1).unsqueeze(-1) * positions).sum(dim=2)
        aa_logits = self.aa_head(h)
        return anchor_coords, aa_logits


class HotspotPredictorSE3(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 2,
        n_heads: int = 4,
        K: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.K = K

        # Invariant per-residue input features
        self.aa_emb = nn.Embedding(21, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Spatial encoder (invariant)
        self.encoder_blocks = nn.ModuleList([
            SpatialEncoderBlock(d_model, n_heads, dropout)
            for _ in range(n_encoder_layers)
        ])

        # K learnable query tokens + cross-attention decoder
        self.queries = nn.Parameter(torch.randn(K, d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, n_decoder_layers)

        # Equivariant output head
        self.output_head = EquivariantOutputHead(d_model, K)

    def forward(self, rec_coords, rec_aa, rec_mask):
        """
        Args:
            rec_coords: (B, R, 3, 3) — atoms in order [N, CA, C]
            rec_aa:     (B, R) long, 0–20
            rec_mask:   (B, R) bool, True = valid
        Returns:
            anchor_coords: (B, K, 3, 3) — global coords, atoms [N, CA, C]
            aa_logits:     (B, K, 20)
        """
        # Build receptor frames
        R_rec, t_rec = _make_frames(rec_coords)
        ca = rec_coords[..., 1, :]  # for distance bias

        # Invariant features
        f = self.input_norm(self.aa_emb(rec_aa.clamp(0, 20)))

        # Spatial encoder
        for block in self.encoder_blocks:
            f = block(f, ca, rec_mask)

        # Cross-attention decoder
        q = self.queries.unsqueeze(0).expand(rec_aa.shape[0], -1, -1).contiguous()
        h = self.decoder(q, f, memory_key_padding_mask=~rec_mask)

        # Equivariant output
        anchor_coords, aa_logits = self.output_head(h, f, R_rec, t_rec, rec_mask)
        return anchor_coords, aa_logits
