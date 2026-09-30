"""HotspotPredictor V2 — plain Transformer + invariant spatial features.

Improvements vs V1 (`hotspot_predictor.py`):
    1. **CA distance bias** added to encoder self-attention (per head).
       Injects geometry signal into otherwise pure-AA Transformer.
    2. **Backbone dihedral (phi, psi)** features per residue.
       SE(3)-invariant geometric features.
    3. **Sequence position encoding** (learnable embedding by residue index).

Not architecturally SE(3)-equivariant — relies on random rotation
augmentation in the training loop to achieve near-equivariance.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _dihedral(a, b, c, d, eps: float = 1e-7):
    """Compute dihedral a-b-c-d. Inputs (..., 3). Output (...,) radians."""
    b1 = b - a
    b2 = c - b
    b3 = d - c
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    b2n = F.normalize(b2, dim=-1, eps=eps)
    m1 = torch.cross(n1, b2n, dim=-1)
    x = (n1 * n2).sum(-1)
    y = (m1 * n2).sum(-1)
    return torch.atan2(y, x)


def compute_backbone_dihedrals(rec_coords: torch.Tensor) -> torch.Tensor:
    """Compute phi, psi per residue.

    Args:
        rec_coords: (B, R, 3, 3), atoms order [N, CA, C].
    Returns:
        (B, R, 4) — [sin(phi), cos(phi), sin(psi), cos(psi)]
        Boundary residues (first/last) have phi/psi set to 0 (sin=0, cos=1).
    """
    N = rec_coords[..., 0, :]
    CA = rec_coords[..., 1, :]
    C = rec_coords[..., 2, :]

    # phi[i] needs C[i-1], N[i], CA[i], C[i]
    Cm1 = F.pad(C[:, :-1, :], (0, 0, 1, 0))   # zero at index 0
    phi = _dihedral(Cm1, N, CA, C)

    # psi[i] needs N[i], CA[i], C[i], N[i+1]
    Np1 = F.pad(N[:, 1:, :], (0, 0, 0, 1))    # zero at last index
    psi = _dihedral(N, CA, C, Np1)

    # Boundary: set first phi and last psi to 0 (sin=0, cos=1)
    B, R = N.shape[:2]
    phi[:, 0] = 0.0
    psi[:, -1] = 0.0

    feats = torch.stack([phi.sin(), phi.cos(), psi.sin(), psi.cos()], dim=-1)
    # Replace any NaN (e.g., from degenerate triplets in padded regions) with 0/1
    feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


# ---------------------------------------------------------------------------
# Encoder block with CA distance bias
# ---------------------------------------------------------------------------

class SpatialEncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block with per-head CA-distance bias."""

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

        dist = torch.cdist(ca_coords, ca_coords)              # (B, R, R)
        bias = self.dist_bias(dist.unsqueeze(-1))             # (B, R, R, H)
        bias = bias.permute(0, 3, 1, 2)                       # (B, H, R, R)
        pair_mask = rec_mask.unsqueeze(1) & rec_mask.unsqueeze(2)
        bias = bias.masked_fill(~pair_mask.unsqueeze(1), -1e4)
        bias = bias.reshape(B * H, R, R)

        h = self.norm1(x)
        out, _ = self.attn(h, h, h, attn_mask=bias, key_padding_mask=~rec_mask)
        x = x + out
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# HotspotPredictor V2
# ---------------------------------------------------------------------------

class HotspotPredictorV2(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 2,
        n_heads: int = 4,
        K: int = 5,
        dropout: float = 0.1,
        max_pos: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.K = K

        # Inputs (per-residue invariant features)
        self.aa_emb = nn.Embedding(21, d_model)
        self.dih_proj = nn.Linear(4, d_model)
        self.pos_emb = nn.Embedding(max_pos, d_model)
        # Optional non-equivariant coord injection (centered)
        self.coord_proj = nn.Linear(9, d_model)
        self.input_proj = nn.Linear(d_model, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Spatial encoder
        self.encoder_blocks = nn.ModuleList([
            SpatialEncoderBlock(d_model, n_heads, dropout)
            for _ in range(n_encoder_layers)
        ])

        # K learnable queries + cross-attention decoder
        self.queries = nn.Parameter(torch.randn(K, d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, n_decoder_layers)

        # Output heads
        self.coord_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 9),  # N, CA, C
        )
        self.aa_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 20),
        )

    def forward(self, rec_coords, rec_aa, rec_mask):
        """
        Args:
            rec_coords: (B, R, 3, 3) — [N, CA, C]
            rec_aa:     (B, R) long, 0–20
            rec_mask:   (B, R) bool
        Returns:
            anchor_coords: (B, K, 3, 3) absolute
            aa_logits:     (B, K, 20)
        """
        B, R = rec_aa.shape

        # Centroid centering
        ca = rec_coords[..., 1, :]                        # (B, R, 3)
        mask_f = rec_mask.float().unsqueeze(-1)
        centroid = (ca * mask_f).sum(1) / mask_f.sum(1).clamp_min(1.0)
        centered = rec_coords - centroid[:, None, None, :]
        ca_centered = centered[..., 1, :]

        # Features
        aa_e = self.aa_emb(rec_aa.clamp(0, 20))            # (B, R, d)
        coord_e = self.coord_proj(centered.reshape(B, R, 9))
        dih = compute_backbone_dihedrals(rec_coords)       # (B, R, 4)
        dih_e = self.dih_proj(dih)
        pos_idx = torch.arange(R, device=rec_aa.device).clamp(max=self.pos_emb.num_embeddings - 1)
        pos_e = self.pos_emb(pos_idx).unsqueeze(0).expand(B, -1, -1)

        f = self.input_proj(aa_e + coord_e + dih_e + pos_e)
        f = self.input_norm(f)

        # Spatial encoder
        for block in self.encoder_blocks:
            f = block(f, ca_centered, rec_mask)

        # Decoder
        q = self.queries.unsqueeze(0).expand(B, -1, -1).contiguous()
        h = self.decoder(q, f, memory_key_padding_mask=~rec_mask)

        # Output (in centered frame → add back centroid)
        coord_centered = self.coord_head(h).reshape(B, self.K, 3, 3)
        anchor_coords = coord_centered + centroid[:, None, None, :]
        aa_logits = self.aa_head(h)
        return anchor_coords, aa_logits
