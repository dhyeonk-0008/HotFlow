"""HotspotPredictor — predicts K peptide-side hotspot anchors from receptor pocket.

Inputs:
    rec_coords: (B, R, 3, 3) backbone (N, CA, C) coords
    rec_aa:     (B, R) amino acid types (0-19, 20=unknown/pad)
    rec_mask:   (B, R) bool — True for valid receptor residue

Outputs:
    coord:     (B, K, 3, 3) predicted (N, CA, C) anchor backbone
    aa_logits: (B, K, 20) AA classification logits

Architecture: receptor centred on its own centroid → AA embedding + position
projection → Transformer encoder → K learnable query tokens → cross-attention
→ MLP heads (coord regression + AA classification). Predictions are returned
in absolute coords (centroid added back).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HotspotPredictor(nn.Module):
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

        # Inputs: 21 (20 AA + unknown)
        self.aa_emb = nn.Embedding(21, d_model)
        # Backbone coord projection (N,CA,C flatten = 9)
        self.pos_proj = nn.Linear(9, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Receptor encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_encoder_layers)

        # K learnable query tokens
        self.queries = nn.Parameter(torch.randn(K, d_model) * 0.02)

        # Decoder = cross-attn + self-attn stack
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, n_decoder_layers)

        # Output heads — predict centered coords + AA
        self.coord_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 9),  # N(3) + CA(3) + C(3)
        )
        self.aa_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 20),
        )

    def forward(
        self,
        rec_coords: torch.Tensor,
        rec_aa: torch.Tensor,
        rec_mask: torch.Tensor,
    ):
        """
        Args:
            rec_coords: (B, R, 3, 3) — (N, CA, C) order
            rec_aa:     (B, R) long, 0–20
            rec_mask:   (B, R) bool, True = valid
        Returns:
            anchor_coords: (B, K, 3, 3) — (N, CA, C) absolute coords
            anchor_aa_logits: (B, K, 20)
        """
        B, R = rec_aa.shape

        # Compute CA centroid for centering (mask-aware)
        ca = rec_coords[:, :, 1, :]  # (B, R, 3) — CA = index 1
        mask_f = rec_mask.float().unsqueeze(-1)
        centroid = (ca * mask_f).sum(1) / mask_f.sum(1).clamp_min(1.0)  # (B, 3)

        centered = rec_coords - centroid[:, None, None, :]  # (B, R, 3, 3)

        # Build per-residue features
        aa_e = self.aa_emb(rec_aa.clamp(0, 20))             # (B, R, d)
        pos_e = self.pos_proj(centered.reshape(B, R, 9))    # (B, R, d)
        x = self.input_norm(aa_e + pos_e)

        # Encoder
        src_key_pad = ~rec_mask   # True = ignore
        memory = self.encoder(x, src_key_padding_mask=src_key_pad)  # (B, R, d)

        # Decoder with K query tokens
        q = self.queries.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, K, d)
        h = self.decoder(q, memory, memory_key_padding_mask=src_key_pad)  # (B, K, d)

        # Heads
        coord_centered = self.coord_head(h).reshape(B, self.K, 3, 3)
        anchor_coords = coord_centered + centroid[:, None, None, :]
        aa_logits = self.aa_head(h)

        return anchor_coords, aa_logits


# ---------------------------------------------------------------------------
# Hungarian matching + loss
# ---------------------------------------------------------------------------

def hungarian_match_and_loss(
    pred_coords: torch.Tensor,
    pred_aa_logits: torch.Tensor,
    gt_coords: torch.Tensor,
    gt_aa: torch.Tensor,
    gt_mask: torch.Tensor,
    coord_weight: float = 1.0,
    aa_weight: float = 1.0,
    match_metric: str = "ca_l1",
):
    """Method C loss: Hungarian match preds to valid GT, loss only on matched pairs.

    Args:
        pred_coords:    (B, K, 3, 3)
        pred_aa_logits: (B, K, 20)
        gt_coords:      (B, K, 3, 3)
        gt_aa:          (B, K) long, 0–20
        gt_mask:        (B, K) bool — True for valid GT anchor
        match_metric: 'ca_l1' (CA position only) or 'bb_l1' (all 9 coords)
    Returns:
        dict(total, coord, aa, n_matched, n_valid)
    """
    from scipy.optimize import linear_sum_assignment

    B, K = gt_aa.shape
    device = pred_coords.device

    coord_loss = torch.zeros((), device=device)
    aa_loss = torch.zeros((), device=device)
    n_matched_total = 0
    n_valid_total = 0

    for b in range(B):
        valid_idx = gt_mask[b].nonzero(as_tuple=True)[0]
        n_valid = valid_idx.numel()
        n_valid_total += int(n_valid)
        if n_valid == 0:
            continue

        pred_b = pred_coords[b]                # (K, 3, 3)
        gt_b = gt_coords[b, valid_idx]         # (n_valid, 3, 3)

        # Cost matrix (K, n_valid)
        if match_metric == "ca_l1":
            cost = (pred_b[:, 1, :].unsqueeze(1) - gt_b[:, 1, :].unsqueeze(0)).abs().sum(-1)
        else:
            cost = (pred_b.unsqueeze(1) - gt_b.unsqueeze(0)).abs().sum(dim=(-1, -2))

        cost_np = cost.detach().cpu().numpy()
        row_idx, col_idx = linear_sum_assignment(cost_np)
        # row_idx: indices into pred (K), col_idx: indices into gt_b (n_valid)

        row_t = torch.as_tensor(row_idx, device=device, dtype=torch.long)
        col_t = torch.as_tensor(col_idx, device=device, dtype=torch.long)

        matched_pred = pred_b[row_t]                      # (n_valid, 3, 3)
        matched_gt = gt_b[col_t]                          # (n_valid, 3, 3)
        coord_loss = coord_loss + F.smooth_l1_loss(
            matched_pred, matched_gt, reduction="sum"
        )

        matched_aa_logits = pred_aa_logits[b][row_t]       # (n_valid, 20)
        matched_aa_target = gt_aa[b, valid_idx][col_t].clamp(0, 19)
        aa_loss = aa_loss + F.cross_entropy(
            matched_aa_logits, matched_aa_target, reduction="sum"
        )

        n_matched_total += int(n_valid)

    denom = max(n_matched_total, 1)
    coord_loss = coord_loss / denom
    aa_loss = aa_loss / denom
    total = coord_weight * coord_loss + aa_weight * aa_loss

    return {
        "total": total,
        "coord": coord_loss.detach(),
        "aa": aa_loss.detach(),
        "n_matched": n_matched_total,
        "n_valid": n_valid_total,
    }
