"""Unit tests for HotspotEncoder: SE(3) invariance and permutation equivariance."""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hotflow.models.hotspot_encoder import HotspotEncoder


def _random_rotation(device="cpu"):
    """Sample a random rotation matrix via QR decomposition."""
    m = torch.randn(3, 3, device=device)
    q, r = torch.linalg.qr(m)
    # Ensure proper rotation (det=+1)
    q = q * torch.sign(torch.diag(r)).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] *= -1
    return q  # (3, 3)


def _apply_se3(coords, R, t):
    """Apply SE(3) transform to anchor coords (B, K, 3, 3)."""
    # coords[..., atom, xyz] -> R @ xyz + t
    return torch.einsum("ij,...j->...i", R, coords) + t


def test_se3_invariance():
    """Output should be identical under random rotation + translation of all coords."""
    torch.manual_seed(42)
    B, K = 2, 5
    device = "cpu"

    encoder = HotspotEncoder(
        d_hotspot=64, use_se3_invariant=True, use_anchor_self_attn=False,
    )
    encoder.eval()

    anchor_coords = torch.randn(B, K, 3, 3, device=device)
    anchor_types = torch.randint(0, 20, (B, K), device=device)
    anchor_mask = torch.ones(B, K, dtype=torch.bool, device=device)
    receptor_center = torch.randn(B, 3, device=device)

    # Original output
    out_orig = encoder(anchor_coords, anchor_types, anchor_mask, receptor_center)

    # Apply random SE(3) transform to both anchors and receptor center
    R = _random_rotation(device)
    t = torch.randn(3, device=device)

    coords_transformed = _apply_se3(anchor_coords, R, t)
    center_transformed = (R @ receptor_center.unsqueeze(-1)).squeeze(-1) + t

    out_transformed = encoder(
        coords_transformed, anchor_types, anchor_mask, center_transformed,
    )

    diff = (out_orig - out_transformed).abs().max().item()
    assert diff < 1e-4, (
        f"SE(3) invariance violated: max diff = {diff:.6f}"
    )
    print(f"[PASS] SE(3) invariance: max diff = {diff:.2e}")


def test_se3_invariance_translation_only():
    """Output should be identical under pure translation."""
    torch.manual_seed(123)
    B, K = 3, 5

    encoder = HotspotEncoder(
        d_hotspot=64, use_se3_invariant=True, use_anchor_self_attn=False,
    )
    encoder.eval()

    anchor_coords = torch.randn(B, K, 3, 3)
    anchor_types = torch.randint(0, 20, (B, K))
    anchor_mask = torch.ones(B, K, dtype=torch.bool)
    receptor_center = torch.randn(B, 3)

    out_orig = encoder(anchor_coords, anchor_types, anchor_mask, receptor_center)

    # Translate everything by a large random offset
    t = torch.randn(3) * 100
    coords_shifted = anchor_coords + t
    center_shifted = receptor_center + t

    out_shifted = encoder(coords_shifted, anchor_types, anchor_mask, center_shifted)

    diff = (out_orig - out_shifted).abs().max().item()
    assert diff < 1e-4, f"Translation invariance violated: max diff = {diff:.6f}"
    print(f"[PASS] Translation invariance: max diff = {diff:.2e}")


def test_se3_invariance_no_receptor_center():
    """SE(3) invariance should hold even without explicit receptor_center
    (falls back to anchor CA centroid, which also transforms covariantly)."""
    torch.manual_seed(7)
    B, K = 2, 5

    encoder = HotspotEncoder(
        d_hotspot=64, use_se3_invariant=True, use_anchor_self_attn=False,
    )
    encoder.eval()

    anchor_coords = torch.randn(B, K, 3, 3)
    anchor_types = torch.randint(0, 20, (B, K))
    anchor_mask = torch.ones(B, K, dtype=torch.bool)

    out_orig = encoder(anchor_coords, anchor_types, anchor_mask)

    R = _random_rotation()
    t = torch.randn(3) * 50
    coords_transformed = _apply_se3(anchor_coords, R, t)

    out_transformed = encoder(coords_transformed, anchor_types, anchor_mask)

    diff = (out_orig - out_transformed).abs().max().item()
    assert diff < 1e-4, f"SE(3) invariance (no center) violated: max diff = {diff:.6f}"
    print(f"[PASS] SE(3) invariance (no receptor_center): max diff = {diff:.2e}")


def test_legacy_mode_not_invariant():
    """Legacy mode (use_se3_invariant=False) should NOT be SE(3)-invariant."""
    torch.manual_seed(99)
    B, K = 2, 5

    encoder = HotspotEncoder(
        d_hotspot=64, use_se3_invariant=False, use_anchor_self_attn=False,
    )
    encoder.eval()

    anchor_coords = torch.randn(B, K, 3, 3)
    anchor_types = torch.randint(0, 20, (B, K))
    anchor_mask = torch.ones(B, K, dtype=torch.bool)

    out_orig = encoder(anchor_coords, anchor_types, anchor_mask)

    R = _random_rotation()
    t = torch.randn(3) * 10
    coords_transformed = _apply_se3(anchor_coords, R, t)

    out_transformed = encoder(coords_transformed, anchor_types, anchor_mask)

    diff = (out_orig - out_transformed).abs().max().item()
    assert diff > 0.01, (
        f"Legacy mode should NOT be SE(3)-invariant, but diff = {diff:.6f}"
    )
    print(f"[PASS] Legacy mode is NOT SE(3)-invariant (diff = {diff:.4f})")


def test_anchor_self_attn_permutation_equivariance():
    """Permuting anchor order should permute output in the same way."""
    torch.manual_seed(77)
    B, K = 2, 5

    encoder = HotspotEncoder(
        d_hotspot=64, use_se3_invariant=True, use_anchor_self_attn=True,
    )
    encoder.eval()

    anchor_coords = torch.randn(B, K, 3, 3)
    anchor_types = torch.randint(0, 20, (B, K))
    anchor_mask = torch.ones(B, K, dtype=torch.bool)
    receptor_center = torch.randn(B, 3)

    out_orig = encoder(anchor_coords, anchor_types, anchor_mask, receptor_center)

    # Apply a random permutation to anchor dimension
    perm = torch.randperm(K)
    coords_perm = anchor_coords[:, perm]
    types_perm = anchor_types[:, perm]
    mask_perm = anchor_mask[:, perm]

    out_perm = encoder(coords_perm, types_perm, mask_perm, receptor_center)

    # The permuted output should match the original output reordered
    out_orig_reordered = out_orig[:, perm]
    diff = (out_orig_reordered - out_perm).abs().max().item()
    assert diff < 1e-4, (
        f"Permutation equivariance violated: max diff = {diff:.6f}"
    )
    print(f"[PASS] Anchor self-attn permutation equivariance: max diff = {diff:.2e}")


def test_self_attn_with_mask():
    """Self-attention should still work correctly with partial anchor masks."""
    torch.manual_seed(55)
    B, K = 2, 5

    encoder = HotspotEncoder(
        d_hotspot=64, use_se3_invariant=True, use_anchor_self_attn=True,
    )
    encoder.eval()

    anchor_coords = torch.randn(B, K, 3, 3)
    anchor_types = torch.randint(0, 20, (B, K))
    anchor_mask = torch.ones(B, K, dtype=torch.bool)
    anchor_mask[0, 3:] = False  # first batch has only 3 valid anchors
    anchor_mask[1, 4:] = False  # second has 4
    receptor_center = torch.randn(B, 3)

    out = encoder(anchor_coords, anchor_types, anchor_mask, receptor_center)

    # Masked positions should be zero
    assert (out[0, 3:] == 0).all(), "Masked positions should be zeroed (batch 0)"
    assert (out[1, 4:] == 0).all(), "Masked positions should be zeroed (batch 1)"
    # Valid positions should be non-zero
    assert out[0, :3].abs().sum() > 0, "Valid positions should be non-zero"
    print("[PASS] Self-attention with partial masks")


def test_output_shape_all_configs():
    """All 4 ablation configurations should produce (B, K, 64) output."""
    torch.manual_seed(0)
    B, K = 2, 5

    configs = [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]

    anchor_coords = torch.randn(B, K, 3, 3)
    anchor_types = torch.randint(0, 20, (B, K))
    anchor_mask = torch.ones(B, K, dtype=torch.bool)

    for use_se3, use_sa in configs:
        enc = HotspotEncoder(
            d_hotspot=64,
            use_se3_invariant=use_se3,
            use_anchor_self_attn=use_sa,
        )
        enc.eval()
        out = enc(anchor_coords, anchor_types, anchor_mask)
        assert out.shape == (B, K, 64), (
            f"Shape mismatch for se3={use_se3}, sa={use_sa}: {out.shape}"
        )
    print("[PASS] All 4 configs produce correct output shape (B, K, 64)")


def test_gradient_flow():
    """Gradients should flow through all new components."""
    torch.manual_seed(11)
    B, K = 2, 5

    encoder = HotspotEncoder(
        d_hotspot=64, use_se3_invariant=True, use_anchor_self_attn=True,
    )
    encoder.train()

    anchor_coords = torch.randn(B, K, 3, 3, requires_grad=True)
    anchor_types = torch.randint(0, 20, (B, K))
    anchor_mask = torch.ones(B, K, dtype=torch.bool)
    receptor_center = torch.randn(B, 3, requires_grad=True)

    out = encoder(anchor_coords, anchor_types, anchor_mask, receptor_center)
    loss = out.sum()
    loss.backward()

    assert anchor_coords.grad is not None, "No gradient for anchor_coords"
    assert receptor_center.grad is not None, "No gradient for receptor_center"

    # Check all encoder parameters have gradients
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"

    print("[PASS] Gradient flow through all components")


if __name__ == "__main__":
    test_se3_invariance()
    test_se3_invariance_translation_only()
    test_se3_invariance_no_receptor_center()
    test_legacy_mode_not_invariant()
    test_anchor_self_attn_permutation_equivariance()
    test_self_attn_with_mask()
    test_output_shape_all_configs()
    test_gradient_flow()
    print("\n=== All HotspotEncoder tests passed ===")
