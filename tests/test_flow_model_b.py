"""Smoke test for FlowModelB: instantiation, forward, sample, gradient flow."""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PEPFLOW_ROOT = REPO_ROOT / "PepFlowww"
sys.path.insert(0, str(PEPFLOW_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from easydict import EasyDict
from hotflow.models.flow_model_b import FlowModelB


def make_cfg():
    return EasyDict(
        encoder=EasyDict(
            node_embed_size=128,
            edge_embed_size=64,
            ipa=EasyDict(
                c_s=128, c_z=64, c_hidden=128,
                no_heads=8, no_qk_points=8, no_v_points=12,
                seq_tfmr_num_heads=4, seq_tfmr_num_layers=2,
                num_blocks=6, stop_grad=False,
            ),
        ),
        interpolant=EasyDict(
            min_t=0.01,
            t_normalization_clip=0.9,
            sample_structure=True,
            sample_sequence=True,
            rots=EasyDict(train_schedule='linear', sample_schedule='exp', exp_rate=10),
            trans=EasyDict(train_schedule='linear', sample_schedule='linear', sigma=1.0),
            seqs=EasyDict(num_classes=20, simplex_value=5.0),
            sampling=EasyDict(num_timesteps=100),
            self_condition=False,
        ),
    )


def make_batch(B=2, L_rec=10, L_pep=6, A=15, device='cpu'):
    L = L_rec + L_pep
    gen_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    gen_mask[:, L_rec:] = True
    res_mask = torch.ones(B, L, dtype=torch.bool, device=device)

    pos = torch.randn(B, L, A, 3, device=device)
    # Place peptide residues close to receptor for some hotspot contacts
    pos[:, L_rec:L_rec+3, :, :] = pos[:, :3, :, :] + 0.1  # within 4A

    mask_ha = torch.ones(B, L, A, dtype=torch.bool, device=device)
    mask_ha[:, :, 5:] = False  # only first 5 atoms valid

    aa = torch.randint(0, 20, (B, L), device=device)
    res_nb = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
    chain_nb = torch.zeros(B, L, dtype=torch.long, device=device)
    chain_nb[:, L_rec:] = 1

    torsion = torch.rand(B, L, 5, device=device) * 2 * 3.14159
    torsion_mask = torch.ones(B, L, 5, dtype=torch.bool, device=device)

    return {
        'aa': aa,
        'res_nb': res_nb,
        'chain_nb': chain_nb,
        'pos_heavyatom': pos,
        'mask_heavyatom': mask_ha,
        'generate_mask': gen_mask,
        'res_mask': res_mask,
        'torsion_angle': torsion,
        'torsion_angle_mask': torsion_mask,
    }


def test_instantiation():
    cfg = make_cfg()
    model = FlowModelB(cfg, num_hotspot_anchors=3)
    print(f"[OK] Instantiation: {sum(p.numel() for p in model.parameters()):,} params")

    # Check that ga_encoder is GAEncoderCrossAttn
    from hotflow.models.ga_encoder_crossattn import GAEncoderCrossAttn
    assert isinstance(model.ga_encoder, GAEncoderCrossAttn), "ga_encoder should be GAEncoderCrossAttn"
    print("[OK] ga_encoder is GAEncoderCrossAttn")

    # Check hotspot_encoder exists
    from hotflow.models.hotspot_encoder import HotspotEncoder
    assert isinstance(model.hotspot_encoder, HotspotEncoder)
    print("[OK] hotspot_encoder is HotspotEncoder")


def test_forward():
    cfg = make_cfg()
    model = FlowModelB(cfg, num_hotspot_anchors=3)
    model.train()

    batch = make_batch()
    loss_dict = model(batch)

    print(f"\n[Forward] Loss keys: {list(loss_dict.keys())}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.6f}")

    # Must have contact_loss
    assert 'contact_loss' in loss_dict, "Missing contact_loss in output"
    print("[OK] contact_loss present")

    # All losses should be finite
    for k, v in loss_dict.items():
        assert torch.isfinite(v), f"{k} is not finite: {v}"
    print("[OK] All losses finite")


def test_backward():
    cfg = make_cfg()
    model = FlowModelB(cfg, num_hotspot_anchors=3)
    model.train()

    batch = make_batch()
    loss_dict = model(batch)

    total_loss = sum(loss_dict.values())
    total_loss.backward()

    # Check gradients flow to cross-attention and hotspot encoder
    has_crossattn_grad = False
    has_hotspot_grad = False
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            if 'hotspot_cross_attn' in name:
                has_crossattn_grad = True
            if 'hotspot_encoder' in name:
                has_hotspot_grad = True

    print(f"\n[Backward] cross_attn grads: {has_crossattn_grad}, hotspot_encoder grads: {has_hotspot_grad}")
    assert has_crossattn_grad, "No gradients in cross-attention layers"
    assert has_hotspot_grad, "No gradients in hotspot encoder"
    print("[OK] Gradients flow to all new modules")


def test_sample():
    cfg = make_cfg()
    model = FlowModelB(cfg, num_hotspot_anchors=3)
    model.eval()

    batch = make_batch()

    # Sample with GT-based hotspot extraction (validation mode)
    traj = model.sample(batch, num_steps=3)
    print(f"\n[Sample] Trajectory length: {len(traj)}")
    print(f"  Keys per step: {list(traj[-1].keys())}")
    assert len(traj) == 3, f"Expected 3 steps, got {len(traj)}"

    # Sample with explicit anchors (inference mode)
    B = 2
    K = 3
    anchor_coords = torch.randn(B, K, 3, 3)
    anchor_types = torch.randint(0, 20, (B, K))
    anchor_mask = torch.ones(B, K, dtype=torch.bool)
    anchor_mask[:, -1] = False  # last anchor invalid

    traj2 = model.sample(
        batch, num_steps=3,
        anchor_coords=anchor_coords,
        anchor_types=anchor_types,
        anchor_mask=anchor_mask,
    )
    assert len(traj2) == 3
    print("[OK] Sample with explicit anchors works")


def test_cfg_sample():
    """Test classifier-free guidance (scale > 1)."""
    cfg = make_cfg()
    model = FlowModelB(cfg, num_hotspot_anchors=3)
    model.eval()

    batch = make_batch()
    traj = model.sample(batch, num_steps=3, guidance_scale=2.0)
    assert len(traj) == 3
    print("\n[OK] Classifier-free guidance sampling works (scale=2.0)")


if __name__ == '__main__':
    test_instantiation()
    test_forward()
    test_backward()
    test_sample()
    test_cfg_sample()
    print("\n=== All FlowModelB tests passed ===")
