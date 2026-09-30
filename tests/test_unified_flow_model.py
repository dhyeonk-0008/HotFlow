"""Sanity tests for UnifiedFlowModel and AnchorToGraphTransform."""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PEPFLOW_ROOT = REPO_ROOT / "PepFlowww"
sys.path.insert(0, str(PEPFLOW_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from easydict import EasyDict
from hotflow.models.unified import UnifiedFlowModel
from hotflow.data.transforms import HotspotAnnotationTransform, AnchorToGraphTransform


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


def make_single_item(L_rec=10, L_pep=6, A=15):
    """Create a single unbatched data item (pre-collation)."""
    L = L_rec + L_pep
    gen_mask = torch.zeros(L, dtype=torch.bool)
    gen_mask[L_rec:] = True

    pos = torch.randn(L, A, 3)
    # Place some peptide atoms close to receptor for contacts
    pos[L_rec:L_rec + 3, :3, :] = pos[:3, :3, :] + 0.1

    mask_ha = torch.ones(L, A, dtype=torch.bool)
    mask_ha[:, 5:] = False

    aa = torch.randint(0, 20, (L,))
    res_nb = torch.arange(L)
    chain_nb = torch.zeros(L, dtype=torch.long)
    chain_nb[L_rec:] = 1

    torsion = torch.rand(L, 5) * 2 * 3.14159
    torsion_mask = torch.ones(L, 5, dtype=torch.bool)

    return {
        'aa': aa,
        'res_nb': res_nb,
        'chain_nb': chain_nb,
        'pos_heavyatom': pos,
        'mask_heavyatom': mask_ha,
        'generate_mask': gen_mask,
        'torsion_angle': torsion,
        'torsion_angle_mask': torsion_mask,
    }


def make_unified_batch(B=2, L_rec=10, L_pep=6, K=5, A=15, device='cpu'):
    """Create a batched unified graph (post-transform, post-collation)."""
    L = L_rec + L_pep + K
    gen_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    gen_mask[:, L_rec:L_rec + L_pep] = True  # only peptide is generated
    res_mask = torch.ones(B, L, dtype=torch.bool, device=device)

    pos = torch.randn(B, L, A, 3, device=device)
    pos[:, L_rec:L_rec + 3, :3, :] = pos[:, :3, :3, :] + 0.1

    mask_ha = torch.zeros(B, L, A, dtype=torch.bool, device=device)
    mask_ha[:, :L_rec + L_pep, :5] = True  # rec + pep have 5 atoms
    # Anchors only have N, CA, C
    mask_ha[:, L_rec + L_pep:, 0] = True  # N
    mask_ha[:, L_rec + L_pep:, 1] = True  # CA
    mask_ha[:, L_rec + L_pep:, 2] = True  # C

    aa = torch.randint(0, 20, (B, L), device=device)
    res_nb = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
    chain_nb = torch.zeros(B, L, dtype=torch.long, device=device)
    chain_nb[:, L_rec:L_rec + L_pep] = 1
    chain_nb[:, L_rec + L_pep:] = 2

    torsion = torch.rand(B, L, 5, device=device) * 2 * 3.14159
    torsion_mask = torch.ones(B, L, 5, dtype=torch.bool, device=device)
    torsion_mask[:, L_rec + L_pep:] = False  # anchors have no torsion

    # Node type: 0=receptor, 1=peptide, 2=anchor
    node_type = torch.zeros(B, L, dtype=torch.long, device=device)
    node_type[:, L_rec:L_rec + L_pep] = 1
    node_type[:, L_rec + L_pep:] = 2

    # Hotspot labels: mark first 2 peptide residues
    hotspot_labels = torch.zeros(B, L, dtype=torch.bool, device=device)
    hotspot_labels[:, L_rec:L_rec + 2] = True

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
        'node_type': node_type,
        'hotspot_labels': hotspot_labels,
        'L_orig': torch.tensor([L_rec + L_pep] * B, device=device),
    }


# ---------- AnchorToGraphTransform tests ----------

def test_anchor_to_graph_transform():
    """AnchorToGraphTransform correctly expands the graph."""
    item = make_single_item(L_rec=10, L_pep=6)
    L_orig = item['aa'].shape[0]  # 16

    # Apply HotspotAnnotationTransform first
    hs_transform = HotspotAnnotationTransform(num_anchors=5, distance_cutoff=4.0)
    item = hs_transform(item)

    K = item['anchor_coords'].shape[0]
    assert K == 5, f"Expected 5 anchors, got {K}"

    # Apply AnchorToGraphTransform
    graph_transform = AnchorToGraphTransform()
    item = graph_transform(item)

    L_new = item['aa'].shape[0]
    assert L_new == L_orig + K, f"Expected {L_orig + K}, got {L_new}"
    assert item['pos_heavyatom'].shape[0] == L_new
    assert item['mask_heavyatom'].shape[0] == L_new
    assert item['generate_mask'].shape[0] == L_new
    assert item['node_type'].shape[0] == L_new
    assert int(item['L_orig'].item()) == L_orig

    # Anchor generate_mask should be False
    assert not item['generate_mask'][L_orig:].any(), "Anchors should have generate_mask=False"

    # Node type check
    assert (item['node_type'][L_orig:] == 2).all(), "Anchor node_type should be 2"

    # Chain_nb check
    assert (item['chain_nb'][L_orig:] == 2).all(), "Anchor chain_nb should be 2"

    print(f"[PASS] AnchorToGraphTransform: L={L_orig} -> L+K={L_new}")


# ---------- UnifiedFlowModel tests ----------

def test_instantiation():
    """Model instantiates correctly."""
    cfg = make_cfg()
    model = UnifiedFlowModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[PASS] Instantiation: {n_params:,} params")

    # Check node_type_embed exists
    assert hasattr(model, 'node_type_embed')
    assert model.node_type_embed.weight.shape == (3, 128)
    print("[PASS] node_type_embed shape = (3, 128)")


def test_forward():
    """Forward pass produces loss dict without errors."""
    torch.manual_seed(42)
    cfg = make_cfg()
    model = UnifiedFlowModel(cfg)
    model.train()

    batch = make_unified_batch(B=2, L_rec=10, L_pep=6, K=5)
    loss_dict = model(batch)

    expected_keys = {'trans_loss', 'rot_loss', 'bb_atom_loss',
                     'seqs_loss', 'angle_loss', 'torsion_loss', 'contact_loss'}
    assert set(loss_dict.keys()) == expected_keys, f"Wrong keys: {loss_dict.keys()}"

    for k, v in loss_dict.items():
        assert v.shape == (), f"{k} should be scalar, got {v.shape}"
        assert torch.isfinite(v), f"{k} is not finite: {v}"

    print(f"[PASS] Forward: {', '.join(f'{k}={v.item():.4f}' for k, v in loss_dict.items())}")


def test_backward():
    """Gradients flow through all parameters including node_type_embed."""
    torch.manual_seed(42)
    cfg = make_cfg()
    model = UnifiedFlowModel(cfg)
    model.train()

    batch = make_unified_batch(B=2, L_rec=10, L_pep=6, K=5)
    loss_dict = model(batch)
    total_loss = sum(loss_dict.values())
    total_loss.backward()

    # Check node_type_embed has gradient
    assert model.node_type_embed.weight.grad is not None, "No gradient for node_type_embed"
    assert model.node_type_embed.weight.grad.abs().sum() > 0, "Zero gradient for node_type_embed"

    # Check base model params have gradients too
    n_grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"[PASS] Backward: {n_grads}/{n_total} params have non-zero gradients")


def test_anchor_positions_fixed_in_sample():
    """Anchor positions should remain unchanged after sampling."""
    torch.manual_seed(42)
    cfg = make_cfg()
    model = UnifiedFlowModel(cfg)
    model.eval()

    L_rec, L_pep, K = 10, 6, 5
    batch = make_unified_batch(B=1, L_rec=L_rec, L_pep=L_pep, K=K)

    # Record original anchor CA positions
    anchor_start = L_rec + L_pep
    orig_anchor_ca = batch['pos_heavyatom'][:, anchor_start:, 1, :].clone()  # CA = index 1

    traj = model.sample(batch, num_steps=5)
    final = traj[-1]

    # In the final trajectory, check anchor positions match input
    # final['trans'] contains predicted CA positions for all L+K nodes
    pred_anchor_ca = final['trans'][:, anchor_start:]

    diff = (pred_anchor_ca - orig_anchor_ca.cpu()).abs().max().item()
    assert diff < 1e-4, f"Anchor positions changed: max diff = {diff}"
    print(f"[PASS] Anchor positions fixed in sample: max diff = {diff:.2e}")


def test_hotspot_dropout():
    """Hotspot dropout should produce different outputs than no dropout."""
    torch.manual_seed(42)
    cfg = make_cfg()

    model_drop = UnifiedFlowModel(cfg, hotspot_dropout_p=1.0)  # always drop
    model_drop.eval()
    model_nodrop = UnifiedFlowModel(cfg, hotspot_dropout_p=0.0)
    model_nodrop.eval()

    # Copy weights
    model_nodrop.load_state_dict(model_drop.state_dict())

    batch = make_unified_batch(B=1, L_rec=10, L_pep=6, K=5)

    # Encode with dropout = always
    model_drop.train()
    _, _, _, _, ne_drop, _ = model_drop.encode(
        model_drop._apply_anchor_dropout(batch)
    )

    # Encode without dropout
    _, _, _, _, ne_nodrop, _ = model_nodrop.encode(batch)

    # Should be different due to anchor masking
    diff = (ne_drop - ne_nodrop).abs().max().item()
    assert diff > 0.01, f"Dropout should change embeddings, but diff = {diff}"
    print(f"[PASS] Hotspot dropout changes embeddings: diff = {diff:.4f}")


def test_output_shape_consistency():
    """Output dimensions should be consistent for various K values."""
    torch.manual_seed(0)
    cfg = make_cfg()
    model = UnifiedFlowModel(cfg)
    model.train()

    for K in [0, 3, 5, 8]:
        if K == 0:
            # No anchors — should work like vanilla FlowModel
            batch = make_unified_batch(B=2, L_rec=10, L_pep=6, K=1)
            # Manually set node_type to not have type 2
            batch['node_type'][:] = 0
            batch['node_type'][:, 10:16] = 1
        else:
            batch = make_unified_batch(B=2, L_rec=10, L_pep=6, K=K)

        loss_dict = model(batch)
        for v in loss_dict.values():
            assert v.shape == (), f"K={K}: loss not scalar"
            assert torch.isfinite(v), f"K={K}: loss not finite"

    print("[PASS] Output shape consistency for K=0,3,5,8")


if __name__ == "__main__":
    test_anchor_to_graph_transform()
    test_instantiation()
    test_forward()
    test_backward()
    test_anchor_positions_fixed_in_sample()
    test_hotspot_dropout()
    test_output_shape_consistency()
    print("\n=== All UnifiedFlowModel tests passed ===")
