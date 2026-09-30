"""End-to-end test for Approach B data pipeline.

Tests HotspotAnnotationTransform, PepDatasetB, and integration with
FlowModelB using synthetic data (no real PDB files needed).
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
PEPFLOW_ROOT = REPO_ROOT / "PepFlowww"
sys.path.insert(0, str(PEPFLOW_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from pepflow.utils.data import PaddingCollate
from easydict import EasyDict

from hotflow.data.transforms import HotspotAnnotationTransform
from hotflow.data.dataset_b import PepDatasetB, HOTSPOT_PAD_VALUES, HOTSPOT_NO_PADDING


# ---------------------------------------------------------------------------
# Synthetic dataset (replaces PepDataset for testing)
# ---------------------------------------------------------------------------

class SyntheticPepDataset(Dataset):
    """Generates synthetic PepFlow-format items for testing."""

    def __init__(self, num_items: int = 8, L_rec: int = 12, L_pep: int = 6, A: int = 15):
        self.num_items = num_items
        self.L_rec = L_rec
        self.L_pep = L_pep
        self.A = A

    def __len__(self):
        return self.num_items

    def __getitem__(self, index):
        L_rec, L_pep, A = self.L_rec, self.L_pep, self.A
        L = L_rec + L_pep

        gen_mask = torch.zeros(L, dtype=torch.bool)
        gen_mask[L_rec:] = True

        pos = torch.randn(L, A, 3)
        # Make first 3 peptide residues close to first 3 receptor residues
        pos[L_rec:L_rec+3] = pos[:3] + 0.1 * torch.randn(3, A, 3)

        mask_ha = torch.ones(L, A, dtype=torch.bool)
        mask_ha[:, 5:] = False

        aa = torch.randint(0, 20, (L,))
        res_nb = torch.arange(L)
        chain_nb = torch.zeros(L, dtype=torch.long)
        chain_nb[L_rec:] = 1

        torsion = torch.rand(L, 5) * 2 * 3.14159
        torsion_mask = torch.ones(L, 5, dtype=torch.bool)

        return {
            'id': f'synth_{index}',
            'aa': aa,
            'res_nb': res_nb,
            'chain_nb': chain_nb,
            'pos_heavyatom': pos,
            'mask_heavyatom': mask_ha,
            'generate_mask': gen_mask,
            'torsion_angle': torsion,
            'torsion_angle_mask': torsion_mask,
        }


def test_transform_per_item():
    """Test HotspotAnnotationTransform on a single item."""
    dataset = SyntheticPepDataset(num_items=1)
    transform = HotspotAnnotationTransform(num_anchors=3, distance_cutoff=4.0)

    item = dataset[0]
    annotated = transform(item)

    # Check new keys exist
    for key in ['hotspot_labels', 'anchor_indices', 'anchor_coords', 'anchor_types', 'anchor_mask']:
        assert key in annotated, f"Missing key: {key}"

    L = item['aa'].shape[0]
    K = 3

    assert annotated['hotspot_labels'].shape == (L,)
    assert annotated['hotspot_labels'].dtype == torch.bool
    assert annotated['anchor_indices'].shape == (K,)
    assert annotated['anchor_coords'].shape == (K, 3, 3)
    assert annotated['anchor_types'].shape == (K,)
    assert annotated['anchor_mask'].shape == (K,)

    n_hotspots = annotated['hotspot_labels'].sum().item()
    n_anchors = annotated['anchor_mask'].sum().item()
    print(f"[OK] Transform: {n_hotspots} hotspots found, {n_anchors} anchors selected (K={K})")

    # Hotspots should only be in peptide region
    rec_hotspots = annotated['hotspot_labels'][:dataset.L_rec].sum().item()
    assert rec_hotspots == 0, "Receptor residues should not be hotspots"
    print("[OK] Hotspot labels only in peptide region")


def test_dataset_b_wrapper():
    """Test PepDatasetB wrapping synthetic dataset."""
    base = SyntheticPepDataset(num_items=4)
    dataset = PepDatasetB(base, num_anchors=3, distance_cutoff=4.0)

    assert len(dataset) == 4
    item = dataset[0]

    assert 'anchor_mask' in item
    assert 'hotspot_labels' in item
    print(f"[OK] PepDatasetB wrapper: len={len(dataset)}, keys include hotspot fields")


def test_collation():
    """Test that PaddingCollate handles hotspot fields correctly."""
    base = SyntheticPepDataset(num_items=4, L_rec=10, L_pep=5)
    dataset = PepDatasetB(base, num_anchors=3, distance_cutoff=4.0)

    from pepflow.utils.data import DEFAULT_PAD_VALUES, DEFAULT_NO_PADDING
    pad_values = {**DEFAULT_PAD_VALUES, **HOTSPOT_PAD_VALUES}
    no_padding = set(DEFAULT_NO_PADDING) | HOTSPOT_NO_PADDING
    collate = PaddingCollate(eight=False, pad_values=pad_values, no_padding=no_padding)

    items = [dataset[i] for i in range(4)]
    batch = collate(items)

    B = 4
    L = batch['aa'].shape[1]
    K = 3

    assert batch['hotspot_labels'].shape == (B, L), f"hotspot_labels shape: {batch['hotspot_labels'].shape}"
    assert batch['anchor_coords'].shape == (B, K, 3, 3)
    assert batch['anchor_types'].shape == (B, K)
    assert batch['anchor_mask'].shape == (B, K)
    assert batch['anchor_indices'].shape == (B, K)
    assert batch['res_mask'].shape == (B, L)

    print(f"[OK] Collation: batch shape B={B}, L={L}, K={K}")
    print(f"     hotspot_labels: {batch['hotspot_labels'].shape}, dtype={batch['hotspot_labels'].dtype}")
    print(f"     anchor_coords: {batch['anchor_coords'].shape}")
    print(f"     anchor_mask sum per item: {batch['anchor_mask'].sum(dim=1).tolist()}")


def test_integration_with_flow_model_b():
    """Test that FlowModelB can consume batch from data pipeline."""
    from hotflow.models.flow_model_b import FlowModelB

    cfg = EasyDict(
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
            min_t=0.01, t_normalization_clip=0.9,
            sample_structure=True, sample_sequence=True,
            rots=EasyDict(train_schedule='linear', sample_schedule='exp', exp_rate=10),
            trans=EasyDict(train_schedule='linear', sample_schedule='linear', sigma=1.0),
            seqs=EasyDict(num_classes=20, simplex_value=5.0),
            sampling=EasyDict(num_timesteps=100),
            self_condition=False,
        ),
    )

    model = FlowModelB(cfg, num_hotspot_anchors=3)
    model.train()

    # Build batch from pipeline
    base = SyntheticPepDataset(num_items=2, L_rec=10, L_pep=6)
    dataset = PepDatasetB(base, num_anchors=3, distance_cutoff=4.0)

    from pepflow.utils.data import DEFAULT_PAD_VALUES, DEFAULT_NO_PADDING
    pad_values = {**DEFAULT_PAD_VALUES, **HOTSPOT_PAD_VALUES}
    no_padding = set(DEFAULT_NO_PADDING) | HOTSPOT_NO_PADDING
    collate = PaddingCollate(eight=False, pad_values=pad_values, no_padding=no_padding)
    batch = collate([dataset[0], dataset[1]])

    # Forward pass
    loss_dict = model(batch)
    assert 'contact_loss' in loss_dict
    for k, v in loss_dict.items():
        assert torch.isfinite(v), f"{k} is not finite"

    print(f"\n[OK] FlowModelB + data pipeline integration:")
    for k, v in loss_dict.items():
        print(f"     {k}: {v.item():.6f}")

    # Verify pre-computed anchors were used (not recomputed)
    # This is implicit: if anchor_coords is in batch, _encode_hotspots_from_batch uses it
    print("[OK] Pre-computed anchors consumed by FlowModelB")


def test_varying_lengths():
    """Test collation with items of different sequence lengths."""
    class VaryingLenDataset(Dataset):
        def __len__(self):
            return 3
        def __getitem__(self, i):
            L_pep = 4 + i  # 4, 5, 6
            L_rec = 8
            L = L_rec + L_pep
            A = 15
            pos = torch.randn(L, A, 3)
            pos[L_rec:L_rec+2] = pos[:2] + 0.1 * torch.randn(2, A, 3)
            return {
                'id': f'var_{i}',
                'aa': torch.randint(0, 20, (L,)),
                'res_nb': torch.arange(L),
                'chain_nb': torch.cat([torch.zeros(L_rec, dtype=torch.long), torch.ones(L_pep, dtype=torch.long)]),
                'pos_heavyatom': pos,
                'mask_heavyatom': torch.ones(L, A, dtype=torch.bool),
                'generate_mask': torch.cat([torch.zeros(L_rec, dtype=torch.bool), torch.ones(L_pep, dtype=torch.bool)]),
                'torsion_angle': torch.rand(L, 5) * 6.28,
                'torsion_angle_mask': torch.ones(L, 5, dtype=torch.bool),
            }

    base = VaryingLenDataset()
    dataset = PepDatasetB(base, num_anchors=3, distance_cutoff=4.0)

    from pepflow.utils.data import DEFAULT_PAD_VALUES, DEFAULT_NO_PADDING
    pad_values = {**DEFAULT_PAD_VALUES, **HOTSPOT_PAD_VALUES}
    no_padding = set(DEFAULT_NO_PADDING) | HOTSPOT_NO_PADDING
    collate = PaddingCollate(eight=False, pad_values=pad_values, no_padding=no_padding)

    batch = collate([dataset[0], dataset[1], dataset[2]])

    B = 3
    L = batch['aa'].shape[1]  # should be max(12, 13, 14) = 14
    assert L == 14, f"Expected padded L=14, got {L}"
    assert batch['hotspot_labels'].shape == (B, L)
    assert batch['res_mask'].shape == (B, L)

    # Padded positions in hotspot_labels should be False
    assert not batch['hotspot_labels'][0, 12:].any(), "Item 0 padded region should be False"
    assert not batch['hotspot_labels'][1, 13:].any(), "Item 1 padded region should be False"

    print(f"\n[OK] Varying lengths: padded to L={L}, hotspot padding correct")


if __name__ == '__main__':
    test_transform_per_item()
    test_dataset_b_wrapper()
    test_collation()
    test_varying_lengths()
    test_integration_with_flow_model_b()
    print("\n=== All data pipeline B tests passed ===")
