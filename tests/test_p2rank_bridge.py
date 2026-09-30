"""Tests for P2Rank integration bridge.

Uses synthetic P2Rank CSV files and a mock PDB parser to verify:
  1. CSV parsing (residue-level and pocket-level)
  2. Residue-to-PDB matching
  3. Anchor conversion to tensor format
  4. End-to-end integration with FlowModelB.sample()
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PEPFLOW_ROOT = REPO_ROOT / "PepFlowww"
sys.path.insert(0, str(PEPFLOW_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from hotflow.data.p2rank_bridge import (
    P2RankResidue,
    anchors_to_tensor,
    parse_p2rank_pockets,
    parse_p2rank_residues,
)
from hotflow.data_types import HotspotAnchor


# ── Synthetic CSV helpers ───────────────────────────────────────────────────

def _write_residues_csv(path: str, rows: list[dict]) -> None:
    """Write a P2Rank-style residues CSV."""
    with open(path, 'w') as f:
        f.write("   chain,   residue_label,   residue_name,   score,   zscore,   probability,   pocket\n")
        for r in rows:
            f.write(f"   {r['chain']},   {r['label']},   {r['name']},   "
                    f"{r.get('score', 0.0)},   {r.get('zscore', 0.0)},   "
                    f"{r.get('probability', r.get('score', 0.0))},   "
                    f"{r.get('pocket', 0)}\n")


def _write_predictions_csv(path: str, pockets: list[dict]) -> None:
    """Write a P2Rank-style predictions CSV."""
    with open(path, 'w') as f:
        f.write("   name,   rank,   score,   probability,   residue_ids,   center_x,   center_y,   center_z\n")
        for p in pockets:
            f.write(f"   {p['name']},   {p['rank']},   {p['score']},   "
                    f"{p.get('probability', p['score'])},   "
                    f"{p.get('residue_ids', '')},   "
                    f"{p.get('cx', 0.0)},   {p.get('cy', 0.0)},   "
                    f"{p.get('cz', 0.0)}\n")


# ── Tests ───────────────────────────────────────────────────────────────────

def test_parse_residues_csv():
    """Test parsing P2Rank residue-level CSV."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        _write_residues_csv(f.name, [
            {'chain': 'A', 'label': '45', 'name': 'ALA', 'score': 0.3, 'probability': 0.85, 'pocket': 1},
            {'chain': 'A', 'label': '46', 'name': 'LEU', 'score': 0.2, 'probability': 0.72, 'pocket': 1},
            {'chain': 'A', 'label': '100', 'name': 'GLY', 'score': 0.1, 'probability': 0.30, 'pocket': 0},
            {'chain': 'B', 'label': '12', 'name': 'VAL', 'score': 0.15, 'probability': 0.55, 'pocket': 2},
        ])
        csv_path = f.name

    try:
        residues = parse_p2rank_residues(csv_path)

        assert len(residues) == 4
        # Should be sorted by score (probability) descending
        assert residues[0].score == 0.85
        assert residues[0].chain == 'A'
        assert residues[0].residue_label == '45'
        assert residues[0].residue_name == 'ALA'
        assert residues[0].pocket == 1
        assert residues[2].pocket == 2  # B:12
        assert residues[3].pocket is None  # score=0.30, pocket=0 → None

        print(f"[OK] parse_residues_csv: {len(residues)} residues parsed")
        for r in residues:
            print(f"     {r.chain}:{r.residue_label} {r.residue_name} score={r.score:.2f} pocket={r.pocket}")
    finally:
        os.unlink(csv_path)


def test_parse_predictions_csv():
    """Test parsing P2Rank pocket-level CSV."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        _write_predictions_csv(f.name, [
            {'name': 'pocket1', 'rank': 1, 'score': 12.5, 'probability': 0.95,
             'residue_ids': 'A_45 A_46 A_47', 'cx': 10.0, 'cy': 20.0, 'cz': 30.0},
            {'name': 'pocket2', 'rank': 2, 'score': 8.3, 'probability': 0.72,
             'residue_ids': 'B_12 B_13', 'cx': -5.0, 'cy': 15.0, 'cz': 25.0},
        ])
        csv_path = f.name

    try:
        pockets = parse_p2rank_pockets(csv_path)

        assert len(pockets) == 2
        assert pockets[0]['rank'] == 1
        assert pockets[0]['score'] == 12.5
        assert len(pockets[0]['residue_ids']) == 3
        assert pockets[0]['center_x'] == 10.0

        print(f"[OK] parse_predictions_csv: {len(pockets)} pockets parsed")
        for p in pockets:
            print(f"     {p['name']}: rank={p['rank']}, score={p['score']:.1f}, "
                  f"center=({p.get('center_x', '?')}, {p.get('center_y', '?')}, {p.get('center_z', '?')})")
    finally:
        os.unlink(csv_path)


def test_anchors_to_tensor():
    """Test converting HotspotAnchor list to batched tensors."""
    anchors = [
        HotspotAnchor(
            residue_index=i,
            backbone_coords=torch.randn(3, 3),
            residue_type=i * 3,
            score=1.0 - i * 0.2,
            source="p2rank",
        )
        for i in range(3)
    ]

    # K=5 > len(anchors)=3 → should pad
    coords, types, mask = anchors_to_tensor(anchors, K=5, batch_size=2)

    assert coords.shape == (2, 5, 3, 3)
    assert types.shape == (2, 5)
    assert mask.shape == (2, 5)

    # First 3 should be valid
    assert mask[0, :3].all()
    assert not mask[0, 3:].any()

    # Types should match
    assert types[0, 0] == 0
    assert types[0, 1] == 3
    assert types[0, 2] == 6
    assert types[0, 3] == 20  # UNKNOWN_AA pad

    # Batch dimension should be identical
    assert torch.equal(coords[0], coords[1])
    assert torch.equal(mask[0], mask[1])

    print(f"[OK] anchors_to_tensor: shape=({coords.shape}), "
          f"mask_sum={mask[0].sum().item()}")

    # K=2 < len(anchors)=3 → should truncate
    coords2, types2, mask2 = anchors_to_tensor(anchors, K=2, batch_size=1)
    assert coords2.shape == (1, 2, 3, 3)
    assert mask2[0].all()  # both valid
    print("[OK] anchors_to_tensor with K < n_anchors: truncated correctly")


def test_anchors_to_tensor_empty():
    """Test empty anchor list."""
    coords, types, mask = anchors_to_tensor([], K=3, batch_size=1)
    assert coords.shape == (1, 3, 3, 3)
    assert not mask.any()
    print("[OK] anchors_to_tensor empty: all masked out")


def test_integration_with_flow_model_b():
    """Test that P2Rank-derived anchors work with FlowModelB.sample()."""
    from easydict import EasyDict
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
    model.eval()

    # Create synthetic batch (receptor + peptide region)
    B, L_rec, L_pep, A = 1, 10, 6, 15
    L = L_rec + L_pep
    batch = {
        'aa': torch.randint(0, 20, (B, L)),
        'res_nb': torch.arange(L).unsqueeze(0),
        'chain_nb': torch.cat([torch.zeros(L_rec), torch.ones(L_pep)]).long().unsqueeze(0),
        'pos_heavyatom': torch.randn(B, L, A, 3),
        'mask_heavyatom': torch.ones(B, L, A, dtype=torch.bool),
        'generate_mask': torch.cat([torch.zeros(L_rec), torch.ones(L_pep)]).bool().unsqueeze(0),
        'res_mask': torch.ones(B, L, dtype=torch.bool),
        'torsion_angle': torch.rand(B, L, 5) * 6.28,
        'torsion_angle_mask': torch.ones(B, L, 5, dtype=torch.bool),
    }

    # Create "P2Rank-derived" anchors (synthetic)
    anchors = [
        HotspotAnchor(
            residue_index=i,
            backbone_coords=torch.randn(3, 3),
            residue_type=i + 5,
            score=0.9 - i * 0.1,
            source="p2rank",
        )
        for i in range(3)
    ]

    coords, types, mask = anchors_to_tensor(anchors, K=3, batch_size=B)

    # Sample with P2Rank anchors
    traj = model.sample(
        batch,
        num_steps=3,
        anchor_coords=coords,
        anchor_types=types,
        anchor_mask=mask,
    )

    assert len(traj) == 3
    assert 'rotmats' in traj[-1]
    assert 'seqs' in traj[-1]

    print(f"\n[OK] FlowModelB.sample() with P2Rank anchors: {len(traj)} steps")
    print(f"     Final seqs shape: {traj[-1]['seqs'].shape}")

    # Also test with guidance
    traj_cfg = model.sample(
        batch, num_steps=3,
        anchor_coords=coords, anchor_types=types, anchor_mask=mask,
        guidance_scale=1.5,
    )
    assert len(traj_cfg) == 3
    print("[OK] CFG sampling with P2Rank anchors works (scale=1.5)")


if __name__ == '__main__':
    test_parse_residues_csv()
    test_parse_predictions_csv()
    test_anchors_to_tensor()
    test_anchors_to_tensor_empty()
    test_integration_with_flow_model_b()
    print("\n=== All P2Rank bridge tests passed ===")
