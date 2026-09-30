"""Pre-compute PepHAR EBM anchors for Approach B training.

Produces an LMDB cache keyed by `data['id']`. Each value is a pickled dict:

    {
        "anchor_coords": Tensor (N, K, 3, 3)  [CA, C, N]
        "anchor_types":  Tensor (N, K)         long, [0..19]
        "anchor_mask":   Tensor (N, K)         bool
    }

N = stochastic samples per entry, K = number of anchors.

Per-entry workflow (mirrors `pephar_gt_seeded` in eval_compare.py):

    1. Pick top-K peptide residues by GT contact frequency
       (same selection as `HotspotAnnotationTransform`).
    2. For each of N samples:
        For each anchor j in K:
            coord, aa = sampler._rand_anchors(GT_coord_j, GT_aa_j)
            coord, aa = sampler._sample_anchor(rec_data, coord, aa, n_steps)
       Collect into (K, 3, 3) and (K,).

The script is resumable: entries already present in the output LMDB are
skipped on rerun.

Usage:
    cd /path/to/HotFlow
    export PYTHONPATH=$PWD

    python hotflow/scripts/precompute_pephar_anchors.py \
        --config hotflow/configs/train_b.yaml \
        --split train \
        --out data/processed/pepbdb/pephar_anchors_train.lmdb \
        --num_samples 1 \
        --anchor_steps 100 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import lmdb
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
PEPFLOW_ROOT = REPO_ROOT / "PepFlowww"
PEPHAR_ROOT = REPO_ROOT / "PepHAR"
_script_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _script_dir]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PEPFLOW_ROOT))
sys.path.insert(0, str(PEPHAR_ROOT))

from pepflow.utils.misc import load_config, seed_all  # noqa: E402
from pepflow.utils.train import recursive_to  # noqa: E402
from pepflow.modules.protein.constants import BBHeavyAtom  # noqa: E402
from models_con.pep_dataloader import PepDataset  # noqa: E402

from hotflow.data.transforms import HotspotAnnotationTransform  # noqa: E402


def _build_pephar_sampler(args, device):
    """Instantiate PepHAR's AnchorBasedSamplerDenovo (density + prediction)."""
    from evaluate.sample_revised import AnchorBasedSamplerDenovo

    class _Args:
        pass

    a = _Args()
    a.density_config_path = str(REPO_ROOT / args.pephar_density_config)
    a.density_param_path = str(REPO_ROOT / args.pephar_density_weights)
    a.prediction_config_path = str(REPO_ROOT / args.pephar_prediction_config)
    a.prediction_param_path = str(REPO_ROOT / args.pephar_prediction_weights)
    a.device = str(device)
    return AnchorBasedSamplerDenovo(a)


def _build_pephar_data(entry, device):
    """Slice receptor + GT peptide from an unpadded PepDataset entry.

    Returns the dict that PepHAR's `_sample_anchor` expects (rec_coord,
    rec_aa, pep_coord, pep_aa — all on `device`, CA/C/N coord order).
    """
    gen_mask = entry["generate_mask"].bool().cpu()
    pep_idx = gen_mask.nonzero(as_tuple=True)[0]
    rec_idx = (~gen_mask).nonzero(as_tuple=True)[0]

    pos = entry["pos_heavyatom"].cpu()
    aa = entry["aa"].cpu()
    bb = [BBHeavyAtom.CA, BBHeavyAtom.C, BBHeavyAtom.N]

    pephar_data = {
        "rec_coord": pos[rec_idx][:, bb],
        "rec_aa": aa[rec_idx],
        "pep_coord": pos[pep_idx][:, bb],
        "pep_aa": aa[pep_idx],
    }
    return recursive_to(pephar_data, device), pep_idx.numel(), rec_idx.numel()


def _existing_keys(env):
    """Return set of bytes keys already present in the LMDB."""
    keys = set()
    with env.begin() as txn:
        cursor = txn.cursor()
        for k, _ in cursor:
            keys.add(bytes(k))
    return keys


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute PepHAR EBM anchors for Approach B training."
    )
    parser.add_argument("--config", type=str, default="hotflow/configs/train_b.yaml",
                        help="Training config — reads dataset paths and "
                             "hotspot.{num_anchors, contact_distance_cutoff}.")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val"],
                        help="Which dataset split to pre-compute (train or val).")
    parser.add_argument("--out", type=str, required=True,
                        help="Output LMDB path.")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="N stochastic anchor samples per entry. "
                             "N=1: ~26h on train set (single GPU). "
                             "N=5: ~5x. Default 1.")
    parser.add_argument("--anchor_steps", type=int, default=100,
                        help="EBM Adam steps per anchor.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map_size_gb", type=int, default=4,
                        help="LMDB max map size in GB (overprovisioned — "
                             "actual usage is ~2MB for full train set).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N entries (debug).")
    parser.add_argument("--shard", type=str, default=None,
                        help="Stride-based shard 'k/N' — this process handles "
                             "entry indices where (idx %% N) == k. Use to "
                             "parallelize across GPUs. Two shards 0/2 and 1/2 "
                             "together cover the whole dataset. Default: full.")
    parser.add_argument("--pephar_density_config", type=str,
                        default="PepHAR/ckpts/density_v4_x5o2_2024_09_08__11_25_36/density_v4_x5o2.yml")
    parser.add_argument("--pephar_density_weights", type=str,
                        default="PepHAR/ckpts/density_v4_x5o2_2024_09_08__11_25_36/checkpoints/1400.pt")
    parser.add_argument("--pephar_prediction_config", type=str,
                        default="PepHAR/ckpts/prediction_d2_x2o1_2024_09_08__11_21_33/prediction_d2_x2o1.yml")
    parser.add_argument("--pephar_prediction_weights", type=str,
                        default="PepHAR/ckpts/prediction_d2_x2o1_2024_09_08__11_21_33/checkpoints/2400.pt")
    args = parser.parse_args()

    seed_all(args.seed)
    device = torch.device(args.device)

    config, _ = load_config(args.config)
    hs_cfg = config.hotspot
    K = int(hs_cfg.num_anchors)
    cutoff = float(hs_cfg.contact_distance_cutoff)

    if args.split == "train":
        ds_cfg = config.dataset.train
    else:
        ds_cfg = config.dataset.val

    print(f"[Data] split={args.split}  structure_dir={ds_cfg.structure_dir}")
    base = PepDataset(
        structure_dir=ds_cfg.structure_dir,
        dataset_dir=ds_cfg.dataset_dir,
        name=ds_cfg.name,
        transform=None,
        reset=False,
    )
    print(f"[Data] {len(base)} entries")

    gt_transform = HotspotAnnotationTransform(num_anchors=K, distance_cutoff=cutoff)

    print(f"[PepHAR] loading sampler on {device}")
    sampler = _build_pephar_sampler(args, device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    map_size = args.map_size_gb * 1024 ** 3
    env = lmdb.open(str(out_path), map_size=map_size, subdir=False,
                    readonly=False, lock=True, max_dbs=1)

    done_keys = _existing_keys(env)
    print(f"[Cache] {len(done_keys)} entries already in {out_path}")

    # Compute the index list for this shard
    if args.shard is not None:
        k_str, N_str = args.shard.split("/")
        shard_k, shard_N = int(k_str), int(N_str)
        if not (0 <= shard_k < shard_N):
            raise ValueError(f"Invalid --shard '{args.shard}': k must be in [0, N)")
        indices = list(range(shard_k, len(base), shard_N))
        shard_tag = f" [shard {shard_k}/{shard_N}]"
    else:
        indices = list(range(len(base)))
        shard_tag = ""

    if args.limit is not None:
        indices = indices[:args.limit]

    n_total = len(indices)

    skipped_no_peptide = 0
    skipped_no_anchor = 0
    written = 0
    t_start = time.time()

    pbar = tqdm(indices, total=n_total,
                desc=f"PepHAR EBM (N={args.num_samples}, "
                     f"steps={args.anchor_steps}){shard_tag}")
    for i in pbar:
        try:
            entry = base[i]
        except Exception as e:
            print(f"\n  [load fail] idx={i}: {e}")
            continue

        entry_id = entry.get("id", None)
        if entry_id is None:
            print(f"\n  [no id] idx={i}, skipping")
            continue
        key = entry_id.encode("utf-8") if isinstance(entry_id, str) else bytes(entry_id)
        if key in done_keys:
            continue

        # Step 1: GT-based top-K selection
        annotated = gt_transform(entry)
        gt_coords = annotated["anchor_coords"]   # (K, 3, 3)
        gt_types = annotated["anchor_types"]     # (K,)
        gt_mask = annotated["anchor_mask"]       # (K,)

        if not bool(gt_mask.any()):
            # Peptide has no contact-based hotspot — skip (no anchors to seed)
            skipped_no_anchor += 1
            continue

        # Step 2: receptor/peptide dict for PepHAR
        try:
            pephar_data, pep_len, rec_len = _build_pephar_data(entry, device)
        except Exception as e:
            print(f"\n  [pephar_data fail] {entry_id}: {e}")
            skipped_no_peptide += 1
            continue

        if pep_len == 0 or rec_len == 0:
            skipped_no_peptide += 1
            continue

        # Step 3: N stochastic samples × K anchors
        all_coords = torch.zeros(args.num_samples, K, 3, 3)
        all_types = torch.full((args.num_samples, K), 20, dtype=torch.long)
        all_mask = torch.zeros(args.num_samples, K, dtype=torch.bool)

        for n in range(args.num_samples):
            for j in range(K):
                if not bool(gt_mask[j]):
                    continue
                coord = gt_coords[j].to(device)
                aa = gt_types[j].to(device)
                # PepHAR baseline init: GT + Gaussian noise + random AA
                coord, aa = sampler._rand_anchors(coord, aa)
                # PepHAR baseline opt: EBM Adam + per-step Langevin
                if args.anchor_steps > 0:
                    coord, aa = sampler._sample_anchor(
                        pephar_data, coord, aa,
                        n_steps=args.anchor_steps,
                    )
                all_coords[n, j] = coord.detach().cpu()
                all_types[n, j] = int(aa.detach().cpu().item())
                all_mask[n, j] = True

        cached = {
            "anchor_coords": all_coords,
            "anchor_types": all_types,
            "anchor_mask": all_mask,
        }
        payload = pickle.dumps(cached, protocol=pickle.HIGHEST_PROTOCOL)
        with env.begin(write=True) as txn:
            txn.put(key, payload)
        written += 1
        done_keys.add(key)

        # Periodic flush; lmdb is durable anyway, but report progress
        if written % 50 == 0:
            elapsed = time.time() - t_start
            rate = written / max(elapsed, 1e-6)
            pbar.set_postfix({"written": written, "rate_/s": f"{rate:.3f}"})

    env.sync()
    env.close()

    elapsed = time.time() - t_start
    print(f"\n[Done] processed={n_total}, written={written}, "
          f"skipped_no_anchor={skipped_no_anchor}, "
          f"skipped_no_peptide={skipped_no_peptide}")
    print(f"       elapsed={elapsed/60:.1f} min  "
          f"({(written/max(elapsed,1e-6)):.3f} entries/s)")


if __name__ == "__main__":
    main()
