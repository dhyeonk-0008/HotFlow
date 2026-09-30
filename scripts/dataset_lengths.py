"""Dump receptor/peptide length distribution of PepBDB LMDB datasets to CSV.

Run:
    PYTHONPATH=$PWD:$PWD/PepFlowww python scripts/dataset_lengths.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "PepFlowww") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "PepFlowww"))

import csv

import yaml
from easydict import EasyDict

from models_con.pep_dataloader import PepDataset


def collect(name, structure_dir, dataset_dir, lmdb_name):
    ds = PepDataset(
        structure_dir=structure_dir,
        dataset_dir=dataset_dir,
        name=lmdb_name,
        transform=None,
        reset=False,
    )
    rows = []
    for i in range(len(ds)):
        try:
            d = ds[i]
        except Exception as e:
            print(f"  [{name}] entry {i}: load failed ({e})")
            continue
        sample_id = d.get("id", str(i))
        gen = d["generate_mask"].bool()
        res = d.get("res_mask", None)
        if res is None:
            res_b = gen | ~gen   # all True
        else:
            res_b = res.bool()
        pep_len = int((gen & res_b).sum().item())
        rec_len = int((~gen & res_b).sum().item())
        total = int(res_b.sum().item())
        rows.append((sample_id, pep_len, rec_len, total))
    return rows


def write_csv(rows, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "pep_length", "rec_length", "total_length"])
        for r in rows:
            w.writerow(r)


def summarize(name, rows):
    if not rows:
        print(f"[{name}] empty")
        return
    import statistics as st
    peps = [r[1] for r in rows]
    recs = [r[2] for r in rows]
    print(f"--- {name} (n={len(rows)}) ---")
    print(f"  peptide len: min={min(peps)}  max={max(peps)}  median={st.median(peps)}  mean={st.mean(peps):.1f}")
    print(f"  receptor len: min={min(recs)}  max={max(recs)}  median={st.median(recs)}  mean={st.mean(recs):.1f}")
    # Receptor length percentile breakdown
    rec_sorted = sorted(recs)
    pct = lambda p: rec_sorted[int(len(rec_sorted) * p / 100)]
    print(f"  receptor len percentiles: 10={pct(10)}  25={pct(25)}  50={pct(50)}  75={pct(75)}  90={pct(90)}")
    # Tiny receptors
    tiny = sum(1 for r in recs if r <= 10)
    short = sum(1 for r in recs if r <= 20)
    print(f"  receptors with len ≤ 10: {tiny} ({100*tiny/len(recs):.1f}%)")
    print(f"  receptors with len ≤ 20: {short} ({100*short/len(recs):.1f}%)")


def main():
    with open(REPO_ROOT / "hotflow/configs/train_b.yaml") as f:
        cfg = EasyDict(yaml.safe_load(f))

    out_dir = REPO_ROOT / "results/dataset_lengths"
    out_dir.mkdir(parents=True, exist_ok=True)

    train = collect("train", cfg.dataset.train.structure_dir,
                    cfg.dataset.train.dataset_dir, cfg.dataset.train.name)
    write_csv(train, out_dir / "train_lengths.csv")
    summarize("train", train)

    val = collect("test", cfg.dataset.val.structure_dir,
                  cfg.dataset.val.dataset_dir, cfg.dataset.val.name)
    write_csv(val, out_dir / "test_lengths.csv")
    summarize("test", val)

    print(f"\nWrote CSVs to {out_dir}/")


if __name__ == "__main__":
    main()
