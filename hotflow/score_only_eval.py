"""Score generated PDBs without FastRelax — raw structure energy comparison."""

import argparse
import csv
import json
import re
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hotflow.utils.rosetta_score import score_only


def find_gen_pdbs(pdb_dir: Path):
    """Yield (pdb_path, peptide_chain) for *_gen.pdb files."""
    for p in sorted(pdb_dir.glob("*_gen.pdb")):
        m = re.match(r"(.+?)_([A-Z])_gen\.pdb", p.name)
        if m:
            yield p, m.group(2)


def score_method(label: str, pdb_dir: Path, also_relaxed: bool = True):
    rows = []
    gen_pdbs = list(find_gen_pdbs(pdb_dir))
    print(f"\n[{label}] Scoring {len(gen_pdbs)} raw PDBs ...")
    t0 = time.time()

    for i, (pdb_path, pep_chain) in enumerate(gen_pdbs):
        result = score_only(str(pdb_path), pep_chain)
        row = {
            "method": label,
            "pdb": pdb_path.stem,
            "type": "raw",
            "stab": result["stab"],
            "bind": result["bind"],
            "success": result["success"],
            "error": result.get("error"),
        }
        rows.append(row)

        if also_relaxed:
            relaxed = pdb_path.parent / pdb_path.name.replace("_gen.pdb", "_gen_relaxed.pdb")
            if relaxed.exists():
                r2 = score_only(str(relaxed), pep_chain)
                rows.append({
                    "method": label,
                    "pdb": relaxed.stem,
                    "type": "relaxed",
                    "stab": r2["stab"],
                    "bind": r2["bind"],
                    "success": r2["success"],
                    "error": r2.get("error"),
                })

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            per_sample = elapsed / (i + 1)
            remaining = per_sample * (len(gen_pdbs) - i - 1)
            print(f"  [{label}] {i+1}/{len(gen_pdbs)}  "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - t0
    print(f"  [{label}] Done in {elapsed:.1f}s")
    return rows


def summarize(rows, out_path: Path):
    from collections import defaultdict
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["success"] and r["stab"] is not None:
            groups[(r["method"], r["type"])]["stab"].append(r["stab"])
            groups[(r["method"], r["type"])]["bind"].append(r["bind"])

    print("\n" + "=" * 70)
    print(f"{'Method':<15} {'Type':<10} {'N':>5} {'Stab (REU)':>12} {'Bind (REU)':>12}")
    print("-" * 70)

    summary = {}
    for (method, typ), vals in sorted(groups.items()):
        n = len(vals["stab"])
        mean_stab = sum(vals["stab"]) / n
        mean_bind = sum(vals["bind"]) / n
        print(f"{method:<15} {typ:<10} {n:>5} {mean_stab:>12.1f} {mean_bind:>12.1f}")
        summary[f"{method}_{typ}"] = {
            "n": n, "mean_stab": mean_stab, "mean_bind": mean_bind,
        }

    print("=" * 70)

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str,
                        default="results/eval_se3_600k_approachB_pephar")
    parser.add_argument("--pepflow_dir", type=str,
                        default="results/compare_600k_gt_seeded/PepFlow/pdbs")
    parser.add_argument("--no_relaxed", action="store_true",
                        help="skip scoring relaxed PDBs (only raw)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    also_relaxed = not args.no_relaxed

    all_rows = []

    pepflow_dir = Path(args.pepflow_dir)
    if pepflow_dir.exists():
        all_rows.extend(score_method("PepFlow", pepflow_dir, also_relaxed))

    pephar_dir = out_dir / "PepHAR" / "pdbs"
    if pephar_dir.exists():
        all_rows.extend(score_method("PepHAR", pephar_dir, also_relaxed))

    appb_dir = out_dir / "Approach_B" / "pdbs"
    if appb_dir.exists():
        all_rows.extend(score_method("Approach_B", appb_dir, also_relaxed))

    csv_path = out_dir / "score_only.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "pdb", "type", "stab", "bind", "success", "error"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nPer-sample CSV saved to {csv_path}")

    summarize(all_rows, out_dir / "score_only_summary.json")


if __name__ == "__main__":
    main()
