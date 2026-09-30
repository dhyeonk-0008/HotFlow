"""Sweep Approach_B checkpoints to pick the best one for full evaluation.

Runs a *quick* per-checkpoint evaluation on a small subset of the test set
without Rosetta — the goal is to rank checkpoints, not to produce
publication-quality numbers. Once a winner is identified, run the full
`eval_compare.py` pipeline (with Rosetta + relax-based metrics) on it.

Output:
- `<outdir>/ckpt_sweep.csv` — one row per checkpoint with mean metrics
- `<outdir>/<ckpt_name>/metrics.csv` — per-sample raw metrics
- Stdout ranking table sorted by the chosen criterion

Usage:
    python -m hotflow.select_ckpt \\
        --config hotflow/configs/train_b.yaml \\
        --ckpt_dir logs_b/train_b_2026_04_14__21_22_14/checkpoints \\
        --every 100000 \\
        --num_samples 30 \\
        --num_steps 50 \\
        --guidance_scale 1.0 \\
        --device cuda:0 \\
        --outdir results/ckpt_sweep \\
        --rank_by rmsd_aligned
"""

from __future__ import annotations

import argparse
import gc
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
PEPFLOW_ROOT = REPO_ROOT / "PepFlowww"
_script_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _script_dir]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PEPFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(PEPFLOW_ROOT))

from pepflow.utils.misc import load_config, seed_all  # noqa: E402

from hotflow.eval_compare import (  # noqa: E402
    build_approach_b,
    build_eval_dataloader,
    sample_approach_b,
)


# Lower-is-better metrics; higher-is-better is the complement.
LOWER_IS_BETTER = {
    "rmsd_unaligned", "rmsd_aligned", "interface_rmsd", "anchor_min_dist",
}
DEFAULT_REPORT_COLS = [
    "rmsd_unaligned", "rmsd_aligned", "aar",
    "anchor_contact_4A", "anchor_contact_6A", "anchor_min_dist",
    "interface_rmsd", "binding_site_overlap",
]


def find_checkpoints(
    ckpt_dir: Path,
    every: int | None = None,
    min_iter: int | None = None,
    max_iter: int | None = None,
    explicit: list[str] | None = None,
) -> list[Path]:
    """List checkpoints to evaluate, ordered by training step."""
    if explicit:
        return [Path(p) for p in explicit]

    pat = re.compile(r"(\d+)\.pt$")
    pts = []
    for p in Path(ckpt_dir).glob("*.pt"):
        m = pat.search(p.name)
        if not m:
            continue
        n = int(m.group(1))
        if min_iter is not None and n < min_iter:
            continue
        if max_iter is not None and n > max_iter:
            continue
        if every and (n % every != 0):
            continue
        pts.append((n, p))
    pts.sort(key=lambda x: x[0])
    return [p for _, p in pts]


def evaluate_checkpoint(
    ckpt_path: Path,
    config,
    loader,
    num_samples: int,
    num_steps: int,
    guidance_scale: float,
    device: torch.device,
) -> tuple[pd.DataFrame, dict]:
    """Load one checkpoint, run `num_samples` test samples, return df + summary."""
    model, iteration = build_approach_b(config, str(ckpt_path), device)

    results = []
    for i, batch in enumerate(tqdm(loader, total=num_samples,
                                    desc=f"  {ckpt_path.name}",
                                    leave=False)):
        if i >= num_samples:
            break
        try:
            metrics, _ = sample_approach_b(
                model, batch, device,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
            )
            sample_id = (batch.get("id", [f"sample_{i:04d}"])[0]
                         if "id" in batch else f"sample_{i:04d}")
            metrics["sample_id"] = sample_id
            results.append(metrics)
        except Exception as e:
            print(f"    [SKIP] sample {i}: {e}")

    # Free GPU memory before loading the next checkpoint.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    summary: dict = {
        "checkpoint": ckpt_path.name,
        "iteration": (int(re.search(r"(\d+)\.pt$", ckpt_path.name).group(1))
                      if re.search(r"(\d+)\.pt$", ckpt_path.name) else -1),
        "n_evaluated": len(df),
    }
    for col in DEFAULT_REPORT_COLS:
        if col in df.columns:
            vals = df[col].dropna()
            summary[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
    return df, summary


def print_ranking(df: pd.DataFrame, rank_by: str) -> None:
    """Pretty-print a ranked table sorted by `rank_by` (with correct direction)."""
    ascending = rank_by in LOWER_IS_BETTER
    sort_key = f"{rank_by}_mean" if not rank_by.endswith("_mean") else rank_by
    if sort_key not in df.columns:
        print(f"\n[warn] {sort_key} not in summary — falling back to first numeric column")
        sort_key = next((c for c in df.columns if c.endswith("_mean")), None)
        if sort_key is None:
            print(df)
            return

    ranked = df.sort_values(sort_key, ascending=ascending).reset_index(drop=True)
    print("\n" + "=" * 110)
    print(f"  CHECKPOINT RANKING (sorted by {sort_key}, "
          f"{'lower' if ascending else 'higher'} is better)")
    print("=" * 110)
    cols_to_show = ["iteration", "checkpoint", "n_evaluated"] + [
        f"{c}_mean" for c in DEFAULT_REPORT_COLS if f"{c}_mean" in ranked.columns
    ]
    header = " | ".join(f"{c[:14]:>14}" for c in cols_to_show)
    print(header)
    print("-" * len(header))
    for _, row in ranked.iterrows():
        cells = []
        for c in cols_to_show:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:>14.4f}")
            else:
                cells.append(f"{str(v):>14}")
        print(" | ".join(cells))

    print("\n" + "=" * 110)
    best = ranked.iloc[0]
    print(f"  BEST: {best['checkpoint']} ({sort_key}={best[sort_key]:.4f})")
    print("=" * 110)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep Approach_B checkpoints to pick the best one.",
    )
    parser.add_argument("--config", type=str, default="hotflow/configs/train_b.yaml")
    parser.add_argument("--ckpt_dir", type=str,
                        default="logs_b/train_b_2026_04_14__21_22_14/checkpoints",
                        help="Directory containing <iter>.pt checkpoints")
    parser.add_argument("--ckpt", type=str, nargs="+", default=None,
                        help="Explicit list of checkpoint paths (overrides --ckpt_dir)")
    parser.add_argument("--every", type=int, default=100000,
                        help="Evaluate only checkpoints whose iteration is a "
                             "multiple of this. Default 100k.")
    parser.add_argument("--min_iter", type=int, default=None,
                        help="Skip checkpoints below this iteration.")
    parser.add_argument("--max_iter", type=int, default=None,
                        help="Skip checkpoints above this iteration.")

    parser.add_argument("--num_samples", type=int, default=30,
                        help="Test-set samples per checkpoint (small for speed).")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Denoising steps (smaller is faster).")
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="results/ckpt_sweep")
    parser.add_argument("--rank_by", type=str, default="rmsd_aligned",
                        choices=DEFAULT_REPORT_COLS,
                        help="Metric to rank checkpoints by. Lower-is-better "
                             "for rmsd_*/interface_rmsd/anchor_min_dist; "
                             "higher-is-better for the rest.")

    args = parser.parse_args()

    seed_all(args.seed)
    device = torch.device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config, _ = load_config(args.config)

    ckpts = find_checkpoints(
        Path(args.ckpt_dir),
        every=args.every,
        min_iter=args.min_iter,
        max_iter=args.max_iter,
        explicit=args.ckpt,
    )
    if not ckpts:
        print(f"No checkpoints found in {args.ckpt_dir} matching filters.")
        return

    print(f"[sweep] Evaluating {len(ckpts)} checkpoint(s):")
    for c in ckpts:
        print(f"  {c}")
    print(f"[sweep] {args.num_samples} samples × {args.num_steps} denoising steps "
          f"each (guidance_scale={args.guidance_scale})")
    print(f"[sweep] Output: {outdir}")

    # Build the dataloader ONCE; iterate it fresh per checkpoint.
    loader, dataset = build_eval_dataloader(config, num_workers=args.num_workers)
    num_samples = min(args.num_samples, len(dataset))

    summaries: list[dict] = []
    for ckpt in ckpts:
        per_ckpt_dir = outdir / ckpt.stem
        per_ckpt_dir.mkdir(parents=True, exist_ok=True)

        df, summary = evaluate_checkpoint(
            ckpt, config, loader, num_samples,
            args.num_steps, args.guidance_scale, device,
        )
        df.to_csv(per_ckpt_dir / "metrics.csv", index=False)
        summaries.append(summary)

        # Save running sweep CSV after each checkpoint so partial progress
        # isn't lost on OOM / interrupt.
        pd.DataFrame(summaries).to_csv(outdir / "ckpt_sweep.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    print_ranking(summary_df, args.rank_by)
    print(f"\n[sweep] Full sweep CSV → {outdir / 'ckpt_sweep.csv'}")


if __name__ == "__main__":
    main()
