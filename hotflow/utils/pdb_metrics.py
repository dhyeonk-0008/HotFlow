"""PDB-based metric computation for post-Rosetta-relax evaluation.

The canonical HotFlow evaluation runs FastRelax on each generated complex
before measuring any metric. This module reads the relaxed PDB straight back
into atom14 tensors and computes every structural / sequence / interface /
clash metric from those tensors, replacing the model-specific tensor metrics
that ran on the raw (pre-relax) generation in benchmark.py / eval_compare.py.

Conventions match benchmark.py: receptor is chain "A", peptide is chain "B"
(see hotflow/benchmark_targets.py:_build_pepflow_item and
benchmark.py:save_pephar_pdb). atom14 ordering follows PepFlow's
`data.residue_constants.restype_name_to_atom14_names` — index 0=N, 1=CA,
2=C, 3=O, then per-residue side-chain atoms.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PEPFLOW_ROOT = _REPO_ROOT / "PepFlowww"
if str(_PEPFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(_PEPFLOW_ROOT))

from data import residue_constants  # noqa: E402

ATOM14_N, ATOM14_CA, ATOM14_C, ATOM14_O = 0, 1, 2, 3

# 3-letter residue name -> integer index in restypes_with_x (0..19; 20 = 'X').
_RESTYPE_3TO_IDX = {
    three: residue_constants.restypes_with_x.index(one)
    for one, three in residue_constants.restype_1to3.items()
}


def parse_complex_pdb(pdb_path: str, peptide_chain: str = "B") -> dict:
    """Parse a peptide-receptor complex PDB into atom14 numpy tensors.

    Hydrogens and non-standard residues are skipped. Atom slots not present
    in the input remain zero with mask=False (e.g. GLY's CB slot, or atoms
    Rosetta failed to model).

    Args:
        pdb_path: path to a complex PDB (relaxed or raw).
        peptide_chain: chain ID containing the peptide. Everything else is
            treated as receptor.

    Returns:
        Dict with:
          pep_aa   (Lp,)     int64 — peptide residue indices in [0, 20]
          pep_seq  str       — peptide 1-letter sequence
          pep_pos  (Lp,14,3) float32 — peptide heavy atoms in atom14 order
          pep_mask (Lp,14)   bool
          pep_ca   (Lp,3)    float32 — alias for pep_pos[:, CA]
          rec_aa, rec_pos, rec_mask, rec_ca — analogous receptor tensors
          peptide_chain str

    Raises:
        ValueError if the peptide chain has no standard residues.
    """
    from Bio.PDB import PDBParser, is_aa

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", str(pdb_path))
    model = next(iter(structure))  # first model only

    pep_records: list[tuple[int, np.ndarray, np.ndarray]] = []
    rec_records: list[tuple[int, np.ndarray, np.ndarray]] = []

    for chain in model:
        for res in chain:
            if not is_aa(res, standard=True):
                continue
            res3 = res.get_resname()
            atom14_names = residue_constants.restype_name_to_atom14_names.get(res3)
            if atom14_names is None:
                continue

            pos = np.zeros((14, 3), dtype=np.float32)
            mask = np.zeros(14, dtype=bool)
            for atom in res:
                aname = atom.get_name().strip()
                if aname in atom14_names:
                    idx = atom14_names.index(aname)
                    pos[idx] = atom.get_coord()
                    mask[idx] = True

            aa_idx = _RESTYPE_3TO_IDX.get(res3, 20)
            record = (aa_idx, pos, mask)
            if chain.id == peptide_chain:
                pep_records.append(record)
            else:
                rec_records.append(record)

    if not pep_records:
        raise ValueError(
            f"No standard peptide residues found in chain '{peptide_chain}' of {pdb_path}"
        )

    def _stack(records):
        aa = np.array([r[0] for r in records], dtype=np.int64)
        pos = np.stack([r[1] for r in records]) if records else np.zeros((0, 14, 3), dtype=np.float32)
        mask = np.stack([r[2] for r in records]) if records else np.zeros((0, 14), dtype=bool)
        return aa, pos, mask

    pep_aa, pep_pos, pep_mask = _stack(pep_records)
    rec_aa, rec_pos, rec_mask = _stack(rec_records)

    pep_seq = "".join(
        residue_constants.restypes_with_x[min(int(a), 20)] for a in pep_aa
    )

    return {
        "pep_aa": pep_aa,
        "pep_seq": pep_seq,
        "pep_pos": pep_pos,
        "pep_mask": pep_mask,
        "pep_ca": pep_pos[:, ATOM14_CA],
        "rec_aa": rec_aa,
        "rec_pos": rec_pos,
        "rec_mask": rec_mask,
        "rec_ca": (rec_pos[:, ATOM14_CA] if len(rec_pos) > 0
                   else np.zeros((0, 3), dtype=np.float32)),
        "peptide_chain": peptide_chain,
    }


def compute_metrics_from_pdb(
    gen_pdb: str,
    gt_pdb: Optional[str],
    peptide_chain: str = "B",
) -> dict:
    """Compute the canonical benchmark metrics from a (relaxed) generated PDB.

    Mirrors the dict shape returned by compute_gt_metrics /
    compute_denovo_metrics_flow in benchmark.py so the result can be merged
    in-place. GT-dependent metrics are NaN when `gt_pdb` is None or missing.

    Args:
        gen_pdb: path to the relaxed generated complex PDB.
        gt_pdb: optional path to ground-truth complex PDB.
        peptide_chain: peptide chain ID (default "B" per save_pephar_pdb and
            _build_pepflow_item).
    """
    # Imports kept inside the function so this module is importable on
    # machines without torch / pyrosetta.
    from hotflow.benchmark import (
        check_validity, check_steric_clash, check_internal_clash,
    )
    from hotflow.eval_b import (
        compute_ca_rmsd, compute_tm_score,
        compute_interface_rmsd, compute_binding_site_overlap,
    )

    gen = parse_complex_pdb(gen_pdb, peptide_chain=peptide_chain)
    pred_ca = gen["pep_ca"]
    pep_pos = gen["pep_pos"]
    pep_mask = gen["pep_mask"]
    rec_pos = gen["rec_pos"]
    rec_mask = gen["rec_mask"]

    validity = check_validity(pred_ca)
    try:
        clash = check_steric_clash(pep_pos, pep_mask, rec_pos, rec_mask)
        internal = check_internal_clash(pep_pos, pep_mask)
    except Exception:
        clash = {"num_clashes": -1, "worst_clash_dist": float("nan"),
                 "clash_free": False}
        internal = {"internal_clashes": -1, "internal_worst_dist": float("nan"),
                    "internal_clash_free": False}

    out: dict = {
        **validity,
        **clash,
        **internal,
        "pep_len": int(len(pred_ca)),
        "pred_seq": gen["pep_seq"],
    }

    if gt_pdb is None or not Path(gt_pdb).exists():
        out.update({
            "rmsd_unaligned": float("nan"),
            "rmsd_aligned": float("nan"),
            "tm_score": float("nan"),
            "aar": float("nan"),
            "interface_rmsd": float("nan"),
            "binding_site_overlap": float("nan"),
            "gt_seq": "",
        })
        return out

    gt = parse_complex_pdb(gt_pdb, peptide_chain=peptide_chain)
    gt_ca = gt["pep_ca"]
    rec_ca_gt = gt["rec_ca"]

    matched_len = (len(pred_ca) == len(gt_ca))
    if matched_len and len(pred_ca) >= 3:
        rmsd_unaligned, rmsd_aligned = compute_ca_rmsd(pred_ca, gt_ca)
        try:
            tm_score = compute_tm_score(pred_ca, gt_ca)
        except Exception:
            tm_score = float("nan")
        irmsd = compute_interface_rmsd(pred_ca, gt_ca, rec_ca_gt,
                                        interface_cutoff=8.0)
        bs_overlap = compute_binding_site_overlap(pred_ca, gt_ca, rec_ca_gt,
                                                   contact_cutoff=10.0)
    else:
        rmsd_unaligned = rmsd_aligned = tm_score = float("nan")
        irmsd = bs_overlap = float("nan")

    if matched_len and len(pred_ca) > 0:
        aar = float((gen["pep_aa"] == gt["pep_aa"]).mean())
    else:
        aar = float("nan")

    out.update({
        "rmsd_unaligned": rmsd_unaligned,
        "rmsd_aligned": rmsd_aligned,
        "tm_score": tm_score,
        "aar": aar,
        "interface_rmsd": irmsd,
        "binding_site_overlap": bs_overlap,
        "gt_seq": gt["pep_seq"],
    })
    return out
