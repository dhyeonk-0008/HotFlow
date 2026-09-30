"""Target preprocessing for the peptide binder design benchmark.

Handles PDB/mmCIF parsing, chain separation, pocket extraction,
batch construction for PepFlow/PepHAR/Approach A/B, and de novo
placeholder peptide generation.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
PEPFLOW_ROOT = REPO_ROOT / "PepFlowww"
PEPHAR_ROOT = REPO_ROOT / "PepHAR"
if str(PEPFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(PEPFLOW_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pepflow.modules.protein.constants import BBHeavyAtom, AA, max_num_heavyatoms  # noqa: E402
from pepflow.modules.protein.parsers import parse_pdb, parse_biopython_structure  # noqa: E402
from pepflow.modules.protein.writers import save_pdb  # noqa: E402
from pepflow.utils.data import PaddingCollate  # noqa: E402
from models_con.torsion import get_torsion_angle  # noqa: E402

from hotflow.data_types import HotspotAnchor  # noqa: E402

# ---------------------------------------------------------------------------
# Target configurations
# ---------------------------------------------------------------------------

TARGET_CONFIGS = {
    "9CDZ": {
        "pdb_file": "test_cases/9CDZ.pdb",
        "format": "pdb",
        "receptor_chains": ["A"],
        "peptide_chains": ["B"],
        "has_gt": True,
        "gt_pep_length": 16,
    },
    "7UXO": {
        "pdb_file": "test_cases/7UXO.pdb",
        "format": "pdb",
        "receptor_chains": ["A"],
        "peptide_chains": ["B"],
        "has_gt": True,
        "gt_pep_length": 12,
    },
    "6YVR": {
        "pdb_file": "test_cases/6YVR.cif",
        "format": "mmcif",
        "receptor_chains": ["AAA"],
        "peptide_chains": ["CCC"],
        "has_gt": True,
        "gt_pep_length": 8,
    },
    "4Y5U": {
        "pdb_file": "test_cases/4Y5U.pdb",
        "format": "pdb",
        "receptor_chains": ["A"],
        "peptide_chains": [],
        "has_gt": False,
        "denovo_pep_lengths": [12, 15, 18],
        "denovo_counts": [17, 17, 16],
    },
    "8TF5": {
        "pdb_file": "test_cases/8TF5.pdb",
        "format": "pdb",
        "receptor_chains": ["A"],
        "peptide_chains": [],
        "has_gt": False,
        "denovo_pep_lengths": [8, 12],
        "denovo_counts": [25, 25],
    },
    "6LUQ": {
        "pdb_file": "test_cases/6LUQ.pdb",
        "format": "pdb",
        "receptor_chains": ["A"],
        "peptide_chains": [],
        "has_gt": False,
        "denovo_pep_lengths": [10, 12, 15],
        "denovo_counts": [17, 17, 16],
    },
}

POCKET_CUTOFF = 10.0
MAX_REC_LENGTH = 224
P2RANK_BIN = REPO_ROOT / "tools" / "p2rank_2.4.2" / "prank"


# ---------------------------------------------------------------------------
# P2Rank pocket prediction
# ---------------------------------------------------------------------------

def _find_java_home() -> str | None:
    """Find JAVA_HOME from conda or system PATH."""
    import shutil
    java_bin = shutil.which("java")
    if java_bin is None:
        return None
    # java binary is typically at $JAVA_HOME/bin/java
    java_home = str(Path(java_bin).resolve().parents[1])
    return java_home


def run_p2rank(receptor_pdb: str | Path, output_dir: str | Path) -> Path:
    """Run P2Rank on a receptor PDB and return the residues CSV path."""
    receptor_pdb = Path(receptor_pdb)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(P2RANK_BIN), "predict",
        "-f", str(receptor_pdb),
        "-o", str(output_dir),
    ]
    env = dict(os.environ)
    java_home = _find_java_home()
    if java_home:
        env["JAVA_HOME"] = java_home
        env["PATH"] = str(Path(java_home) / "bin") + ":" + env.get("PATH", "")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"P2Rank failed:\n{result.stderr}")

    # Find the residues CSV — P2Rank names it "{filename}_residues.csv"
    # where filename includes extension, e.g. "receptor_full.pdb_residues.csv"
    candidates = list(output_dir.rglob("*_residues.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"P2Rank residues CSV not found in {output_dir}")
    return candidates[0]


def parse_p2rank_pocket_residues(
    csv_path: str | Path,
    min_score: float = 0.5,
    pocket_id: int | None = 1,
) -> list[dict[str, Any]]:
    """Parse P2Rank residues CSV and return pocket residues sorted by score.

    Returns list of dicts with keys: chain, resseq, resname, score, pocket.
    """
    results = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Column names may have leading spaces
            cleaned = {k.strip(): v.strip() for k, v in row.items()}
            score = float(cleaned.get("score", "0"))
            pocket = cleaned.get("pocket", "").strip()
            pocket_num = int(pocket) if pocket and pocket != "" else None

            if score < min_score:
                continue
            if pocket_id is not None and pocket_num != pocket_id:
                continue

            chain = cleaned.get("chain", "A")
            res_label = cleaned.get("residue_label", "0")
            # Extract numeric part from residue_label (e.g., "45" or "45A")
            resseq = int("".join(c for c in res_label if c.isdigit()) or "0")

            results.append({
                "chain": chain,
                "resseq": resseq,
                "resname": cleaned.get("residue_name", "UNK"),
                "score": score,
                "pocket": pocket_num,
            })

    results.sort(key=lambda x: -x["score"])
    return results


def _extract_pocket_by_p2rank(
    receptor: dict[str, Any],
    receptor_pdb_path: str | Path,
    p2rank_outdir: str | Path,
    max_residues: int = MAX_REC_LENGTH,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Extract pocket residues using P2Rank predictions.

    Returns:
        (pocket_dict, pocket_center)
    """
    # Run P2Rank
    csv_path = run_p2rank(receptor_pdb_path, p2rank_outdir)

    # Parse results — try pocket 1 first, fall back to all pockets
    pocket_residues = parse_p2rank_pocket_residues(csv_path, min_score=0.3, pocket_id=1)
    if len(pocket_residues) < 5:
        pocket_residues = parse_p2rank_pocket_residues(csv_path, min_score=0.3, pocket_id=None)
    if len(pocket_residues) < 3:
        print(f"  [P2Rank] Warning: only {len(pocket_residues)} pocket residues found, "
              "falling back to surface heuristic")
        return _extract_pocket_by_surface(receptor, max_residues)

    # Match P2Rank residues to receptor dict by resseq
    rec_resseq = receptor["resseq"] if "resseq" in receptor else torch.arange(receptor["aa"].shape[0])
    p2rank_resseqs = {r["resseq"] for r in pocket_residues}

    pocket_mask = torch.tensor(
        [int(rs.item()) in p2rank_resseqs for rs in rec_resseq],
        dtype=torch.bool,
    )

    if pocket_mask.sum() < 3:
        print(f"  [P2Rank] Warning: matched only {pocket_mask.sum().item()} residues, "
              "falling back to surface heuristic")
        return _extract_pocket_by_surface(receptor, max_residues)

    # Pocket center
    ca_coords = receptor["pos_heavyatom"][:, BBHeavyAtom.CA]
    pocket_center = ca_coords[pocket_mask].mean(dim=0)

    # Expand pocket: include all residues within POCKET_CUTOFF of pocket center
    dist_to_center = (ca_coords - pocket_center.unsqueeze(0)).norm(dim=-1)
    expanded_mask = dist_to_center < POCKET_CUTOFF
    combined_mask = pocket_mask | expanded_mask

    # Truncate if too many
    if combined_mask.sum() > max_residues:
        _, keep_indices = torch.topk(dist_to_center, max_residues, largest=False)
        combined_mask = torch.zeros_like(combined_mask)
        combined_mask[keep_indices] = True

    pocket_residue_count = pocket_mask.sum().item()
    total_count = combined_mask.sum().item()
    print(f"  [P2Rank] Pocket: {pocket_residue_count} core + "
          f"{total_count - pocket_residue_count} expanded = {total_count} residues")

    return _filter_by_mask(receptor, combined_mask), pocket_center


# ---------------------------------------------------------------------------
# PDB/mmCIF parsing
# ---------------------------------------------------------------------------

def parse_target_structure(pdb_path: str | Path, fmt: str = "pdb") -> dict[str, Any]:
    """Parse a target structure file, returning all chains merged."""
    pdb_path = str(pdb_path)
    if fmt == "pdb":
        data, _ = parse_pdb(pdb_path)
    elif fmt == "mmcif":
        from Bio.PDB import MMCIFParser
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("target", pdb_path)
        data, _ = parse_biopython_structure(structure[0])
    else:
        raise ValueError(f"Unknown format: {fmt}")
    if data is None:
        raise ValueError(f"Failed to parse: {pdb_path}")
    return data


def extract_chains(parsed_data: dict[str, Any], chain_ids: list[str]) -> dict[str, Any]:
    """Extract specific chains from parsed structure data."""
    chain_id_list = parsed_data["chain_id"]
    mask = torch.tensor([cid in chain_ids for cid in chain_id_list], dtype=torch.bool)
    if mask.sum() == 0:
        raise ValueError(f"No residues found for chains {chain_ids}")

    result: dict[str, Any] = {}
    for key, value in parsed_data.items():
        if isinstance(value, torch.Tensor):
            result[key] = value[mask]
        elif isinstance(value, list):
            result[key] = [value[i] for i in range(len(value)) if mask[i]]
        else:
            result[key] = value

    # Renumber chain_nb starting from 0
    if "chain_nb" in result:
        unique_chains = result["chain_nb"].unique(sorted=True)
        remap = {int(old): new for new, old in enumerate(unique_chains.tolist())}
        result["chain_nb"] = torch.tensor(
            [remap[int(c)] for c in result["chain_nb"].tolist()],
            dtype=result["chain_nb"].dtype,
        )

    # Renumber res_nb sequentially
    if "res_nb" in result:
        result["res_nb"] = torch.arange(1, mask.sum().item() + 1, dtype=result["res_nb"].dtype)

    return result


# ---------------------------------------------------------------------------
# Pocket extraction
# ---------------------------------------------------------------------------

def _extract_pocket_by_peptide(
    receptor: dict[str, Any],
    peptide: dict[str, Any],
    cutoff: float = POCKET_CUTOFF,
) -> dict[str, Any]:
    """Extract receptor residues within cutoff of peptide atoms."""
    pep_atoms = peptide["pos_heavyatom"][peptide["mask_heavyatom"]]
    rec_ca = receptor["pos_heavyatom"][:, BBHeavyAtom.CA]
    dist = torch.cdist(rec_ca.unsqueeze(0), pep_atoms.unsqueeze(0)).squeeze(0)
    min_dist = dist.min(dim=1).values
    pocket_mask = min_dist < cutoff

    # Also filter non-standard residues
    standard_mask = receptor["aa"] <= 19
    pocket_mask = pocket_mask & standard_mask

    if pocket_mask.sum() < 1:
        raise ValueError("No receptor residues within pocket cutoff")

    return _filter_by_mask(receptor, pocket_mask)


def _extract_pocket_by_surface(
    receptor: dict[str, Any],
    max_residues: int = MAX_REC_LENGTH,
    surface_cutoff_rank: int = 60,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Extract pocket residues using surface exposure heuristic.

    Returns:
        (pocket_dict, pocket_center)
    """
    # Filter non-standard residues first
    standard_mask = receptor["aa"] <= 19
    receptor = _filter_by_mask(receptor, standard_mask)

    ca_coords = receptor["pos_heavyatom"][:, BBHeavyAtom.CA]
    R = ca_coords.shape[0]

    # Exposure heuristic: fewer neighbours within 10A = more exposed
    dist_mat = torch.cdist(ca_coords.unsqueeze(0), ca_coords.unsqueeze(0)).squeeze(0)
    neighbour_count = (dist_mat < 10.0).sum(dim=-1).float()

    # Select most exposed residues
    k = min(surface_cutoff_rank, R)
    _, exposed_indices = torch.topk(neighbour_count, k, largest=False)

    # Pocket center = centroid of exposed residues
    pocket_center = ca_coords[exposed_indices].mean(dim=0)

    # Select residues closest to pocket center (up to max_residues)
    dist_to_center = (ca_coords - pocket_center.unsqueeze(0)).norm(dim=-1)
    n_keep = min(max_residues, R)
    _, keep_indices = torch.topk(dist_to_center, n_keep, largest=False)
    keep_mask = torch.zeros(R, dtype=torch.bool)
    keep_mask[keep_indices] = True

    return _filter_by_mask(receptor, keep_mask), pocket_center


def _filter_by_mask(data: dict[str, Any], mask: torch.Tensor) -> dict[str, Any]:
    """Filter all fields in a parsed dict by boolean mask."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            result[key] = value[mask]
        elif isinstance(value, list):
            result[key] = [value[i] for i in range(len(value)) if mask[i]]
        else:
            result[key] = value
    return result


def _truncate_receptor(
    receptor: dict[str, Any],
    peptide: dict[str, Any],
    max_length: int = MAX_REC_LENGTH,
) -> dict[str, Any]:
    """Truncate receptor to closest residues to peptide."""
    if receptor["aa"].shape[0] <= max_length:
        return receptor

    rec_ca = receptor["pos_heavyatom"][:, BBHeavyAtom.CA]
    pep_ca = peptide["pos_heavyatom"][:, BBHeavyAtom.CA]
    dist = torch.cdist(pep_ca.unsqueeze(0), rec_ca.unsqueeze(0)).squeeze(0)
    min_dist = dist.min(dim=0).values
    _, keep = torch.topk(min_dist, max_length, largest=False)
    keep_mask = torch.zeros(receptor["aa"].shape[0], dtype=torch.bool)
    keep_mask[keep] = True
    return _filter_by_mask(receptor, keep_mask)


# ---------------------------------------------------------------------------
# Target preparation (main entry point)
# ---------------------------------------------------------------------------

def prepare_target(
    target_name: str,
    outdir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    """Prepare a target for benchmarking.

    Returns:
        (receptor_dict, peptide_dict_or_None, metadata)
    """
    cfg = TARGET_CONFIGS[target_name]
    outdir = Path(outdir) / target_name
    outdir.mkdir(parents=True, exist_ok=True)

    pdb_path = REPO_ROOT / cfg["pdb_file"]
    parsed = parse_target_structure(pdb_path, cfg["format"])

    # Extract chains
    receptor = extract_chains(parsed, cfg["receptor_chains"])
    peptide = None
    if cfg["has_gt"] and cfg["peptide_chains"]:
        peptide = extract_chains(parsed, cfg["peptide_chains"])

    # Filter non-standard residues from receptor
    std_mask = receptor["aa"] <= 19
    receptor = _filter_by_mask(receptor, std_mask)

    if peptide is not None:
        # Filter non-standard from peptide
        pep_std_mask = peptide["aa"] <= 19
        peptide = _filter_by_mask(peptide, pep_std_mask)

        # Center on peptide COM
        pep_ca_mask = peptide["mask_heavyatom"][:, BBHeavyAtom.CA]
        center = peptide["pos_heavyatom"][pep_ca_mask, BBHeavyAtom.CA].mean(dim=0)
        peptide["pos_heavyatom"] = peptide["pos_heavyatom"] - center[None, None, :]
        receptor["pos_heavyatom"] = receptor["pos_heavyatom"] - center[None, None, :]

        # Extract pocket
        receptor = _extract_pocket_by_peptide(receptor, peptide)
        receptor = _truncate_receptor(receptor, peptide)
    else:
        # De novo: save full receptor PDB first, then run P2Rank
        # Save a temporary receptor PDB for P2Rank (before centering)
        tmp_rec_pdb = outdir / "receptor_full.pdb"
        _save_chain_pdb(receptor, tmp_rec_pdb, chain_letter="A")

        p2rank_outdir = outdir / "p2rank"
        try:
            receptor, pocket_center = _extract_pocket_by_p2rank(
                receptor, tmp_rec_pdb, p2rank_outdir)
            center = pocket_center
        except Exception as e:
            print(f"  [P2Rank] Failed ({e}), falling back to surface heuristic")
            receptor, pocket_center = _extract_pocket_by_surface(receptor)
            center = pocket_center

        receptor["pos_heavyatom"] = receptor["pos_heavyatom"] - center[None, None, :]

    # Compute torsion angles
    receptor["torsion_angle"], receptor["torsion_angle_mask"] = get_torsion_angle(
        receptor["pos_heavyatom"], receptor["aa"]
    )
    if peptide is not None:
        peptide["torsion_angle"], peptide["torsion_angle_mask"] = get_torsion_angle(
            peptide["pos_heavyatom"], peptide["aa"]
        )

    # Set chain_nb: receptor = 0, peptide = 1
    receptor["chain_nb"] = torch.zeros_like(receptor["aa"])
    if peptide is not None:
        peptide["chain_nb"] = torch.ones_like(peptide["aa"])

    # Renumber res_nb
    receptor["res_nb"] = torch.arange(1, receptor["aa"].shape[0] + 1, dtype=torch.long)
    if peptide is not None:
        peptide["res_nb"] = torch.arange(1, peptide["aa"].shape[0] + 1, dtype=torch.long)

    # Save PDBs
    _save_chain_pdb(receptor, outdir / "receptor.pdb", chain_letter="A")
    if peptide is not None:
        _save_chain_pdb(peptide, outdir / "peptide.pdb", chain_letter="B")

    # Build metadata
    metadata = {
        "target_name": target_name,
        "has_gt": cfg["has_gt"],
        "outdir": str(outdir),
    }
    if cfg["has_gt"]:
        pep_len = peptide["aa"].shape[0] if peptide is not None else cfg["gt_pep_length"]
        metadata["pep_length"] = int(pep_len)
    else:
        metadata["denovo_pep_lengths"] = cfg["denovo_pep_lengths"]
        metadata["denovo_counts"] = cfg["denovo_counts"]

    pep_info = f", peptide: {peptide['aa'].shape[0]} res" if peptide is not None else ", de novo"
    print(f"[{target_name}] receptor: {receptor['aa'].shape[0]} res{pep_info}")

    return receptor, peptide, metadata


def _save_chain_pdb(
    chain_dict: dict[str, Any],
    path: str | Path,
    chain_letter: str = "A",
) -> None:
    """Save a single chain dict as PDB."""
    n = chain_dict["aa"].shape[0]
    data = {
        "chain_nb": torch.zeros(n, dtype=torch.long),
        "chain_id": [chain_letter] * n,
        "resseq": torch.arange(1, n + 1, dtype=torch.long),
        "icode": [" "] * n,
        "aa": chain_dict["aa"],
        "mask_heavyatom": chain_dict["mask_heavyatom"],
        "pos_heavyatom": chain_dict["pos_heavyatom"],
    }
    save_pdb(data, path=str(path))


# ---------------------------------------------------------------------------
# Batch construction for PepFlow / Approach B
# ---------------------------------------------------------------------------

def build_pepflow_batch_gt(
    receptor: dict[str, Any],
    peptide: dict[str, Any],
    batch_size: int = 1,
    target_name: str = "unknown",
) -> dict[str, Any]:
    """Build PepFlow batch from receptor + GT peptide."""
    item = _build_pepflow_item(receptor, peptide, example_id=target_name)
    collate = PaddingCollate(eight=False)
    return collate([deepcopy(item) for _ in range(batch_size)])


def build_pepflow_batch_denovo(
    receptor: dict[str, Any],
    pep_length: int,
    batch_size: int = 1,
    target_name: str = "unknown",
) -> dict[str, Any]:
    """Build PepFlow batch with placeholder peptide for de novo generation."""
    placeholder_pep = _create_placeholder_peptide(receptor, pep_length)
    item = _build_pepflow_item(receptor, placeholder_pep, example_id=target_name)
    collate = PaddingCollate(eight=False)
    return collate([deepcopy(item) for _ in range(batch_size)])


def _build_pepflow_item(
    receptor: dict[str, Any],
    peptide: dict[str, Any],
    example_id: str = "unknown",
) -> dict[str, Any]:
    """Concatenate receptor + peptide into a single PepFlow item."""
    combined: dict[str, Any] = {
        "id": example_id,
        "generate_mask": torch.cat(
            [torch.zeros_like(receptor["aa"]), torch.ones_like(peptide["aa"])], dim=0
        ).bool(),
    }
    for key in ["aa", "pos_heavyatom", "mask_heavyatom", "chain_nb", "res_nb",
                "resseq", "torsion_angle", "torsion_angle_mask"]:
        if key in receptor and key in peptide:
            if isinstance(receptor[key], torch.Tensor):
                combined[key] = torch.cat([receptor[key], peptide[key]], dim=0)
            elif isinstance(receptor[key], list):
                combined[key] = receptor[key] + peptide[key]

    # chain_id
    rec_n = receptor["aa"].shape[0]
    pep_n = peptide["aa"].shape[0]
    combined["chain_id"] = ["A"] * rec_n + ["B"] * pep_n
    combined["icode"] = [" "] * (rec_n + pep_n)

    # Fix resseq to be sequential
    combined["resseq"] = torch.arange(1, rec_n + pep_n + 1, dtype=torch.long)

    return combined


def _create_placeholder_peptide(
    receptor: dict[str, Any],
    pep_length: int,
) -> dict[str, Any]:
    """Create placeholder peptide residues near the receptor pocket center."""
    rec_ca = receptor["pos_heavyatom"][:, BBHeavyAtom.CA]
    pocket_center = rec_ca.mean(dim=0)

    # Placeholder positions: near pocket center with Gaussian noise
    pos = torch.zeros(pep_length, max_num_heavyatoms, 3)
    mask = torch.zeros(pep_length, max_num_heavyatoms, dtype=torch.bool)

    for i in range(pep_length):
        noise = 5.0 * torch.randn(3)
        ca_pos = pocket_center + noise
        pos[i, BBHeavyAtom.N] = ca_pos + torch.tensor([1.458, 0.0, 0.0])
        pos[i, BBHeavyAtom.CA] = ca_pos
        pos[i, BBHeavyAtom.C] = ca_pos + torch.tensor([-0.553, 1.419, 0.0])
        mask[i, BBHeavyAtom.N] = True
        mask[i, BBHeavyAtom.CA] = True
        mask[i, BBHeavyAtom.C] = True

    placeholder = {
        "aa": torch.zeros(pep_length, dtype=torch.long),  # ALA
        "pos_heavyatom": pos,
        "mask_heavyatom": mask,
        "chain_nb": torch.ones(pep_length, dtype=torch.long),
        "res_nb": torch.arange(1, pep_length + 1, dtype=torch.long),
        "resseq": torch.arange(1, pep_length + 1, dtype=torch.long),
        "chain_id": ["B"] * pep_length,
        "icode": [" "] * pep_length,
        "torsion_angle": torch.zeros(pep_length, 5),
        "torsion_angle_mask": torch.zeros(pep_length, 5, dtype=torch.bool),
    }
    return placeholder


# ---------------------------------------------------------------------------
# PepHAR data construction
# ---------------------------------------------------------------------------

def build_pephar_data_gt(
    receptor: dict[str, Any],
    peptide: dict[str, Any],
) -> dict[str, Any]:
    """Build PepHAR-format data dict from receptor + GT peptide."""
    return {
        "rec_coord": receptor["pos_heavyatom"][:, [BBHeavyAtom.CA, BBHeavyAtom.C, BBHeavyAtom.N]],
        "rec_aa": receptor["aa"],
        "pep_coord": peptide["pos_heavyatom"][:, [BBHeavyAtom.CA, BBHeavyAtom.C, BBHeavyAtom.N]],
        "pep_aa": peptide["aa"],
    }


def build_pephar_data_denovo(
    receptor: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract receptor coords/aa for PepHAR de novo sampling.

    Returns:
        (rec_coord: (R, 3, 3), rec_aa: (R,))
    """
    rec_coord = receptor["pos_heavyatom"][:, [BBHeavyAtom.CA, BBHeavyAtom.C, BBHeavyAtom.N]]
    rec_aa = receptor["aa"]
    return rec_coord, rec_aa


# ---------------------------------------------------------------------------
# De novo anchor construction (for Approach B)
# ---------------------------------------------------------------------------

def build_denovo_anchors(
    receptor: dict[str, Any],
    num_anchors: int = 5,
    pephar_sampler: Any | None = None,
    anchor_steps: int = 100,
    spread: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build peptide hotspot anchors for de novo Approach B.

    Uses PepHAR's EBM density model to predict peptide hotspot positions
    near the receptor binding pocket.  The pocket center is estimated as
    the centroid of the most surface-exposed receptor residues.

    Falls back to the exposure-based heuristic (receptor-side anchors)
    only when ``pephar_sampler`` is *None*.

    Returns:
        (anchor_coords: (1, K, 3, 3), anchor_types: (1, K), anchor_mask: (1, K))
    """
    from hotflow.data.p2rank_bridge import anchors_to_tensor

    K = num_anchors

    # Estimate pocket center from surface-exposed residues
    ca_coords = receptor["pos_heavyatom"][:, BBHeavyAtom.CA]
    R = ca_coords.shape[0]
    dist_mat = torch.cdist(ca_coords.unsqueeze(0), ca_coords.unsqueeze(0)).squeeze(0)
    neighbour_count = (dist_mat < 10.0).sum(dim=-1).float()
    surface_k = min(30, R)
    _, exposed_indices = torch.topk(neighbour_count, surface_k, largest=False)
    pocket_center = ca_coords[exposed_indices].mean(dim=0)  # (3,)

    if pephar_sampler is not None:
        # Build receptor coords in [CA, C, N] order for PepHAR
        rec_coord = torch.stack([
            receptor["pos_heavyatom"][:, BBHeavyAtom.CA],
            receptor["pos_heavyatom"][:, BBHeavyAtom.C],
            receptor["pos_heavyatom"][:, BBHeavyAtom.N],
        ], dim=1)  # (R, 3, 3)
        rec_aa = receptor["aa"].long()

        try:
            anchors = pephar_sampler.sample_denovo(
                rec_coord=rec_coord,
                rec_aa=rec_aa,
                pep_length=K * 3,  # rough length estimate for anchor spacing
                anchor_steps=anchor_steps,
                anchor_nums=K,
            )
            # sample_denovo returns (gen_dict, metrics) — extract anchor coords
            if isinstance(anchors, tuple):
                gen, _ = anchors
                pep_coord = gen["pep_coord"]  # (L, 3, 3)
                pep_aa = gen["pep_aa"]        # (L,)
                # Pick K evenly spaced residues as anchors
                L = pep_coord.shape[0]
                indices = torch.linspace(0, L - 1, K).long()
                anchor_list = []
                for idx in indices:
                    anchor_list.append(HotspotAnchor(
                        residue_index=int(idx.item()),
                        backbone_coords=pep_coord[idx].cpu(),
                        residue_type=int(pep_aa[idx].item()),
                        source="pephar_denovo",
                    ))
                return anchors_to_tensor(anchor_list, K=K, batch_size=1)
        except Exception as e:
            print(f"  [build_denovo_anchors] PepHAR fallback: {e}")

    # Fallback: receptor-side heuristic (kept for backward compatibility)
    perm = torch.randperm(surface_k)[:K]
    selected = exposed_indices[perm]

    anchor_coords = torch.zeros(1, K, 3, 3)
    anchor_types = torch.full((1, K), 20, dtype=torch.long)
    anchor_mask = torch.zeros(1, K, dtype=torch.bool)

    for i, idx in enumerate(selected):
        bb = receptor["pos_heavyatom"][idx, [BBHeavyAtom.CA, BBHeavyAtom.C, BBHeavyAtom.N]]
        anchor_coords[0, i] = bb
        anchor_types[0, i] = int(receptor["aa"][idx].item())
        anchor_mask[0, i] = True

    return anchor_coords, anchor_types, anchor_mask


# ---------------------------------------------------------------------------
# De novo peptide length selection
# ---------------------------------------------------------------------------

def get_denovo_pep_length(metadata: dict[str, Any], sample_idx: int) -> int:
    """Determine peptide length for a de novo sample based on index."""
    lengths = metadata["denovo_pep_lengths"]
    counts = metadata["denovo_counts"]

    cumulative = 0
    for length, count in zip(lengths, counts):
        if sample_idx < cumulative + count:
            return length
        cumulative += count

    # Fallback to last length
    return lengths[-1]
