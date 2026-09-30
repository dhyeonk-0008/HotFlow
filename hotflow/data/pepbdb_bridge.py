from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PEPFLOW_ROOT = REPO_ROOT / "PepFlowww"
if str(PEPFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(PEPFLOW_ROOT))

from pepflow.modules.protein.constants import BBHeavyAtom  # noqa: E402
from pepflow.modules.protein.parsers import parse_pdb  # noqa: E402
from pepflow.utils.data import PaddingCollate  # noqa: E402
from models_con.torsion import get_torsion_angle  # noqa: E402

from hotflow.data_types import HotspotAnchor


@dataclass(frozen=True)
class ResidueContactSummary:
    peptide_index: int
    residue_name: str
    chain_id: str
    resseq: int
    min_heavy_dist: float
    contact_count: int


def _three_letter(index: int) -> str:
    from pepflow.modules.protein.constants import AA

    return AA(index).name


def _pick_contact_atom(pos_heavyatom: torch.Tensor, mask_heavyatom: torch.Tensor) -> torch.Tensor:
    cb_mask = mask_heavyatom[:, BBHeavyAtom.CB]
    ca_pos = pos_heavyatom[:, BBHeavyAtom.CA]
    cb_pos = pos_heavyatom[:, BBHeavyAtom.CB]
    return torch.where(cb_mask.unsqueeze(-1), cb_pos, ca_pos)


def _truncate_receptor(
    receptor: dict[str, Any],
    peptide: dict[str, Any],
    rec_length: int,
) -> dict[str, Any]:
    if receptor["aa"].shape[0] <= rec_length:
        return receptor

    rec_points = _pick_contact_atom(receptor["pos_heavyatom"], receptor["mask_heavyatom"])
    pep_points = _pick_contact_atom(peptide["pos_heavyatom"], peptide["mask_heavyatom"])
    dist = torch.cdist(pep_points, rec_points).min(dim=0).values
    keep = dist.argsort()[:rec_length]

    truncated: dict[str, Any] = {}
    for key, value in receptor.items():
        if isinstance(value, torch.Tensor):
            truncated[key] = value[keep]
        elif isinstance(value, list):
            truncated[key] = [value[i] for i in keep.tolist()]
        else:
            truncated[key] = value
    return truncated


def load_pepbdb_complex(
    example_dir: str | Path,
    rec_length: int = 224,
) -> tuple[dict[str, Any], dict[str, Any]]:
    example_dir = Path(example_dir)
    receptor, _ = parse_pdb(str(example_dir / "receptor.pdb"))
    peptide, _ = parse_pdb(str(example_dir / "peptide.pdb"))
    if receptor is None or peptide is None:
        raise ValueError(f"Failed to parse PepBDB example: {example_dir}")

    center = torch.sum(
        peptide["pos_heavyatom"][peptide["mask_heavyatom"][:, BBHeavyAtom.CA], BBHeavyAtom.CA],
        dim=0,
    ) / (
        torch.sum(peptide["mask_heavyatom"][:, BBHeavyAtom.CA]) + 1e-8
    )
    peptide["pos_heavyatom"] = peptide["pos_heavyatom"] - center[None, None, :]
    receptor["pos_heavyatom"] = receptor["pos_heavyatom"] - center[None, None, :]

    peptide["torsion_angle"], peptide["torsion_angle_mask"] = get_torsion_angle(
        peptide["pos_heavyatom"], peptide["aa"]
    )
    receptor["torsion_angle"], receptor["torsion_angle_mask"] = get_torsion_angle(
        receptor["pos_heavyatom"], receptor["aa"]
    )

    receptor = _truncate_receptor(receptor, peptide, rec_length=rec_length)
    receptor["chain_nb"] = receptor["chain_nb"] + 1
    return receptor, peptide


def build_pephar_example(
    receptor: dict[str, Any],
    peptide: dict[str, Any],
) -> dict[str, Any]:
    return {
        "rec_coord": receptor["pos_heavyatom"][:, [BBHeavyAtom.CA, BBHeavyAtom.C, BBHeavyAtom.N]],
        "pep_coord": peptide["pos_heavyatom"][:, [BBHeavyAtom.CA, BBHeavyAtom.C, BBHeavyAtom.N]],
        "rec_aa": receptor["aa"],
        "pep_aa": peptide["aa"],
        "rec_pos_heavyatom": receptor["pos_heavyatom"],
        "pep_pos_heavyatom": peptide["pos_heavyatom"],
        "rec_mask_heavyatom": receptor["mask_heavyatom"],
        "pep_mask_heavyatom": peptide["mask_heavyatom"],
    }


def build_pepflow_item(
    receptor: dict[str, Any],
    peptide: dict[str, Any],
    example_id: str,
) -> dict[str, Any]:
    combined: dict[str, Any] = {
        "id": example_id,
        "generate_mask": torch.cat(
            [torch.zeros_like(receptor["aa"]), torch.ones_like(peptide["aa"])], dim=0
        ).bool(),
    }
    for key in receptor.keys():
        value = receptor[key]
        if isinstance(value, torch.Tensor):
            combined[key] = torch.cat([receptor[key], peptide[key]], dim=0)
        elif isinstance(value, list):
            combined[key] = receptor[key] + peptide[key]
        else:
            raise TypeError(f"Unsupported field type for key {key!r}: {type(value)!r}")
    return combined


def build_pepflow_batch(
    example_dir: str | Path,
    batch_size: int = 1,
    rec_length: int = 224,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    example_dir = Path(example_dir)
    receptor, peptide = load_pepbdb_complex(example_dir, rec_length=rec_length)
    pephar_example = build_pephar_example(receptor, peptide)
    pepflow_item = build_pepflow_item(receptor, peptide, example_id=example_dir.name)
    collate = PaddingCollate(eight=False)
    pepflow_batch = collate([pepflow_item for _ in range(batch_size)])
    return receptor, peptide, pephar_example, pepflow_batch


def compute_native_contact_hotspots(
    receptor: dict[str, Any],
    peptide: dict[str, Any],
    top_k: int = 3,
    contact_cutoff: float = 4.5,
) -> tuple[list[ResidueContactSummary], list[HotspotAnchor]]:
    receptor_atoms = receptor["pos_heavyatom"][receptor["mask_heavyatom"]]
    ranked_rows: list[ResidueContactSummary] = []

    for idx in range(int(peptide["aa"].shape[0])):
        peptide_atoms = peptide["pos_heavyatom"][idx][peptide["mask_heavyatom"][idx]]
        dist = torch.cdist(peptide_atoms, receptor_atoms)
        ranked_rows.append(
            ResidueContactSummary(
                peptide_index=idx,
                residue_name=_three_letter(int(peptide["aa"][idx].item())),
                chain_id=peptide["chain_id"][idx],
                resseq=int(peptide["resseq"][idx].item()),
                min_heavy_dist=float(dist.min().item()),
                contact_count=int((dist <= contact_cutoff).sum().item()),
            )
        )

    ranked_rows = sorted(
        ranked_rows,
        key=lambda row: (-row.contact_count, row.min_heavy_dist, row.peptide_index),
    )
    selected_rows = [row for row in ranked_rows if row.contact_count > 0][:top_k]
    anchors = [
        HotspotAnchor(
            residue_index=row.peptide_index,
            backbone_coords=peptide["pos_heavyatom"][row.peptide_index][
                [BBHeavyAtom.CA, BBHeavyAtom.C, BBHeavyAtom.N]
            ].detach().cpu(),
            residue_type=int(peptide["aa"][row.peptide_index].item()),
            score=float(row.contact_count),
            source="native_contact",
        )
        for row in selected_rows
    ]
    return ranked_rows, anchors
