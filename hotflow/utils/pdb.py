from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
PEPFLOW_ROOT = REPO_ROOT / "PepFlowww"
if str(PEPFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(PEPFLOW_ROOT))

from pepflow.modules.protein.writers import save_pdb  # noqa: E402
from models_con.torsion import full_atom_reconstruction, get_heavyatom_mask  # noqa: E402


def _extract_single_item(batch: dict[str, Any], batch_index: int = 0) -> dict[str, Any]:
    item: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            item[key] = value[batch_index].detach().cpu()
        elif isinstance(value, list):
            if not value:
                item[key] = value
            elif isinstance(value[0], tuple):
                item[key] = [entry[batch_index] for entry in value]
            elif len(value) > batch_index and not isinstance(value[0], str):
                item[key] = value[batch_index]
            else:
                item[key] = value
        else:
            item[key] = value
    return item


def _select_item_fields(item: dict[str, Any], residue_mask: torch.Tensor | None) -> dict[str, Any]:
    if residue_mask is None:
        return item

    residue_mask = residue_mask.bool().cpu()
    selected: dict[str, Any] = {}
    residue_count = int(item["aa"].shape[0])
    for key, value in item.items():
        if torch.is_tensor(value) and value.shape[:1] == (residue_count,):
            selected[key] = value[residue_mask]
        elif torch.is_tensor(value) and value.shape[0] == residue_count:
            selected[key] = value[residue_mask]
        elif isinstance(value, list) and len(value) == residue_count:
            selected[key] = [entry for entry, keep in zip(value, residue_mask.tolist()) if keep]
        else:
            selected[key] = value
    return selected


def save_batch_pdb(
    batch: dict[str, Any],
    path: str | Path,
    batch_index: int = 0,
    residue_mask: torch.Tensor | None = None,
) -> None:
    item = _extract_single_item(batch, batch_index=batch_index)
    item = _select_item_fields(item, residue_mask=residue_mask)
    data = {
        "chain_nb": item["chain_nb"],
        "chain_id": item["chain_id"],
        "resseq": item["resseq"],
        "icode": item["icode"],
        "aa": item["aa"],
        "mask_heavyatom": item["mask_heavyatom"],
        "pos_heavyatom": item["pos_heavyatom"],
    }
    save_pdb(data, path=str(path))


def save_batch_context_pdb(
    batch: dict[str, Any],
    path: str | Path,
    batch_index: int = 0,
) -> None:
    generate_mask_key = "backbone_generate_mask" if "backbone_generate_mask" in batch else "generate_mask"
    if generate_mask_key not in batch:
        raise KeyError("batch must contain generate_mask or backbone_generate_mask")
    context_mask = ~batch[generate_mask_key][batch_index].bool().cpu()
    save_batch_pdb(batch, path=path, batch_index=batch_index, residue_mask=context_mask)


def save_sample_final_pdb(
    batch: dict[str, Any],
    final_state: dict[str, torch.Tensor],
    path: str | Path,
    batch_index: int = 0,
) -> None:
    item = _extract_single_item(batch, batch_index=batch_index)
    rotmats = final_state["rotmats"][batch_index : batch_index + 1]
    trans = final_state["trans"][batch_index : batch_index + 1]
    angles = final_state["angles"][batch_index : batch_index + 1]
    seqs = final_state["seqs"][batch_index : batch_index + 1]
    pos14, _, _ = full_atom_reconstruction(R_bb=rotmats, t_bb=trans, angles=angles, aa=seqs)
    pos15 = F.pad(pos14, pad=(0, 0, 0, 15 - 14), value=0.0)[0].detach().cpu()
    mask15 = get_heavyatom_mask(seqs)[0].detach().cpu()
    backbone_generate_mask = item.get("backbone_generate_mask", item["generate_mask"]).bool()
    sequence_generate_mask = item.get("sequence_generate_mask", item["generate_mask"]).bool()
    torsion_generate_mask = item.get("torsion_generate_mask", item["generate_mask"]).bool()
    reconstruction_mask = backbone_generate_mask | sequence_generate_mask | torsion_generate_mask

    pos_new = torch.where(reconstruction_mask[:, None, None], pos15, item["pos_heavyatom"])
    mask_new = torch.where(reconstruction_mask[:, None], mask15, item["mask_heavyatom"])
    aa_new = torch.where(reconstruction_mask, seqs[0].detach().cpu(), item["aa"])
    data = {
        "chain_nb": item["chain_nb"],
        "chain_id": item["chain_id"],
        "resseq": item["resseq"],
        "icode": item["icode"],
        "aa": aa_new,
        "mask_heavyatom": mask_new,
        "pos_heavyatom": pos_new,
    }
    save_pdb(data, path=str(path))
