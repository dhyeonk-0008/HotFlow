"""De novo capable PepHAR hotspot sampler for Approach A.

Original ``PepHARHotspotSampler.sample()`` initializes anchor positions from
GT peptide coordinates.  This revised version adds ``sample_denovo()`` which
initializes anchors from the receptor surface, enabling true de novo usage.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

from hotflow.data_types import HotspotAnchor


REPO_ROOT = Path(__file__).resolve().parents[2]
PEPHAR_ROOT = REPO_ROOT / "PepHAR"
if str(PEPHAR_ROOT) not in sys.path:
    sys.path.insert(0, str(PEPHAR_ROOT))

from evaluate.geometry import (  # noqa: E402
    construct_3d_basis,
    get_peptide_position,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from models import get_model  # noqa: E402
from utils.misc import load_config  # noqa: E402


def _recursive_to(obj: Any, device: str) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: _recursive_to(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_recursive_to(value, device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_recursive_to(value, device) for value in obj)
    return obj


def _data_to_batch(data: dict[str, Any]) -> dict[str, Any]:
    return default_collate([data])


def _coord_from(x: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    structure = get_peptide_position()
    mat = quaternion_to_rotation_matrix(o)
    ca = x + torch.matmul(mat, structure["CA1"].to(x).unsqueeze(-1)).squeeze(-1)
    c = x + torch.matmul(mat, structure["C1"].to(x).unsqueeze(-1)).squeeze(-1)
    n = x + torch.matmul(mat, structure["N1"].to(x).unsqueeze(-1)).squeeze(-1)
    return torch.stack([ca, c, n], dim=-2)


class PepHARHotspotSamplerDenovo:
    """Hotspot sampler supporting both GT-seeded and de novo anchor init.

    Extends the original ``PepHARHotspotSampler`` with a ``sample_denovo()``
    method that requires only receptor coordinates and a desired peptide
    length — no GT peptide structure needed.
    """

    def __init__(
        self,
        *,
        density_model: torch.nn.Module | None = None,
        device: str = "cpu",
        pephar_sampler: object | None = None,
    ):
        self.device = device
        self.density_model = density_model
        self.pephar_sampler = pephar_sampler

    @classmethod
    def from_checkpoint(
        cls,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "cpu",
    ) -> "PepHARHotspotSamplerDenovo":
        config, _ = load_config(str(config_path))
        density_model = get_model(config.model).to(device)
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        state_dict = (checkpoint["model"]
                      if isinstance(checkpoint, dict) and "model" in checkpoint
                      else checkpoint)
        density_model.load_state_dict(state_dict)
        density_model.eval()
        return cls(density_model=density_model, device=device)

    # ------------------------------------------------------------------
    # Original GT-seeded sample (backward compatible)
    # ------------------------------------------------------------------

    def sample(
        self,
        example: dict,
        num_anchors: int = 1,
        anchor_steps: int = 100,
        anchor_strategy: str = "ebm",
        verbose: bool = False,
    ) -> Sequence[HotspotAnchor]:
        if self.density_model is None:
            raise ValueError("No density model configured")

        pep_length = int(example["pep_aa"].shape[0])
        anchor_indices = self._get_anchor_indices(pep_length, num_anchors)
        anchors: list[HotspotAnchor] = []
        for residue_index in anchor_indices:
            seed = HotspotAnchor(
                residue_index=int(residue_index),
                backbone_coords=example["pep_coord"][residue_index].detach().cpu(),
                residue_type=int(example["pep_aa"][residue_index].item()),
                source="native_seed",
            )
            if anchor_strategy == "gt":
                anchors.append(seed)
            elif anchor_strategy == "rand":
                randomized_coords, randomized_type = self._randomize_anchor(seed)
                anchors.append(HotspotAnchor(
                    residue_index=seed.residue_index,
                    backbone_coords=randomized_coords.detach().cpu(),
                    residue_type=int(randomized_type.item()),
                    source="random_seed",
                ))
            elif anchor_strategy == "ebm":
                anchors.append(self._sample_anchor(
                    example=example, seed_anchor=seed,
                    n_steps=anchor_steps, verbose=verbose,
                ))
            else:
                raise NotImplementedError(
                    f"Unsupported anchor_strategy: {anchor_strategy}")
        return self._deduplicate_anchors(anchors)

    # ------------------------------------------------------------------
    # De novo sample — no GT peptide required
    # ------------------------------------------------------------------

    def sample_denovo(
        self,
        rec_coord: torch.Tensor,
        rec_aa: torch.Tensor,
        pep_length: int,
        num_anchors: int = 1,
        anchor_steps: int = 100,
        verbose: bool = False,
    ) -> Sequence[HotspotAnchor]:
        """Sample hotspot anchors using only receptor information.

        Anchors are initialized from exposed receptor surface residues,
        then optimized via the EBM density model.

        Args:
            rec_coord: (R, 3, 3) receptor backbone coords (CA, C, N).
            rec_aa: (R,) receptor amino acid types.
            pep_length: desired peptide length (for anchor index assignment).
            num_anchors: number of anchors to produce.
            anchor_steps: EBM optimization steps per anchor.
            verbose: print progress.

        Returns:
            List of HotspotAnchor with optimized positions.
        """
        if self.density_model is None:
            raise ValueError("No density model configured")

        anchor_indices = self._get_anchor_indices(pep_length, num_anchors)
        surface_points = self._sample_surface_points(
            rec_coord.to(self.device), n_points=len(anchor_indices))

        anchors: list[HotspotAnchor] = []
        for residue_index, (coord_init, aa_init) in zip(
                anchor_indices, surface_points):
            anchor = self._sample_anchor_denovo(
                rec_coord=rec_coord.to(self.device),
                rec_aa=rec_aa.to(self.device),
                coord_init=coord_init,
                n_steps=anchor_steps,
                verbose=verbose,
            )
            anchors.append(HotspotAnchor(
                residue_index=int(residue_index),
                backbone_coords=anchor[0].detach().cpu(),
                residue_type=int(anchor[1].item()),
                source="denovo_ebm",
            ))
        return self._deduplicate_anchors(anchors)

    # ------------------------------------------------------------------
    # Surface sampling
    # ------------------------------------------------------------------

    def _sample_surface_points(
        self, rec_coord: torch.Tensor, n_points: int = 1,
        cutoff_rank: int = 30,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Sample initial positions from exposed receptor surface residues."""
        ca_coords = rec_coord[:, 0, :]  # (R, 3)
        R = ca_coords.shape[0]

        dist_mat = torch.cdist(
            ca_coords.unsqueeze(0), ca_coords.unsqueeze(0)).squeeze(0)
        neighbour_count = (dist_mat < 10.0).sum(dim=-1).float()

        k = min(cutoff_rank, R)
        _, exposed_indices = torch.topk(neighbour_count, k, largest=False)

        perm = torch.randperm(k, device=rec_coord.device)[:n_points]
        selected_indices = exposed_indices[perm]

        results = []
        for idx in selected_indices:
            coord = rec_coord[idx].clone()
            x = coord[0] + 3.0 * torch.randn(3, device=coord.device)
            o = rotation_matrix_to_quaternion(
                construct_3d_basis(coord[0], coord[1], coord[2]))
            o = F.normalize(o + 0.5 * torch.randn_like(o), p=2, dim=-1)
            coord_noisy = _coord_from(x, o)
            aa_placeholder = torch.randint(0, 20, (1,),
                                           device=coord.device).long()[0]
            results.append((coord_noisy, aa_placeholder))
        return results

    # ------------------------------------------------------------------
    # EBM anchor optimization
    # ------------------------------------------------------------------

    def _sample_anchor(
        self, example: dict, seed_anchor: HotspotAnchor,
        n_steps: int = 100, lr: float = 3e-2, noise_eps: float = 1e-2,
        verbose: bool = False,
    ) -> HotspotAnchor:
        """Original GT-seeded EBM anchor optimization."""
        coord_init, _ = self._randomize_anchor(seed_anchor)
        return self._run_ebm_optimization(
            rec_coord=example["rec_coord"],
            rec_aa=example["rec_aa"],
            coord_init=coord_init,
            seed_residue_index=seed_anchor.residue_index,
            n_steps=n_steps, lr=lr, noise_eps=noise_eps,
            verbose=verbose,
        )

    def _sample_anchor_denovo(
        self, rec_coord: torch.Tensor, rec_aa: torch.Tensor,
        coord_init: torch.Tensor, n_steps: int = 100,
        lr: float = 3e-2, noise_eps: float = 1e-2,
        verbose: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """EBM anchor optimization with receptor-only context."""
        x_init = coord_init[0]
        o_init = rotation_matrix_to_quaternion(
            construct_3d_basis(coord_init[0], coord_init[1], coord_init[2]))
        par_x = x_init.detach().clone().to(self.device).requires_grad_(True)
        par_o = o_init.detach().clone().to(self.device).requires_grad_(True)
        par_cls = torch.zeros(20, device=self.device, requires_grad=True)
        par_T = torch.zeros(1, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([par_x, par_o, par_cls, par_T], lr=lr)

        empty_pep_coord = rec_coord[:0]
        empty_pep_aa = rec_aa[:0]

        for step in range(n_steps):
            new_coord = _coord_from(par_x, par_o)
            query = {
                "rec_coord": rec_coord,
                "rec_aa": rec_aa,
                "pep_coord": empty_pep_coord,
                "pep_aa": empty_pep_aa,
                "qry_coord": new_coord.unsqueeze(0),
                "qry_aa": torch.tensor([0], device=self.device),
            }
            logits = self.density_model(
                _recursive_to(_data_to_batch(query), self.device))[0, 0]
            prob = F.softmax(logits, dim=-1)
            obj = -torch.log(
                (prob[:20] * F.softmax(par_cls / par_T.exp(), dim=-1)).sum(-1))
            optimizer.zero_grad()
            obj.backward()
            optimizer.step()
            with torch.no_grad():
                par_x.add_(noise_eps * torch.randn_like(par_x))
                par_o.add_(noise_eps * torch.randn_like(par_o))
                par_o.copy_(F.normalize(par_o, p=2, dim=-1))
            if verbose and step % max(n_steps // 5, 1) == 0:
                print(f"[denovo] step={step} obj={obj.item():.4f} "
                      f"type={int(par_cls.argmax(-1).item())}")

        coord = _coord_from(par_x, par_o).detach()
        aa = par_cls.argmax(-1).detach()
        return coord, aa

    def _run_ebm_optimization(
        self, rec_coord, rec_aa, coord_init, seed_residue_index,
        n_steps=100, lr=3e-2, noise_eps=1e-2, verbose=False,
    ) -> HotspotAnchor:
        x_init = coord_init[0]
        o_init = rotation_matrix_to_quaternion(
            construct_3d_basis(coord_init[0], coord_init[1], coord_init[2]))
        par_x = x_init.detach().clone().to(self.device).requires_grad_(True)
        par_o = o_init.detach().clone().to(self.device).requires_grad_(True)
        par_cls = torch.zeros(20, device=self.device, requires_grad=True)
        par_T = torch.zeros(1, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([par_x, par_o, par_cls, par_T], lr=lr)

        rec_coord_dev = rec_coord.to(self.device)
        rec_aa_dev = rec_aa.to(self.device)
        empty_pep_coord = rec_coord_dev[:0]
        empty_pep_aa = rec_aa_dev[:0]

        for step in range(n_steps):
            new_coord = _coord_from(par_x, par_o)
            query = {
                "rec_coord": rec_coord_dev,
                "rec_aa": rec_aa_dev,
                "pep_coord": empty_pep_coord,
                "pep_aa": empty_pep_aa,
                "qry_coord": new_coord.unsqueeze(0),
                "qry_aa": torch.tensor([0], device=self.device),
            }
            logits = self.density_model(
                _recursive_to(_data_to_batch(query), self.device))[0, 0]
            prob = F.softmax(logits, dim=-1)
            obj = -torch.log(
                (prob[:20] * F.softmax(par_cls / par_T.exp(), dim=-1)).sum(-1))
            optimizer.zero_grad()
            obj.backward()
            optimizer.step()
            with torch.no_grad():
                par_x.add_(noise_eps * torch.randn_like(par_x))
                par_o.add_(noise_eps * torch.randn_like(par_o))
                par_o.copy_(F.normalize(par_o, p=2, dim=-1))
            if verbose and step % max(n_steps // 5, 1) == 0:
                print(f"[EBM] step={step} obj={obj.item():.4f} "
                      f"type={int(par_cls.argmax(-1).item())}")

        return HotspotAnchor(
            residue_index=seed_residue_index,
            backbone_coords=_coord_from(par_x, par_o).detach().cpu(),
            residue_type=int(par_cls.argmax(-1).detach().cpu().item()),
            score=float(F.softmax(par_cls.detach().cpu(), dim=-1).max().item()),
            source="pephar_density",
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _get_anchor_indices(pep_length: int, num_anchors: int) -> list[int]:
        if pep_length <= 0 or num_anchors <= 0:
            return []
        effective = min(int(num_anchors), int(pep_length))
        candidates = [
            ((i + 1) * pep_length) // (effective + 1)
            for i in range(effective)
        ]
        unique = PepHARHotspotSamplerDenovo._deduplicate_anchor_indices(
            candidates, pep_length)
        if len(unique) == effective:
            return unique
        for idx in range(pep_length):
            if idx not in unique:
                unique.append(idx)
                if len(unique) == effective:
                    break
        return unique

    @staticmethod
    def _deduplicate_anchor_indices(
        anchor_indices: Sequence[int], pep_length: int,
    ) -> list[int]:
        deduped: list[int] = []
        seen: set[int] = set()
        for idx in anchor_indices:
            i = int(idx)
            if 0 <= i < pep_length and i not in seen:
                seen.add(i)
                deduped.append(i)
        return deduped

    @staticmethod
    def _deduplicate_anchors(
        anchors: Sequence[HotspotAnchor],
    ) -> list[HotspotAnchor]:
        deduped: list[HotspotAnchor] = []
        seen: set[int] = set()
        for a in anchors:
            if a.residue_index not in seen:
                seen.add(a.residue_index)
                deduped.append(a)
        return deduped

    def _randomize_anchor(
        self, anchor: HotspotAnchor,
        x_sigma: float = 2.0, o_sigma: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coord = anchor.backbone_coords.clone()
        x_init = coord[0] + x_sigma * torch.randn_like(coord[0])
        rot = construct_3d_basis(coord[0], coord[1], coord[2])
        quat = rotation_matrix_to_quaternion(rot)
        quat = F.normalize(quat + o_sigma * torch.randn_like(quat), p=2, dim=-1)
        randomized_coords = _coord_from(x_init, quat)
        randomized_type = torch.randint(0, 20, (1,), dtype=torch.long)[0]
        return randomized_coords, randomized_type
