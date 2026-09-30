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


class PepHARHotspotSampler:
    """Checkpoint-backed hotspot sampler for PepHAR density models.

    For benchmark settings we currently seed the optimizer from the native peptide
    positions, following the original PepHAR sampling code.
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
    ) -> "PepHARHotspotSampler":
        config, _ = load_config(str(config_path))
        density_model = get_model(config.model).to(device)
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        density_model.load_state_dict(state_dict)
        density_model.eval()
        return cls(density_model=density_model, device=device)

    def sample_de_novo(
        self,
        rec_coord: torch.Tensor,
        rec_aa: torch.Tensor,
        pocket_center: torch.Tensor,
        num_anchors: int = 5,
        anchor_steps: int = 100,
        spread: float = 5.0,
        verbose: bool = False,
    ) -> Sequence[HotspotAnchor]:
        """Sample peptide hotspot anchors de novo from a pocket region.

        Uses PepHAR's EBM to optimize peptide residue positions starting
        from random points near the pocket center.  No native peptide
        coordinates are required.

        Args:
            rec_coord: (R, 3, 3) receptor backbone coords [CA, C, N].
            rec_aa: (R,) receptor residue types.
            pocket_center: (3,) xyz center of the binding pocket
                (e.g. from P2Rank pocket prediction).
            num_anchors: number of hotspot anchors to produce.
            anchor_steps: Langevin optimization steps per anchor.
            spread: std-dev (Angstroms) of the Gaussian used to scatter
                initial positions around ``pocket_center``.
            verbose: print optimization progress.

        Returns:
            List of HotspotAnchor objects with optimized peptide
            backbone coordinates and predicted residue types.
        """
        if self.density_model is None:
            raise ValueError("density_model is required for de novo sampling")

        anchors: list[HotspotAnchor] = []
        for i in range(num_anchors):
            # Initialise near the pocket center with Gaussian spread
            init_x = pocket_center.clone() + spread * torch.randn(3)
            init_o = F.normalize(torch.randn(4), p=2, dim=-1)
            anchor = self._sample_anchor_from_position(
                rec_coord=rec_coord,
                rec_aa=rec_aa,
                init_x=init_x,
                init_o=init_o,
                residue_index=i,
                n_steps=anchor_steps,
                verbose=verbose,
            )
            anchors.append(anchor)
        return self._deduplicate_anchors(anchors)

    def _sample_anchor_from_position(
        self,
        rec_coord: torch.Tensor,
        rec_aa: torch.Tensor,
        init_x: torch.Tensor,
        init_o: torch.Tensor,
        residue_index: int = 0,
        n_steps: int = 100,
        lr: float = 3e-2,
        noise_eps: float = 1e-2,
        verbose: bool = False,
    ) -> HotspotAnchor:
        """Run EBM Langevin optimisation from an arbitrary starting position.

        Same optimisation loop as ``_sample_anchor`` but seeded from
        explicit ``init_x`` / ``init_o`` instead of a native peptide
        coordinate, so it can be used for de-novo sampling.
        """
        par_x = init_x.detach().clone().to(self.device).requires_grad_(True)
        par_o = init_o.detach().clone().to(self.device).requires_grad_(True)
        par_cls = torch.zeros(20, device=self.device, requires_grad=True)
        par_T = torch.zeros(1, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([par_x, par_o, par_cls, par_T], lr=lr)

        rc = rec_coord.to(self.device)
        ra = rec_aa.to(self.device)
        empty_coord = rc[:0]  # (0, 3, 3)
        empty_aa = ra[:0]     # (0,)

        for step in range(n_steps):
            new_coord = _coord_from(par_x, par_o)
            query = {
                "rec_coord": rc,
                "rec_aa": ra,
                "pep_coord": empty_coord,
                "pep_aa": empty_aa,
                "qry_coord": new_coord.unsqueeze(0),
                "qry_aa": torch.tensor([0], device=self.device),
            }
            logits = self.density_model(
                _recursive_to(_data_to_batch(query), self.device)
            )[0, 0]
            prob = F.softmax(logits, dim=-1)
            obj = -torch.log(
                (prob[:20] * F.softmax(par_cls / par_T.exp(), dim=-1)).sum(-1)
            )
            optimizer.zero_grad()
            obj.backward()
            optimizer.step()
            with torch.no_grad():
                par_x.add_(noise_eps * torch.randn_like(par_x))
                par_o.add_(noise_eps * torch.randn_like(par_o))
                par_o.copy_(F.normalize(par_o, p=2, dim=-1))
            if verbose and step % max(n_steps // 5, 1) == 0:
                print(
                    f"[PepHARHotspotSampler] step={step} obj={obj.item():.4f} "
                    f"type={int(par_cls.argmax(-1).item())}"
                )

        return HotspotAnchor(
            residue_index=residue_index,
            backbone_coords=_coord_from(par_x, par_o).detach().cpu(),
            residue_type=int(par_cls.argmax(-1).detach().cpu().item()),
            score=float(F.softmax(par_cls.detach().cpu(), dim=-1).max().item()),
            source="pephar_density",
        )

    def sample(
        self,
        example: dict,
        num_anchors: int = 1,
        anchor_steps: int = 100,
        anchor_strategy: str = "ebm",
        verbose: bool = False,
    ) -> Sequence[HotspotAnchor]:
        if self.pephar_sampler is not None:
            return self._sample_with_private_sampler(
                example=example,
                num_anchors=num_anchors,
                anchor_steps=anchor_steps,
                anchor_strategy=anchor_strategy,
                verbose=verbose,
            )
        if self.density_model is None:
            raise ValueError("No density model or PepHAR sampler is configured")

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
                anchors.append(
                    HotspotAnchor(
                        residue_index=seed.residue_index,
                        backbone_coords=randomized_coords.detach().cpu(),
                        residue_type=int(randomized_type.item()),
                        source="random_seed",
                    )
                )
            elif anchor_strategy == "ebm":
                anchors.append(
                    self._sample_anchor(
                        example=example,
                        seed_anchor=seed,
                        n_steps=anchor_steps,
                        verbose=verbose,
                    )
                )
            else:
                raise NotImplementedError(f"Unsupported anchor_strategy: {anchor_strategy}")
        return self._deduplicate_anchors(anchors)

    def _sample_with_private_sampler(
        self,
        example: dict,
        num_anchors: int,
        anchor_steps: int,
        anchor_strategy: str,
        verbose: bool,
    ) -> Sequence[HotspotAnchor]:
        if not hasattr(self.pephar_sampler, "_get_anchor_index"):
            raise TypeError("pephar_sampler must expose _get_anchor_index()")
        if not hasattr(self.pephar_sampler, "reverse_engineer_dist_list"):
            raise TypeError("pephar_sampler must expose reverse_engineer_dist_list()")
        if not hasattr(self.pephar_sampler, "_get_anchors"):
            raise TypeError("pephar_sampler must expose _get_anchors()")

        pep_length = int(example["pep_aa"].shape[0])
        anchor_indices = self._deduplicate_anchor_indices(
            self.pephar_sampler._get_anchor_index(pep_length, num_anchors),
            pep_length,
        )
        dist_list = self.pephar_sampler.reverse_engineer_dist_list(anchor_indices, pep_length)
        frag_list = self.pephar_sampler._get_anchors(
            example,
            dist_list,
            anchor_steps=anchor_steps,
            anchor_strategy=anchor_strategy,
            verbose=verbose,
        )
        return self._deduplicate_anchors(
            [
            HotspotAnchor(
                residue_index=int(residue_index),
                backbone_coords=coords.squeeze(0).detach().cpu(),
                residue_type=int(aa.squeeze(0).item()),
                source="pephar_density",
            )
            for residue_index, (coords, aa) in zip(anchor_indices, frag_list)
            ]
        )

    @staticmethod
    def _get_anchor_indices(pep_length: int, num_anchors: int) -> list[int]:
        if pep_length <= 0 or num_anchors <= 0:
            return []

        effective_num_anchors = min(int(num_anchors), int(pep_length))
        candidate_indices = [
            ((i + 1) * pep_length) // (effective_num_anchors + 1)
            for i in range(effective_num_anchors)
        ]
        unique_indices = PepHARHotspotSampler._deduplicate_anchor_indices(candidate_indices, pep_length)
        if len(unique_indices) == effective_num_anchors:
            return unique_indices

        for residue_index in range(pep_length):
            if residue_index in unique_indices:
                continue
            unique_indices.append(residue_index)
            if len(unique_indices) == effective_num_anchors:
                break
        return unique_indices

    @staticmethod
    def _deduplicate_anchor_indices(anchor_indices: Sequence[int], pep_length: int) -> list[int]:
        deduplicated: list[int] = []
        seen: set[int] = set()
        for residue_index in anchor_indices:
            idx = int(residue_index)
            if idx < 0 or idx >= pep_length or idx in seen:
                continue
            seen.add(idx)
            deduplicated.append(idx)
        return deduplicated

    @staticmethod
    def _deduplicate_anchors(anchors: Sequence[HotspotAnchor]) -> list[HotspotAnchor]:
        deduplicated: list[HotspotAnchor] = []
        seen: set[int] = set()
        for anchor in anchors:
            if anchor.residue_index in seen:
                continue
            seen.add(anchor.residue_index)
            deduplicated.append(anchor)
        return deduplicated

    def _randomize_anchor(
        self,
        anchor: HotspotAnchor,
        x_sigma: float = 2.0,
        o_sigma: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coord = anchor.backbone_coords.clone()
        x_init = coord[0] + x_sigma * torch.randn_like(coord[0])
        rot = construct_3d_basis(coord[0], coord[1], coord[2])
        quat = rotation_matrix_to_quaternion(rot)
        quat = F.normalize(quat + o_sigma * torch.randn_like(quat), p=2, dim=-1)
        randomized_coords = _coord_from(x_init, quat)
        randomized_type = torch.randint(0, 20, (1,), dtype=torch.long)[0]
        return randomized_coords, randomized_type

    def _sample_anchor(
        self,
        example: dict,
        seed_anchor: HotspotAnchor,
        n_steps: int = 100,
        lr: float = 3e-2,
        noise_eps: float = 1e-2,
        verbose: bool = False,
    ) -> HotspotAnchor:
        coord_init, _ = self._randomize_anchor(seed_anchor)
        x_init = coord_init[0]
        o_init = rotation_matrix_to_quaternion(
            construct_3d_basis(coord_init[0], coord_init[1], coord_init[2])
        )
        par_x = x_init.detach().clone().to(self.device).requires_grad_(True)
        par_o = o_init.detach().clone().to(self.device).requires_grad_(True)
        par_cls = torch.zeros(20, device=self.device, requires_grad=True)
        par_T = torch.zeros(1, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([par_x, par_o, par_cls, par_T], lr=lr)

        rec_coord = example["rec_coord"].to(self.device)
        rec_aa = example["rec_aa"].to(self.device)
        pep_coord = example["pep_coord"][:0].to(self.device)
        pep_aa = example["pep_aa"][:0].to(self.device)

        for step in range(n_steps):
            new_coord = _coord_from(par_x, par_o)
            query = {
                "rec_coord": rec_coord,
                "rec_aa": rec_aa,
                "pep_coord": pep_coord,
                "pep_aa": pep_aa,
                "qry_coord": new_coord.unsqueeze(0),
                "qry_aa": torch.tensor([0], device=self.device),
            }
            logits = self.density_model(_recursive_to(_data_to_batch(query), self.device))[0, 0]
            prob = F.softmax(logits, dim=-1)
            obj = -torch.log((prob[:20] * F.softmax(par_cls / par_T.exp(), dim=-1)).sum(-1))
            optimizer.zero_grad()
            obj.backward()
            optimizer.step()
            with torch.no_grad():
                par_x.add_(noise_eps * torch.randn_like(par_x))
                par_o.add_(noise_eps * torch.randn_like(par_o))
                par_o.copy_(F.normalize(par_o, p=2, dim=-1))
            if verbose and step % max(n_steps // 5, 1) == 0:
                print(
                    f"[PepHARHotspotSampler] step={step} obj={obj.item():.4f} "
                    f"type={int(par_cls.argmax(-1).item())}"
                )

        return HotspotAnchor(
            residue_index=seed_anchor.residue_index,
            backbone_coords=_coord_from(par_x, par_o).detach().cpu(),
            residue_type=int(par_cls.argmax(-1).detach().cpu().item()),
            score=float(F.softmax(par_cls.detach().cpu(), dim=-1).max().item()),
            source="pephar_density",
        )
