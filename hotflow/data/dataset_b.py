"""Dataset and DataLoader utilities for Approach B training.

Wraps PepFlow's PepDataset with HotspotAnnotationTransform and provides
a convenience function to build a ready-to-use DataLoader.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
PEPFLOW_ROOT = REPO_ROOT / "PepFlowww"
if str(PEPFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(PEPFLOW_ROOT))

from pepflow.utils.data import PaddingCollate  # noqa: E402
from models_con.pep_dataloader import PepDataset  # noqa: E402

from hotflow.data.transforms import HotspotAnnotationTransform  # noqa: E402


class PepDatasetB(Dataset):
    """PepDataset wrapper that applies a hotspot annotation transform.

    Wraps an existing PepDataset (or any dataset returning PepFlow-format
    dicts) and applies hotspot annotation + optional extra transforms.

    Args:
        base_dataset: underlying dataset (typically PepDataset).
        num_anchors: K, number of hotspot anchors.
        distance_cutoff: contact distance threshold for labeling.
        hotspot_transform: pre-built hotspot transform (e.g.
            `PepHARAnchorTransform` to read pre-computed EBM anchors). If
            None, defaults to `HotspotAnnotationTransform` (GT contacts).
        extra_transform: optional additional transform applied after
            hotspot annotation.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        num_anchors: int = 5,
        distance_cutoff: float = 4.0,
        hotspot_transform: Optional[Any] = None,
        extra_transform: Optional[Any] = None,
    ):
        self.base_dataset = base_dataset
        if hotspot_transform is None:
            hotspot_transform = HotspotAnnotationTransform(
                num_anchors=num_anchors,
                distance_cutoff=distance_cutoff,
            )
        self.hotspot_transform = hotspot_transform
        self.extra_transform = extra_transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        data = self.base_dataset[index]
        data = self.hotspot_transform(data)
        if self.extra_transform is not None:
            data = self.extra_transform(data)
        return data


# Pad values for the new hotspot fields
HOTSPOT_PAD_VALUES = {
    'anchor_types': 20,       # unknown residue type
    'hotspot_labels': False,
    'anchor_mask': False,
}

# Anchor fields have fixed size K (not L) — must NOT be padded along dim 0.
HOTSPOT_NO_PADDING = {
    'anchor_indices',
    'anchor_coords',
    'anchor_types',
    'anchor_mask',
}


def build_dataloader_b(
    structure_dir: str,
    dataset_dir: str,
    name: str,
    batch_size: int = 32,
    num_anchors: int = 5,
    distance_cutoff: float = 4.0,
    num_workers: int = 4,
    shuffle: bool = True,
    reset: bool = False,
) -> DataLoader:
    """Build a DataLoader for Approach B training/validation.

    Args:
        structure_dir: path to PepBDB/PepMerge structures.
        dataset_dir: path to LMDB cache directory.
        name: dataset name (e.g., 'pep_pocket_train').
        batch_size: batch size.
        num_anchors: K hotspot anchors per sample.
        distance_cutoff: contact cutoff for hotspot labeling.
        num_workers: DataLoader workers.
        shuffle: whether to shuffle.
        reset: whether to rebuild LMDB cache.

    Returns:
        DataLoader yielding batches with hotspot annotation fields.
    """
    base_dataset = PepDataset(
        structure_dir=structure_dir,
        dataset_dir=dataset_dir,
        name=name,
        transform=None,
        reset=reset,
    )

    dataset = PepDatasetB(
        base_dataset=base_dataset,
        num_anchors=num_anchors,
        distance_cutoff=distance_cutoff,
    )

    # Merge pad values and no-padding sets
    from pepflow.utils.data import DEFAULT_PAD_VALUES, DEFAULT_NO_PADDING
    pad_values = {**DEFAULT_PAD_VALUES, **HOTSPOT_PAD_VALUES}
    no_padding = set(DEFAULT_NO_PADDING) | HOTSPOT_NO_PADDING

    collate_fn = PaddingCollate(eight=False, pad_values=pad_values, no_padding=no_padding)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
