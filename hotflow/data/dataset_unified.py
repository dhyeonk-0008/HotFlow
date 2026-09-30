"""Dataset and DataLoader utilities for unified graph architecture training.

Wraps PepFlow's PepDataset with HotspotAnnotationTransform ->
AnchorToGraphTransform, which appends anchor nodes to the residue graph.
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

from hotflow.data.transforms import (  # noqa: E402
    HotspotAnnotationTransform,
    AnchorToGraphTransform,
)
from hotflow.data.dataset_b import HOTSPOT_PAD_VALUES, HOTSPOT_NO_PADDING  # noqa: E402


class PepDatasetUnified(Dataset):
    """PepDataset wrapper that appends anchor nodes to the residue graph.

    Applies HotspotAnnotationTransform (GT contact labels + anchor
    selection) followed by AnchorToGraphTransform (anchor → graph node
    expansion).  An optional extra hotspot_transform can override the
    GT anchors (e.g., PepHARAnchorTransform for EBM-derived anchors)
    before graph expansion.

    Args:
        base_dataset: underlying dataset (typically PepDataset).
        num_anchors: K, number of hotspot anchors.
        distance_cutoff: contact distance threshold for labeling.
        hotspot_transform: pre-built hotspot transform. If None, defaults
            to HotspotAnnotationTransform (GT contacts).
        extra_transform: optional transform applied after graph expansion.
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
        self.graph_transform = AnchorToGraphTransform()
        self.extra_transform = extra_transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        data = self.base_dataset[index]
        data = self.hotspot_transform(data)
        data = self.graph_transform(data)
        if self.extra_transform is not None:
            data = self.extra_transform(data)
        return data


# Pad values: inherit hotspot pad values + unified-specific fields.
UNIFIED_PAD_VALUES = {
    **HOTSPOT_PAD_VALUES,
    'node_type': 0,         # pad as receptor type
    'hotspot_labels': False,
}

# Fields with fixed size (not padded along L dimension).
UNIFIED_NO_PADDING = HOTSPOT_NO_PADDING | {'L_orig'}
