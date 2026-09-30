# Upstream patches

HotFlow builds on two MIT-licensed repositories that are **not vendored** into
this repository. `scripts/setup_upstream.sh` clones both at a pinned commit and
applies the patches here.

| Patch | Upstream | Pinned commit |
|---|---|---|
| `pepflow.patch` | [Ced3-han/PepFlowww](https://github.com/Ced3-han/PepFlowww) | `16e0d26` (2025-05-25) |
| `pephar.patch` | [Ced3-han/PepHAR](https://github.com/Ced3-han/PepHAR) | `4a809d5` (2025-05-05) |

Both patches were verified to apply cleanly to a fresh clone of the pinned
commit and to reproduce the development tree byte-for-byte.

## What `pepflow.patch` changes

Three files, ~138 changed lines.

**`models_con/flow_model.py`** — splits the single `generate_mask` into three
per-modality masks: `backbone_generate_mask`, `sequence_generate_mask` and
`torsion_generate_mask`, read through a new `_get_modality_generate_masks()`
helper. Each falls back to `generate_mask` when absent, so upstream behaviour is
unchanged unless the new keys are supplied.

This is what makes **Approach A (inpainting)** expressible: anchor residues can
have their backbone frozen to the predicted hotspot position while their
sequence and torsions are still generated. Without it the model can only freeze
or generate a residue in all modalities at once.

**`models_con/pep_dataloader.py`** — adapts the loader to PepBDB:

- Replaces the upstream hardcoded test-split path
  (`/datapool/data2/home/ruihan/...`) with a repo-relative `data/names.txt`
  lookup that is skipped when the file is absent.
- Reads `receptor.pdb` and extracts the pocket on the fly (residues with a CA
  within 10 Å of any peptide heavy atom) instead of expecting a prebuilt
  `pocket.pdb`, which PepBDB does not ship.
- Drops complexes containing non-standard residues, and resets `chain_nb` after
  the pocket filter.

**`pepflow/modules/common/geometry.py`** — adds `from __future__ import
annotations` (one line) for compatibility with the Python version used here.

## What `pephar.patch` changes

Adds one new file, `evaluate/sample_revised.py`, containing
`AnchorBasedSamplerDenovo`: a de novo anchor sampler that initialises hotspot
anchors from the receptor surface rather than from ground-truth peptide
coordinates, as upstream `evaluate/sample.py` does.

`hotflow/models/hotspot_sampler_revised.py`, `hotflow/eval_compare.py` and
`hotflow/scripts/precompute_pephar_anchors.py` all import from it, so Approach A
and Approach B inference will not run against a stock PepHAR checkout.
