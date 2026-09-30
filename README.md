# HotFlow

**Hotspot-driven full-atom peptide binder design with SE(3) flow matching.**

HotFlow is a two-stage framework for structure-based peptide binder design. An
energy-based model identifies the binding hotspots a peptide should occupy on a
receptor surface; a flow matching network then generates the full-atom peptide
structure and sequence conditioned on those hotspots.

It combines two published methods:

- **[PepHAR](https://github.com/Ced3-han/PepHAR)** (ICLR 2025) predicts hotspot
  residue densities on the receptor surface and grows peptides by autoregressive
  fragment extension. Strong interface quality, but sequential placement
  introduces steric clashes — later fragments cannot see the global geometry.
- **[PepFlow](https://github.com/Ced3-han/PepFlowww)** (ICML 2024) generates
  full-atom peptides non-autoregressively with multi-modal flow matching on
  SE(3) × torus. Globally consistent conformations, but it treats all receptor
  surface residues uniformly and has no notion of which ones matter for binding.

HotFlow keeps PepHAR's hotspot awareness and PepFlow's global consistency by
using the former as a frozen conditioning signal for the latter.

![HotFlow architecture](docs/architecture.png)

<sub>Cross-attention is injected into the last three of PepFlow's six IPA blocks;
blocks 0–2 stay as pretrained. Figure from the final project presentation.</sub>

> **Status.** Built at KAIST (BS89902) and published here as a record of the
> work — research code, not a maintained tool. The full pipeline is implemented:
> data, training, sampling, and a four-method evaluation harness, with unit
> tests last run during development.
>
> **The headline result is negative.** Hotspot conditioning places the peptide in
> the right pocket and cuts steric clashes, but does not produce peptides that
> are simultaneously well-formed, correctly posed and engaging the intended
> residues. The more useful outcomes are the diagnosis of *why* standard metrics
> obscured this, and the finding that anchor quality — not conditioning capacity
> — is the binding constraint. See [Findings](#findings).
>
> Evaluation is also incomplete: the 179-entry test set run used
> ground-truth-derived anchors at inference time, which does not match the
> intended de novo protocol, so those numbers are not reported. The two de novo
> case studies are unaffected and are reported.

---

## Approaches

The repository implements three ways to combine the two models, plus the two
baselines.

### Approach B — cross-attention conditioning (the main method)

```
Stage 1   receptor pocket
              │
              ▼
          PepHAR EBM density model  (frozen, pretrained)
              │
              ▼
          K hotspot anchors  —  CA/C/N coordinates + amino-acid type
              │
Stage 2       ▼
          HotspotEncoder  →  anchor context  (B, K, d_hotspot)
              │
              ▼
          cross-attention injected into PepFlow IPA encoder blocks 3/4/5
              │
              ▼
          peptide nodes attend to the anchor context
              │
              ▼
          conditioned structure + sequence generation
```

Only the cross-attention layers and the `HotspotEncoder` are trained; PepHAR
stays frozen, and PepFlow is initialised from its pretrained weights. Hotspot
context is dropped with p = 0.1 during training so that classifier-free guidance
is available at sampling time (`--guidance_scales`).

Implemented in [`hotflow/models/flow_model_b.py`](hotflow/models/flow_model_b.py),
[`hotflow/models/hotspot_encoder.py`](hotflow/models/hotspot_encoder.py) and
[`hotflow/models/ga_encoder_crossattn.py`](hotflow/models/ga_encoder_crossattn.py).

### Approach A — inpainting (ablation baseline)

PepHAR predicts the anchors, K peptide residues are frozen at those positions,
and a pretrained PepFlow fills in the remaining L−K residues. A hard constraint
with no learned conditioning and no training step — every module is pretrained.

Implemented in [`hotflow/models/hotflow_model_revised.py`](hotflow/models/hotflow_model_revised.py)
and [`hotflow/sampling/inpainting.py`](hotflow/sampling/inpainting.py).

This is what motivated the per-modality generate masks in the PepFlow patch:
an anchor residue needs its backbone frozen while its sequence and torsions are
still generated, which upstream's single `generate_mask` cannot express.

### Unified — anchors as graph nodes (exploratory)

Instead of cross-attention, anchors are appended to the residue graph as a third
node type and attended to by the ordinary IPA machinery.
See [`hotflow/models/unified/flow_model_unified.py`](hotflow/models/unified/flow_model_unified.py).

### Baselines

**PepFlow** (receptor only, no hotspot conditioning) and **PepHAR** (its own EBM
density plus sequence prediction network).

---

## Where anchors come from

This is the detail most worth tracking when reading the code, because it differs
between training and inference by design.

| Method | Anchors during training | Anchors at inference |
|---|---|---|
| PepFlow | — | — |
| PepHAR | — (trained upstream) | own EBM density |
| Approach A | — (no training) | PepHAR EBM prediction |
| Approach B | **ground-truth contacts** | **PepHAR EBM prediction** |

Approach B trains on clean ground-truth anchors — the top-K peptide residues by
receptor contact frequency within 4 Å — for stability, then runs de novo from
PepHAR predictions. The resulting train/inference distribution shift is a
deliberate design choice, and one the evaluation is meant to probe.

An `--ab_anchor_source` flag on `eval_compare.py` selects between
`pephar_denovo` (default), `pephar_gt_seeded` and `gt_contact` so the shift can
be measured directly. `hotflow/scripts/precompute_pephar_anchors.py` and the
`configs/train_b_pephar_*.yaml` configs additionally support *training* on
precomputed PepHAR anchors, which closes the gap at considerable preprocessing
cost (the script's own docstring quotes ~26 h for one sample per training
complex).

### Terminology

- **Hotspot residue** — a *peptide* residue contacting the receptor within 4 Å.
  It is a property of the peptide, defined after the fact from a native complex.
- **Hotspot anchor** — the spatial template standing in for a hotspot residue:
  CA/C/N coordinates plus an amino-acid type. This is the model's conditioning
  input.
- **K** — the number of anchors, 5 by default.

### Anchor tensor format

Anything producing anchors must emit this signature:

| Field | Shape | dtype | Meaning |
|---|---|---|---|
| `anchor_coords` | (B, K, 3, 3) | float | backbone coordinates, ordered CA, C, N |
| `anchor_types` | (B, K) | long | amino-acid index, 0–19 (20 = unknown) |
| `anchor_mask` | (B, K) | bool | validity |

---

## Setup

### 1. Environment

```bash
conda create -n hotflow python=3.10 -y
conda activate hotflow

# PyTorch first — match the CUDA version to your machine
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
pip install torch_scatter==2.1.2+pt26cu124 \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

pip install -r requirements.txt
```

PyRosetta is needed for the default evaluation path (FastRelax and interface
energies) and requires a separate licence:

```bash
pip install pyrosetta-installer
python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"
```

### 2. Upstream code

PepFlow and PepHAR are **not vendored** in this repository. Both are MIT
licensed, and HotFlow depends on small modifications to each, so they are
reconstructed from pinned upstream commits plus patches:

```bash
bash scripts/setup_upstream.sh
```

This clones both repositories into `PepFlowww/` and `PepHAR/` and applies
`patches/`. The patches are required, not optional — stock upstream checkouts
will fail on import. [`patches/README.md`](patches/README.md) documents exactly
what they change and why.

Pretrained weights are distributed by the upstream authors via Google Drive and
must be downloaded by hand:

- `PepFlowww/model2.pt` — see the [PepFlow repository](https://github.com/Ced3-han/PepFlowww)
- `PepHAR/ckpts/` — see the [PepHAR Drive folder](https://drive.google.com/drive/folders/1jJFPZbczI7Nxai-9X5UcNsv5U8rcBUEY),
  which should yield `density_v4_x5o2_2024_09_08__11_25_36/` and
  `prediction_d2_x2o1_2024_09_08__11_21_33/`

`scripts/setup_upstream.sh` prints these instructions again when it finishes.

### 3. Data

Training uses [PepBDB](http://huanglab.phys.hust.edu.cn/pepbdb/). Expected
layout, with one directory per complex holding `receptor.pdb` and `peptide.pdb`:

```
data/
├── raw/
│   ├── pepbdb/pepbdb/      # all complexes
│   └── pepbdb_test/        # held-out subset
├── processed/pepbdb/       # LMDB caches, built on first run
├── names.txt               # test-split IDs, excluded from the train cache
└── test_pdb_ids.txt
```

The LMDB structure caches are built automatically on first use from
`structure_dir`, by PepFlow's `PepDataset`. Paths are set per split in the
config's `dataset:` block and resolve relative to the repository root.

`data/names.txt` drives test-split exclusion when the training cache is built.
Because the same filter also applies while building the *test* cache, move the
file aside for that one build and restore it afterwards.

### 4. Case-study targets (optional)

The de novo benchmark reads its receptor structures from `test_cases/`. They are
public RCSB entries and are not redistributed here:

```bash
bash scripts/fetch_targets.sh
```

### 5. P2Rank (optional)

Pocket detection for de novo targets shells out to P2Rank at
`tools/p2rank_2.4.2/prank` and needs a JVM. Only
`hotflow/benchmark_targets.py` uses it.

### Running

Every entry point assumes the repository root as the working directory:

```bash
cd /path/to/HotFlow
export PYTHONPATH=$PWD
```

This matters more than usual here. The entry-point scripts deliberately strip
their own directory from `sys.path` before inserting `PepFlowww/`, because
`hotflow/data/` would otherwise shadow PepFlow's top-level `data` package.
Library modules only prepend, so importing them without the repository root on
`PYTHONPATH` is the usual cause of a confusing `data` import error.

---

## Usage

### Train Approach B

```bash
python hotflow/train_b.py --config hotflow/configs/train_b.yaml --name hotflow_b
```

Multi-GPU via `torchrun` (DDP is detected from `WORLD_SIZE`):

```bash
torchrun --nproc_per_node=4 hotflow/train_b.py \
    --config hotflow/configs/train_b.yaml --num_workers 4
```

Checkpoints land in `logs_b/<config>_<timestamp>/checkpoints/<iter>.pt` every
`train.val_freq` iterations, each carrying config, model, optimizer, scheduler
and iteration. `--resume <ckpt>` continues a run in place; `--debug` disables
wandb, which is otherwise a hard dependency.

Initialisation is PepFlow's pretrained `model2.pt` loaded with `strict=False`,
so the new cross-attention and encoder parameters start random. Uncomment the
`pretrained:` block in the config to enable it.

Sample `scripts/*.sh` are provided for Slurm. They honour `HOTFLOW_ROOT`,
`HOTFLOW_ENV` and `CUDA_VISIBLE_DEVICES`, and otherwise expect to be submitted
from the repository root.

### Compare all four methods

```bash
python -m hotflow.eval_compare \
    --config hotflow/configs/train_b.yaml \
    --pepflow_ckpt PepFlowww/model2.pt \
    --approach_b_ckpt logs_b/<run>/checkpoints/<iter>.pt \
    --ab_anchor_source pephar_denovo \
    --num_samples 200 --num_steps 100 \
    --outdir results/eval
```

Writes `comparison.csv` and `summaries.json`. Rosetta FastRelax is **on by
default** here (`--no_rosetta` to skip); `--skip_approach_a` and `--skip_pephar`
drop methods. Note that `eval_b.py`, which evaluates a single Approach B
checkpoint, takes the opposite default and needs an explicit `--rosetta`.

### Sweep checkpoints

```bash
python -m hotflow.select_ckpt \
    --ckpt_dir logs_b/<run>/checkpoints \
    --every 100000 --rank_by rmsd_aligned \
    --outdir results/ckpt_sweep
```

A fast scan on a small test subset with Rosetta disabled.

### De novo targets

```bash
python -m hotflow.benchmark \
    --targets 9CDZ 7UXO --methods PepFlow PepHAR Approach_A Approach_B \
    --approach_b_ckpt logs_b/<run>/checkpoints/<iter>.pt \
    --outdir test_cases
```

Targets are defined in `hotflow/benchmark_targets.py:TARGET_CONFIGS`: **9CDZ,
7UXO, 6YVR, 4Y5U, 8TF5** and **6LUQ**. Fetch their structures first — they are
read from `test_cases/`, which is also where results are written:

```bash
bash scripts/fetch_targets.sh
```

---

## Evaluation protocol

All metrics are measured **after** Rosetta FastRelax (N = 2), on the relaxed
structure rather than the raw generated one:

- Structure — CA-RMSD, TM-score, interface RMSD
- Sequence — amino-acid recovery
- Interface — anchor contact rate, binding-site overlap, clash count
- Energy — full-atom REU (`stab`) and `dG_separated` (`bind`)

Splits: 179 held-out PepBDB chain entries, plus a small set of case-study
targets for de novo design.

### What is and is not measured here

The **179-entry PepBDB test set** results are not reported. Approach A and B
were meant to be evaluated de novo — anchors predicted by PepHAR from the
receptor alone — but the one full test set run fed them anchors derived from
ground-truth peptide contacts instead. That inflates them relative to the
baselines and does not measure the de novo setting the method claims. Re-running
with `--ab_anchor_source pephar_denovo` (now the default) is the outstanding
work. Training-time use of ground-truth anchors is *not* affected — that is the
intended design, and the trained model does not need to be redone.

The **two case studies below are de novo** and are reported. 6LUQ has no native
peptide at all (`has_gt: False`), and the 9CDZ run explicitly disables GT
anchors during sampling, using the native peptide only as a reference pose for
post-hoc metrics.

---

## Findings

The headline result is negative, and the most useful thing this project produced
is the diagnosis of *why* the usual metrics did not show it.

### Validity is a misleading headline metric

6LUQ (D2 dopamine receptor), de novo, n = 50 per method, after FastRelax:

| | PepFlow | PepHAR | HotFlow |
|---|---:|---:|---:|
| valid | 0.84 | **1.00** | 0.08 |
| clash-free | 1.00 | 1.00 | 1.00 |
| Rosetta bind (REU) ↓ | −18.8 | −11.1 | **−19.3** |
| Rosetta stab (REU) ↓ | 42.7 | **31.3** | 97.8 |

Read naively, PepHAR wins outright and HotFlow collapses. Neither reading
survives contact with the structures:

- **PepHAR's 100 % is cheap.** Autoregressive placement preserves *local*
  backbone geometry by construction, so a connected chain is trivially "valid"
  regardless of whether it binds — and PepHAR binds weakest of the three
  (−11.1 REU), drifting out of the pocket after relaxation.
- **HotFlow's 8 % is largely an aggregation artifact.** The threshold is
  all-or-nothing at 4.5 Å, so roughly one localized CA break zeroes an otherwise
  intact peptide. The project presentation put HotFlow's per-bond backbone
  quality at 49 % against PepHAR's 51 % — that figure comes from the
  presentation analysis and is not reproducible from the metrics shipped here,
  but the mean CA–CA distances (3.96 Å vs 3.79 Å) are consistent with a
  localized defect rather than a global collapse.
- **"Clash-free" is partly a FastRelax artifact.** These are post-relaxation
  numbers, and relaxation removes most clashes. The presentation reports that
  raw outputs of all three methods clash; the 9CDZ study below quantifies it
  (12.7–31.3 mean receptor clashes before relaxation).

**Sequence collapse is not confined to PepHAR.** Glycine fraction across the
same 50 samples per method, computed from `pred_seq`:

| | PepFlow | PepHAR | HotFlow |
|---|---:|---:|---:|
| mean Gly fraction | 0.30 | 0.50 | **0.61** |
| samples ≥ 50 % Gly | 2/50 | 24/50 | **34/50** |

HotFlow is the *most* Gly-collapsed of the three (e.g. `GGGGGGGGRGIS`), while
also scoring the best binding energy. So low-complexity sequence and apparent
binding quality are not in tension here, and the collapse cannot be used to
dismiss any one method's score — it is a shared failure mode that the
interface-level metrics do not penalise.

The same structures change rank depending on the threshold, on all-or-nothing
versus per-bond aggregation, and on whether the metric is computed before or
after relaxation.

### Hotspot information does not guarantee better binding

PepHAR carries an explicit hotspot density and still binds weakest. PepFlow's
strong RMSD and sequence recovery on the held-out set largely reflect
reconstruction of the native peptide rather than de novo physical reasoning.
Neither demonstrates a hotspot-driven affinity gain.

This also means generator effects and inductive-bias effects have to be
separated before either can be credited: PepHAR's diversity comes from
autoregression, and PepFlow's RMSD from ground-truth recovery. Both are
properties of the generator, independent of whether hotspots help.

### MDM2 follow-up

9CDZ (MDM2), held out from training, de novo, n = 20 per method, native 16-mer
used only as a reference pose. After FastRelax:

| | mean iRMSD (Å) ↓ | mean overlap ↑ | clash-free | strict pass |
|---|---:|---:|---:|---:|
| PepFlow | 6.87 | **0.644** | 1.00 | 0.00 |
| PepHAR | 6.49 | 0.455 | 0.95 | 0.05 |
| HotFlow | **5.92** | 0.594 | 1.00 | 0.05 |

Strict pass: `valid & iRMSD < 6 Å & receptor_clashes < 10 & overlap > 0.75`.

FastRelax removed nearly all steric clashes, but the relaxed designs still did
not recover the native MDM2 binding mode. HotFlow achieved the best mean iRMSD
and, before relaxation, far fewer receptor clashes than either baseline (12.7 vs
31.3 and 25.9), yet only one sample in twenty passed the strict filter — as did
PepHAR, and none of PepFlow's.

Measured against the native hotspot positions, HotFlow was in fact the *worst*
of the three at hotspot-level alignment: it contacted 0.35 of the native hotspot
positions, against 0.58 for PepFlow and 0.68 for PepHAR. The PepHAR-derived
anchors placed the peptide in the right pocket but not on the right residues.

### The anchor source is the binding constraint

The same target run with ground-truth anchors instead of PepHAR-predicted ones
isolates the cost of that mismatch. Both rows below are pre-relaxation, so they
are directly comparable:

| HotFlow on 9CDZ | n | mean iRMSD (Å) ↓ | mean overlap ↑ | mean receptor clashes ↓ | strict pass ↑ |
|---|---:|---:|---:|---:|---:|
| GT anchors *(optimistic, not de novo)* | 50 | 5.58 | 0.815 | 11.1 | **0.42** |
| PepHAR de novo anchors | 20 | 6.60 | 0.826 | 12.7 | **0.00** |

Swapping only the anchor source takes the strict-pass rate from 42 % to zero
while binding-site overlap and clash count barely move. The cross-attention
machinery can exploit good anchors; PepHAR's de novo predictions are not yet
good enough to supply them. The GT-anchor row is *not* a de novo result and is
shown only to locate the bottleneck.

### Interpretation

Hotspot conditioning delivers a real but narrow benefit: it places the peptide
in the intended pocket and substantially reduces steric clashes relative to
PepFlow. It does not deliver a peptide that is simultaneously well-formed,
correctly posed, and engaging the intended residues.

Three distinct failures stack up, and separating them is the main thing this
project established:

1. **Anchor quality.** The train/inference mismatch is not a technicality — it
   is the dominant term. HotFlow trains on clean ground-truth contacts and then
   runs on PepHAR predictions that are accurate enough for pocket placement but
   not for residue-level alignment. The 42 % → 0 % drop above is the cost.
2. **Geometry.** Conditioning steers *where* the peptide engages but supplies no
   pressure toward a chemically valid backbone, so in a tight pocket the model
   satisfies the anchors by straining the chain (the 6LUQ result above).
3. **Sequence.** All three methods drift toward low-complexity glycine-rich
   sequences, HotFlow most of all, and none of the interface metrics penalise
   it. Any future comparison needs a sequence-quality term, or it will keep
   scoring poly-Gly chains as successes.

Hotspot information alone is therefore insufficient for chemically robust,
strongly binding designs. The direction still looks worth pursuing — with good
anchors the conditioning clearly works — but the next step is a better hotspot
predictor and explicit geometric inductive biases in the generator, not more
cross-attention capacity.

---

## Repository layout

```
hotflow/
├── models/
│   ├── flow_model_b.py            FlowModelB — Approach B, PepFlow + cross-attention
│   ├── hotspot_encoder.py         anchors → conditioning context
│   ├── ga_encoder_crossattn.py    IPA encoder with cross-attention blocks
│   ├── hotflow_model_revised.py   Approach A orchestration (de novo)
│   ├── hotspot_sampler_revised.py PepHAR de novo anchor sampler wrapper
│   ├── hotspot_predictor*.py      standalone DETR-style anchor predictors
│   └── unified/                   anchors-as-graph-nodes variant
├── data/
│   ├── dataset_b.py               PepDatasetB — PepDataset + hotspot transform
│   ├── hotspot_labeling.py        ground-truth contact labelling, top-K selection
│   ├── transforms.py              annotation / graph / PepHAR-cache transforms
│   ├── p2rank_bridge.py           P2Rank pockets → anchors
│   └── pepbdb_bridge.py           PepBDB → PepFlow and PepHAR formats
├── losses/contact_preservation.py hinge loss keeping anchors in contact
├── sampling/inpainting.py         Approach A hard constraints
├── utils/                         PDB I/O, metrics, Rosetta scoring
├── configs/                       training configs and ablations
├── train_b.py                     Approach B training
├── eval_compare.py                four-method comparison
├── select_ckpt.py                 checkpoint sweep
└── benchmark.py                   de novo target case studies

patches/       upstream modifications, applied by scripts/setup_upstream.sh
scripts/       upstream setup, target download, Slurm job scripts
tests/         unit tests
docs/          architecture figure; an earlier schematic and its generator
```

### Configs

`configs/train_b.yaml` is the reference. K = 5 anchors, cross-attention in
encoder blocks 3/4/5, `d_hotspot` 64, hotspot dropout 0.1, Adam at lr 5e-4.
Loss weights: translation 0.5, rotation 0.5, backbone atom 0.25, sequence 1.0,
angle 1.0, torsion 0.5, contact preservation 0.1.

The variants cover a contact-weight ablation (`train_b_cw05.yaml`),
`HotspotEncoder` ablations (`train_b_ablation_*.yaml` — SE(3)-invariant encoding
and anchor self-attention on or off) and training on precomputed PepHAR anchors
(`train_b_pephar_*.yaml`).

### Tests

```bash
PYTHONPATH=$PWD pytest tests/
```

Covers the data pipeline, `FlowModelB`, `HotspotEncoder`, the P2Rank bridge, the
unified model, and an overfit sanity check. The tests need the upstream code in
place but not the datasets.

---

## Licence

MIT — see [LICENSE](LICENSE).

PepFlow and PepHAR are MIT licensed and © their respective authors. They are not
redistributed here; `patches/` contains only the modifications made for this
project. PepBDB and the pretrained checkpoints are covered by their own terms.

## Citation

This work builds directly on:

```bibtex
@inproceedings{li2024pepflow,
  title     = {Full-Atom Peptide Design based on Multi-modal Flow Matching},
  author    = {Li, Jiahan and Cheng, Chaoran and Wu, Zuofan and Guo, Ruihan and
               Luo, Shitong and Ren, Zhizhou and Peng, Jian and Ma, Jianzhu},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2024},
  url       = {https://arxiv.org/abs/2406.00735}
}

@inproceedings{li2025pephar,
  title     = {Hotspot-Driven Peptide Design via Multi-Fragment Autoregressive Extension},
  author    = {Li, Jiahan and Chen, Tong and Luo, Shitong and Cheng, Chaoran and
               Guan, Jiaqi and Guo, Ruihan and Wang, Sheng and Liu, Ge and
               Peng, Jian and Ma, Jianzhu},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2411.18463}
}
```
