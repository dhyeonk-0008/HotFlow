# HotFlow

**Hotspot-driven peptide binder design with SE(3) flow matching.**

Peptide binders are designed against a target protein by first predicting which
surface hotspots the peptide should occupy, then generating the full-atom
peptide conditioned on them. HotFlow combines two published models:
[PepHAR](https://github.com/Ced3-han/PepHAR) (ICLR 2025) predicts hotspot
anchors and is used frozen; [PepFlow](https://github.com/Ced3-han/PepFlowww)
(ICML 2024) generates structure and sequence by flow matching. Cross-attention
layers inserted into PepFlow's encoder let the peptide attend to the anchors.

Built at KAIST (BS89902).

![HotFlow architecture](docs/architecture.png)

## What I built

- **Cross-attention conditioning** (`hotflow/models/`) — a `HotspotEncoder` that
  turns K hotspot anchors into a context tensor, and cross-attention injected
  into the last three of PepFlow's six IPA blocks. Trained from PepFlow's
  pretrained weights; only the new layers learn. Classifier-free guidance via
  hotspot dropout.
- **An inpainting baseline** — anchors imposed as a hard constraint instead of a
  learned signal, to isolate what the training actually buys. This needed
  per-modality generate masks in PepFlow (freeze an anchor's backbone while its
  sequence is still generated), which upstream could not express.
- **Data pipeline** (`hotflow/data/`) — contact-based hotspot labelling from
  native complexes, PepBDB adaptation, and P2Rank pocket detection for targets
  with no known binder.
- **Evaluation harness** (`hotflow/eval_compare.py`) — four methods compared on
  a held-out set and on de novo targets, scored after Rosetta FastRelax on
  structure, sequence, interface and energy metrics.

## Result

The result is negative, and the interesting part is why.

Hotspot conditioning reliably places the peptide in the intended pocket and cuts
steric clashes, but does not produce peptides that are simultaneously
well-formed, correctly posed, and engaging the intended residues. Two things
turned out to matter more than the conditioning architecture. First, **anchor
quality dominates**: the model trains on ground-truth contacts but runs on
PepHAR predictions, and on an MDM2 case study, swapping only the anchor source
dropped the pass rate from 42 % to zero while pocket overlap barely moved.
Second, **the standard metrics hide this** — "validity" flips ranking depending
on threshold, on per-bond versus all-or-nothing aggregation, and on whether it
is measured before or after relaxation, and every method drifts toward
low-complexity glycine-rich sequences that no interface metric penalises.

Detailed numbers and the evaluation packages are not included here; the
held-out-set evaluation under the correct de novo protocol was not finished.

## Setup

```bash
conda create -n hotflow python=3.10 -y && conda activate hotflow
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install torch_scatter==2.1.2+pt26cu124 -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install -r requirements.txt
```

PepFlow and PepHAR are not vendored. Both are MIT licensed and both needed small
changes, so they are reconstructed from pinned commits plus patches:

```bash
bash scripts/setup_upstream.sh
```

Pretrained weights are distributed by the upstream authors and must be
downloaded by hand — see the paths the script prints when it finishes.
Training data is [PepBDB](http://huanglab.phys.hust.edu.cn/pepbdb/), expected
under `data/raw/pepbdb/`; LMDB caches build on first run. PyRosetta (licensed
separately) is needed for evaluation.

## Usage

Run from the repository root with `PYTHONPATH=$PWD`.

```bash
# train
python hotflow/train_b.py --config hotflow/configs/train_b.yaml

# evaluate all four methods
python -m hotflow.eval_compare \
    --config hotflow/configs/train_b.yaml \
    --pepflow_ckpt PepFlowww/model2.pt \
    --approach_b_ckpt logs_b/<run>/checkpoints/<iter>.pt \
    --ab_anchor_source pephar_denovo \
    --outdir results/eval
```

`hotflow/benchmark.py` runs the de novo case-study targets;
`bash scripts/fetch_targets.sh` downloads their structures first.

## Layout

```
hotflow/models/     cross-attention model, hotspot encoder, samplers
hotflow/data/       hotspot labelling, PepBDB and P2Rank bridges
hotflow/configs/    training configs and encoder ablations
hotflow/            train / eval / benchmark entry points
patches/            upstream modifications (see patches/README.md)
scripts/            setup and Slurm job scripts
tests/              unit tests
```

## Licence

MIT — see [LICENSE](LICENSE). PepFlow and PepHAR are MIT licensed and © their
respective authors; `patches/` holds only the modifications made here. PepBDB
and the pretrained checkpoints carry their own terms.

## Citation

- Li et al., *Full-Atom Peptide Design based on Multi-modal Flow Matching*, ICML 2024. [arXiv:2406.00735](https://arxiv.org/abs/2406.00735)
- Li et al., *Hotspot-Driven Peptide Design via Multi-Fragment Autoregressive Extension*, ICLR 2025. [arXiv:2411.18463](https://arxiv.org/abs/2411.18463)
