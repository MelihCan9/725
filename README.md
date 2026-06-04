# DI 725 Term Project — Multi-modal Transformers

**Author:** Melih Can Hamurcu
**Course:** DI 725 — Transformers and Attention-Based Deep Networks
**Institution:** Middle East Technical University, Graduate School of Informatics

## Project Overview

This project investigates whether **segmentation-derived semantic information** can improve
remote-sensing image captioning when injected into a frozen vision-language model via
**learned prefix embeddings** (segmentation-conditioned prompt learning).

The base BLIP weights are frozen throughout; only the learnable prefix (B1) or the
segmentation-to-prefix bridge network (M1) receives gradients.

| Tag | Description |
|-----|-------------|
| **B0** | Zero-shot pretrained BLIP (Phase 1 baseline) |
| **B1** | BLIP + unconditional learnable prefix (control) |
| **M1** | BLIP + segmentation-conditioned prefix (proposed) |

## Phase 3 — final experiments & ablations (current)

Phase 3 runs on the full 10,000-image dataset and contributes:

**Length-aware baseline reporting.** B0's near-zero BLEU is a length artefact, not a lack of
lexical overlap, so we report the brevity penalty, length ratio, and brevity-penalty-free
unigram precision alongside BLEU, and add **BERTScore-F1** (length-robust, semantic) for all
three models.

**Ablation suite (around M1):**

- **A — segmentation fidelity:** decode M1 with the true vs. a mismatched vs. a zero
  segmentation vector (no retraining) to test whether the prefix actually uses the signal.
- **B — input modality:** the 7-dim land-cover composition vector (MLP) vs. the segmentation
  **mask image** (small CNN). The vector is the spatial average of the mask, so this isolates
  the contribution of spatial structure.
- **C — capacity:** prefix length K ∈ {4, 8, 16} and bridge depth ∈ {1, 2, 3}.
- **D — decoding:** greedy vs. repetition-penalised vs. beam search.

## Repository Layout

```
.
├── phase3_main.ipynb     # Phase 3 notebook — final experiments & ablations (outputs included)
├── phase2_main.ipynb     # Phase 2 notebook
├── requirements.txt      # pinned dependencies
├── splits.json           # train/val/test indices (seed 42) for reproducibility
├── README.md
└── .gitignore
```

The **dataset** and **model checkpoints** are **not** committed (large files, per project
requirements). Place the dataset under `<dataset_dir>/images/`, `<dataset_dir>/masks/`, and
`<dataset_dir>/captions.csv`; the notebook regenerates everything else.

## Quickstart

1. Open `phase3_main.ipynb` in Google Colab (A100 recommended) — or local Jupyter with a CUDA GPU.
2. Put the dataset on Drive at `MyDrive/DI725/DI725_project_dataset/`
   (`images/`, `masks/`, `captions.csv`). `masks/` is required only for Ablation B.
3. Add your Weights & Biases key as a Colab **Secret** named `WANDB_API_KEY`
   (or set `CFG.use_wandb = False` to skip tracking).
4. `Runtime → Run all`. The notebook will:
   - load the full 10,000-sample dataset and build a deterministic 80 / 10 / 10 split (seed = 42),
   - train B1, M1, and the ablation variants (BLIP stays frozen; only prefix / bridge / CNN train),
   - generate captions on the test split and compute BLEU-1, BLEU-4, METEOR, ROUGE-L, BERTScore-F1,
   - run ablations A–D,
   - write all checkpoints, metrics, predictions, and figures to `MyDrive/DI725/phase3_outputs/`.

The run is **resumable**: outputs are written to Drive, and re-running skips any experiment that
already has a checkpoint / metrics, so a runtime disconnect costs at most one short experiment.

## Experiment tracking (Weights & Biases)

Training and ablation runs are logged to W&B:

**Project:** https://wandb.ai/mchamurcu-metu-middle-east-technical-university/di725-phase3

The key is read from the Colab Secret `WANDB_API_KEY` — no credentials are stored in the
notebook or this repository.

## Reproducibility

- All randomness is seeded (`CFG.seed = 42`); split indices are written to `splits.json`.
- Pinned dependency versions are in `requirements.txt`.
- Per-run metrics are saved as `metrics_<tag>.json` and aggregated into `all_results.csv`;
  per-model training curves are saved as `<tag>_history.json` (all under the Drive output dir).
- Every path is configurable via the `Config` dataclass at the top of the notebook; relative
  paths inside the dataset match the layout described in the project document.

## Phase Roadmap

- [x] **Phase 1** — Literature survey, proposal, proof of concept.
- [x] **Phase 2** — Preliminary results, benchmarking against the zero-shot baseline.
- [x] **Phase 3** — Length-aware baseline analysis, BERTScore, ablation study (segmentation
  fidelity, input modality, capacity, decoding), final results.
- [ ] **Phase 4** — Presentation.
