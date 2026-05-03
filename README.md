# DI 725 Term Project — Multi-modal Transformers

**Author:** Melih Can Hamurcu  
**Course:** DI 725 — Transformers and Attention-Based Deep Networks  
**Institution:** Middle East Technical University, Graduate School of Informatics

## Project Overview

This project investigates whether **segmentation-derived semantic information** can improve
remote-sensing image captioning when injected into a frozen vision-language model via
**learned prefix embeddings** (segmentation-conditioned prompt learning).

### Phase 2 — current

We compare three configurations on the same test split:

| Tag | Description |
|-----|-------------|
| **B0** | Zero-shot pretrained BLIP (Phase 1 baseline) |
| **B1** | BLIP + unconditional learnable prefix (control) |
| **M1** | BLIP + segmentation-conditioned prefix (proposed) |

The base BLIP weights are frozen in B1 and M1; only the prefix parameters (B1) or the
seg-to-prefix bridge network (M1) receive gradients.

## Repository Layout

```
.
├── phase2_main.ipynb    # main notebook (data, training, evaluation, results)
├── requirements.txt     # pinned dependencies
├── README.md
└── .gitignore
```

The dataset is **not** included in this repository (per project requirements). Place it
under `<base_dir>/images`, `<base_dir>/masks`, and `<base_dir>/captions.csv`.

## Quickstart

1. **Clone** the repo and open `phase2_main.ipynb` in Google Colab (recommended) or a
   local Jupyter with a CUDA GPU.
2. **Mount Google Drive** (or set `CFG.base_dir` to your local path).
3. **Install dependencies** — first cell runs `pip install -q -r requirements.txt`.
4. **Run all cells.** The notebook will:
   - load the full 10 000-sample dataset,
   - build a deterministic 80 / 10 / 10 train / val / test split (seed = 42),
   - train B1 and M1 (~30–60 min total on a Colab Pro GPU),
   - generate captions on the test split for B0, B1, M1,
   - compute BLEU-1, BLEU-4, METEOR, ROUGE-L,
   - save predictions, metrics, and figures to `CFG.output_dir`.

### Optional: WandB tracking

To log training to Weights & Biases:

```bash
pip install wandb
wandb login
```

Then set `CFG.use_wandb = True` in the configuration cell.

## Reproducibility

- All randomness is seeded (`CFG.seed = 42`); split indices are written to
  `splits.json` alongside the run outputs.
- Pinned dependency versions are in `requirements.txt`.
- The full configuration used for a run is saved in `run_summary.json`.
- The dataset path is configurable via `CFG.base_dir`; relative paths inside the dataset
  match the layout described in the project document.

## Phase Roadmap

- [x] **Phase 1** — Literature survey, proposal, proof of concept.
- [x] **Phase 2** — Preliminary results, benchmarking against zero-shot baseline (this submission).
- [ ] **Phase 3** — Ablation study (prefix length, bridge depth, mask-image input variant), final results.
- [ ] **Phase 4** — Presentation.
