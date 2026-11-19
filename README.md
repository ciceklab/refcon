# REFCON

**Reference-free copy-number inference from single-cell RNA-seq.**

REFCON is a deep-learning model that predicts a genome-wide, per-cell copy-number profile from
scRNA-seq expression alone, without any reference panel, diploid baseline, or in-sample normal cells. It estimates
local copy-number deviations within short windows of neighboring genes and stitches them into each cell's
genome-wide profile. Trained once on cell lines with bulk-DNA copy-number labels and on healthy cells labelled
diploid, and applied without per-dataset tuning, it generalizes across cell lines, tissues, and platforms (10x,
Smart-seq2, DNTR-seq, scONE-seq, BD Rhapsody).

Outputs are relative copy-number ratios (per-cell mean = 1), not absolute copy number or ploidy. REFCON averages two
models with a context of 2,048 genes per window, trained with different seeds.

## Install
```bash
conda create -n refcon python=3.12 && conda activate refcon
pip install -e .                      # installs REFCON + dependencies (pyproject.toml)
python scripts/fetch_data.py          # pulls data + checkpoints from Zenodo (see data/README.md)
```

[`quickstart.ipynb`](quickstart.ipynb) runs the whole pipeline end to end: install,
fetch, reference-free inference on one cohort, and per-cell copy-number visualization (GPU/MPS/CPU).

## System requirements
REFCON is pure Python (3.10 or newer; developed with 3.12). It runs on Linux and macOS and needs no non-standard
hardware: a GPU (NVIDIA CUDA or Apple Silicon) is optional and a laptop CPU is enough for the demo. Dependencies and
the versions used for the paper are in `requirements.txt`. Tested on Ubuntu 22.04.5 LTS with PyTorch 2.10
(CUDA 12.8, NVIDIA GeForce RTX 2080 Ti) and on macOS 15.7 (Apple M4 Pro) with PyTorch 2.10 and 2.15.

Installation takes about one minute on a desktop computer, and `scripts/fetch_data.py` downloads 166 MB (under a minute).

## Demo
The A375 cohort (96 cells) is the demo; commands and expected metrics are in [`docs/reproduce.md`](docs/reproduce.md).
Inference takes about 20 seconds on an Apple M4 Pro GPU and about 1.5 minutes on its CPU, and scoring prints Pearson
0.470, Spearman 0.479, AUROC loss 0.734 and AUROC gain 0.759.

## Inference
```bash
python scripts/infer.py --data your_sample.h5ad --out out/your_sample --device cuda:0
# -> out/your_sample/refcon_cn.npz  (cn ratios, spread between the two models, cells, genes) + runtime.json
# --cell-batch: cells per forward call (default 512; the output does not depend on it)
# --forward-only: skip the reversed pass (half the model passes, slightly lower accuracy)
```
The input is a raw-UMI `.h5ad` with genomic-ordered genes, a `chr_boundary` column, and each gene's
`chromosome` and `start` in GRCh38 (every window carries its genes' positions in megabases); see
[`docs/input_format.md`](docs/input_format.md). `--device` accepts `cuda:N`, `mps`, or `cpu`.

To score predictions against per-cell copy-number ground truth:
```bash
python scripts/score.py --pred out/your_sample/refcon_cn.npz --gt your_sample_cn.npy
```
The bundled per-cell-DNA cohorts and their expected metrics are in
[`docs/reproduce.md`](docs/reproduce.md).

## Downstream tumor/normal classification
```bash
python scripts/classify.py --data your_sample.h5ad --cn out/your_sample/refcon_cn.npz
```
Confident-normal detection (marker enrichment on expression), then GMM clustering on the CN
predictions, then per-cluster labeling: a cluster is aneuploid if the median per-cell dispersion of
its cells is at least tau1 = 1.05 times that of the confident-normal cells, with a pure-sample
dispersion fallback (T_sigma = 0.125). The confident-normal cells only label clusters; they are not used to compute
the copy-number profiles.

## Optional refinement with diploid cells
```bash
python -m refcon.refine --data your_sample.h5ad --cn out/your_sample/refcon_cn.npz
# -> out/your_sample/refcon_cn_refined.npz; diploid cells = the confident-normal cells REFCON detects itself
# --ref-col <obs column> --ref-label <value>: refine against annotated normal cells instead; --svd-k (default 3)
```
Refinement removes the model's systematic bias using the sample's diploid cells (per-gene bias, top-k SVD modes,
per-cell renormalization). It runs after classification and does not change any label.

## Training
```bash
python scripts/train.py --train-data /path/to/train_cohorts --val-data /path/to/val_cohorts \
    --device cuda:0 --out-dir checkpoints/
```
Trains three models at context length 2048 (seeds 1, 2, 3) and keeps the two with the higher
validation Pearson as `refcon_v3_s<seed>.pt`. Recipe:
- training data: every cohort of `--train-data`, the cell lines and 4,200 Tabula Sapiens v2 healthy cells
  labelled diploid (see [`data/README.md`](data/README.md));
- windows: half are bridge windows across the boundary of two adjacent chromosomes (each chromosome with
  its own start offset, so the boundary stays on a bin edge), half are reversed, and every window starts
  0 to 7 genes later;
- robustness: thinning of the counts (half of the windows, rate log-uniform in [0.1, 1]), gene removal
  (a quarter, up to 50% of the genes) and gene zeroing (a quarter, up to 30%); targets never change;
- model: rotary positions in megabases and chromosome separation;
- optimisation: SmoothL1 (Huber) loss, AdamW lr 1e-4, wd 0.01, batch 64, grad-clip 1.0, bin_size 8; a
  fixed cosine decay of the learning rate to 1e-6 over 150 epochs;
- checkpoint: every epoch the complete inference pipeline is scored on monitoring cells of the
  validation lines; snapshots every 10 epochs; a run's weights are the snapshot at the peak of the
  10-epoch moving average of that score.

`--exclude-lines` removes cell lines from the training data (matched on `obs` `cell_line`, `Cell_line` or
`cn_profile`). `--train-data` and `--val-data` each point at a directory (or single file) of processed `.h5ad`
cohorts; see [`docs/input_format.md`](docs/input_format.md). Training data is not bundled (see Data availability
in the paper).

## Repository layout
```
refcon/
  refcon/                    importable package
    model.py                 rotary-position Transformer backbone (MaskedConvStemTransformer)
    model_binned_dev.py      BinnedDevModel: masked InstanceNorm -> Conv binner -> Transformer -> ratio head
    eval_cellwide.py         single-model cell-wide inference (windows + bridges + joint LSQ stitch)
    data.py                  dataset loading (chromosome-chunked)
    train_binned_dev.py      training loop for one model
    refine.py                optional reference-cell SVD refinement (python -m refcon.refine)
    metrics.py               per-cell / per-line CN metrics
    downstream/              confident-normal markers + tumor/normal classifier
  scripts/
    infer.py                 inference with both models -> refcon_cn.npz + runtime.json
    score.py                 score predictions vs per-cell CN ground truth
    classify.py              downstream tumor/normal classification
    train.py                 train the models
    fetch_data.py            fetch data + checkpoints from Zenodo
  data/                      evaluation cohorts (from Zenodo; one subdirectory per cohort)
  checkpoints/               refcon_v3_s2.pt, refcon_v3_s3.pt (from Zenodo)
  docs/                      input_format.md, reproduce.md
```

## Data & citation
Processed cohorts and checkpoints are on Zenodo (https://doi.org/10.5281/zenodo.23534511), fetched by
`scripts/fetch_data.py`; primary data accessions are listed in the paper's Data availability. If you use REFCON,
please cite the paper (see the manuscript for the current reference) and the Zenodo record.
