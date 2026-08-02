# REFCON

**Reference-free copy-number inference from single-cell RNA-seq.**

REFCON is a deep-learning model that predicts a genome-wide, per-cell copy-number profile from
scRNA-seq expression alone, without any reference panel, diploid baseline, or in-sample normal cells. It estimates local copy-number deviations within short windows of neighboring genes and
stitches them into each cell's genome-wide profile. Trained once on cell lines with bulk-DNA
copy-number labels and applied without per-dataset tuning, it generalizes across cell lines, tissues,
and platforms (10x, Smart-seq2, DNTR-seq, scONE-seq, BD Rhapsody).

Outputs are relative copy-number ratios (per-cell mean = 1), not absolute copy number or ploidy. The
reported model is a 3-checkpoint ensemble (`ens3`) over context lengths 512, 1024, and 2048.

## Install
```bash
conda create -n refcon python=3.12 && conda activate refcon
pip install -e .                      # installs REFCON + dependencies (pyproject.toml)
python scripts/fetch_data.py          # pulls data + checkpoints from Zenodo (see data/README.md)
```
Tested with PyTorch 2.10 on CUDA 12.8 and on Apple Silicon (MPS).

[`quickstart.ipynb`](quickstart.ipynb) runs the whole pipeline end to end: install,
fetch, reference-free inference on one cohort, and per-cell copy-number visualization (GPU/MPS/CPU).

## Inference
```bash
python scripts/infer.py --data your_sample.h5ad --out out/your_sample --device cuda:0
# -> out/your_sample/ens3_cn.npz  (cn ratios, cells, genes) + runtime.json
```
The input is a raw-UMI `.h5ad` with genomic-ordered genes and a `chr_boundary` column; see
[`docs/input_format.md`](docs/input_format.md). `--device` accepts `cuda:N`, `mps`, or `cpu`.

To score predictions against per-cell copy-number ground truth:
```bash
python scripts/score.py --pred out/your_sample/ens3_cn.npz --gt your_sample_cn.npy
```
The bundled per-cell-DNA cohorts and their expected metrics are in
[`docs/reproduce.md`](docs/reproduce.md).

## Downstream tumor/normal classification
```bash
python scripts/classify.py --data your_sample.h5ad --cn out/your_sample/ens3_cn.npz
```
Confident-diploid detection (marker enrichment on expression), then GMM clustering on the CN
predictions, then per-cluster Pearson labeling (T=0.75), with a pure-sample dispersion fallback
(T_sigma=0.135). The confident-diploid cells only label clusters; they are not used to compute the
copy-number profiles.

## Training
```bash
python scripts/train.py --train-data /path/to/train_cohorts --val-data /path/to/val_cohorts \
    --device cuda:0 --out-dir checkpoints/
```
Trains the three ensemble members (C = 512/1024/2048) with the paper hyperparameters (AdamW lr 1e-4,
wd 0.01, batch 64, grad-clip 1.0, ReduceLROnPlateau, early-stop 30, seed 42, bin_size 8, SmoothL1 +
variance floor). `--train-data` and `--val-data` each point at a directory (or single file) of
processed `.h5ad` cohorts; see [`docs/input_format.md`](docs/input_format.md). Training data is not
bundled (see Data availability in the paper).

## Repository layout
```
refcon/
  refcon/                    importable package
    model.py                 RoPE Transformer backbone (MaskedConvStemTransformer)
    model_binned_dev.py      BinnedDevModel: masked InstanceNorm -> Conv binner -> Transformer -> ratio head
    eval_cellwide.py         single-checkpoint cell-wide inference (windows + bridges + joint LSQ stitch)
    data.py                  dataset loading (chromosome-chunked)
    train_binned_dev.py      training loop for one ensemble member
    refine.py                optional reference-cell SVD refinement
    metrics.py               per-cell / per-line CN metrics
    downstream/              confident-diploid markers + tumor/normal classifier
  scripts/
    infer.py                 ens3 inference -> ens3_cn.npz + runtime.json
    score.py                 score predictions vs per-cell CN ground truth
    classify.py              downstream tumor/normal classification
    train.py                 train the three ensemble members
    fetch_data.py            fetch data + checkpoints from Zenodo
  data/                      evaluation cohorts (from Zenodo; one subdirectory per cohort)
  checkpoints/               ensemble_bs8_mc512.pt, baseline_bs8_mc1024.pt, ensemble_bs8_mc2048.pt
  docs/                      input_format.md, reproduce.md
```

## Data & citation
Processed cohorts and checkpoints are on Zenodo (`scripts/fetch_data.py`); primary data accessions are
listed in the paper's Data availability. If you use REFCON, please cite the paper (see the manuscript
for the current reference).
