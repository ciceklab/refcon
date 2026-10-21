# Reproducing the paper's per-cell numbers

This code reproduces the reference-free copy-number results on the cohorts bundled with the data: the two paired
single-cell-DNA cohorts A375 and HCT116, the scONE-seq astrocytoma, and the three cell-line panels. REFCON is the mean
of two models (context length 2,048 genes, seeds 2 and 3): the NaN-aware mean of the models' `predicted_cn_ratios` per
gene, each cell then rescaled to mean 1.

## Run
```bash
# A375 (DNTR-seq melanoma; 64 cells with per-cell GT)
python scripts/infer.py --data data/a375/a375_dntr_per_cell.h5ad --out out/a375 --device cuda:0
python scripts/score.py --pred out/a375/refcon_cn.npz --gt data/a375/a375_dntr_per_cell_cn.npy

# HCT116 (DNTR-seq colorectal; all 1,656 RNA cells are run, the 1,467 evaluable cells with paired DNA are scored)
python scripts/infer.py --data data/hct116/hct116_dntr_per_cell.h5ad --out out/hct116 --device cuda:0
python scripts/score.py --pred out/hct116/refcon_cn.npz --gt data/hct116/hct116_dntr_per_cell_cn.npy
```
`--device` accepts `cuda:N`, `mps` (Apple GPU), or `cpu` (auto-falls back to cpu on failure).
Inference and scoring are decoupled: `infer.py` writes `refcon_cn.npz`, and `score.py` evaluates it
against per-cell ground truth (`refcon/metrics.py`).

## Expected metrics (reference-free)
| cohort  | n cells | Pearson | Spearman | AUROC loss | AUROC gain |
|---------|--------:|--------:|---------:|-----------:|-----------:|
| A375    |      64 |  0.470  |  0.479   |   0.734    |   0.759    |
| HCT116  |   1,467 |  0.429  |  0.389   |   0.737    |   0.835    |
| Astrocytoma (tumor cells alone) | 390 | 0.360 | 0.345 | 0.694 | 0.657 |

With `--forward-only` (no reversed pass, about half the model time), HCT116 gives a Pearson of 0.417.

## Astrocytoma and cell-line panels: one subset at a time
The paper runs the 390 astrocytoma tumor cells alone, and each line of the three cell-line panels alone, each with the
genes that have counts in its cells. A subset is written as its own input:
```python
import os, anndata as ad, numpy as np, scipy.sparse as sp

def subset(h5ad, cn_npy, rows, out_h5ad, out_npy):
    a, gt = ad.read_h5ad(h5ad), np.load(cn_npy)
    X = sp.csc_matrix(a.X[rows])
    keep = np.asarray((X > 0).sum(0)).ravel() > 0
    b = a[rows][:, keep].copy()
    ch = b.var["chromosome"].astype(str).values
    b.var["chr_boundary"] = np.r_[True, ch[1:] != ch[:-1]].astype(np.float32)
    b.write_h5ad(out_h5ad)
    np.save(out_npy, gt[rows][:, keep])

os.makedirs("out", exist_ok=True)
a = ad.read_h5ad("data/scone/scone_astrocytoma.h5ad", backed="r")
subset("data/scone/scone_astrocytoma.h5ad", "data/scone/scone_astrocytoma_cn.npy",
       np.where(a.obs["cn_role"] == "tumor")[0], "out/scone_tumor.h5ad", "out/scone_tumor_cn.npy")
```
Then run `scripts/infer.py` and `scripts/score.py` on `out/scone_tumor.h5ad` as above. For the panels, select each
line by `obs['cell_line']`, run it, and score it with `refcon.metrics.per_line_metrics`; the mean over lines is 0.553
(esophageal, 7 lines), 0.467 (mixed, 3 lines) and 0.566 (MIX-seq, 49 lines).

Scoring runs per cell over genes with a valid ground-truth call (`gt >= 0`; the per-cell GT uses `-1`
as a no-call sentinel, which is excluded). Predictions and metrics are identical across devices (on
Apple MPS a sub-0.002 float drift can appear relative to CUDA) and do not depend on the cell batch size
(differences below 1e-5 in the ratios). Metric definitions and the ground-truth convention are in
[`input_format.md`](input_format.md).

## Runtime
`scripts/infer.py` writes `runtime.json` (device, per-model wall time, total, cells/s, peak memory). The two
models run one at a time, so peak accelerator memory is that of one model; `--cell-batch` lowers it without
changing the output. Time scales linearly with the number of cells and rises with more genes, since more genes
mean more windows per cell.

## Optional: refinement with diploid cells
When a sample contains diploid cells, they can optionally sharpen the aneuploid profiles (an
enhancement, never a requirement):
```bash
python -m refcon.refine --data <sample>.h5ad --cn out/<sample>/refcon_cn.npz
```
By default the diploid cells are the confident-normal cells REFCON detects itself; `--ref-col` and
`--ref-label` select annotated normal cells instead.

## Downstream tumor/normal classification
```bash
python scripts/classify.py --data <sample>.h5ad --cn out/<sample>/refcon_cn.npz
```
Confident-normal detection (markers on expression), then GMM clustering on the CN predictions, then
per-cluster labeling by the dispersion ratio to the confident-normal cells (aneuploid if >= 1.05),
with a pure-sample dispersion fallback at T_sigma = 0.125.
