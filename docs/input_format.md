# Expected input format

REFCON reads a single **AnnData `.h5ad`** per sample.

## Expression matrix `adata.X`
- **Raw UMI counts** (integers), shape `(n_cells, n_genes)`. Do **not** pre-normalize  -  a per-cell
  masked InstanceNorm inside the model removes library-size and platform scale.
- Dense or sparse both work.
- Smart-seq2 / plate data that only ship TPM-derived values are accepted as-is (the model never
  computes TPM); pass whatever expression you have in `X`.

## Gene metadata `adata.var` (required topological features)
Genes must be **sorted by genomic position within each chromosome**, and `var` must carry:

| column        | meaning                                                            |
|---------------|-------------------------------------------------------------------|
| `chromosome`  | chromosome label per gene (`chr1` … `chr22`, `chrX`, `chrY`, or without the `chr` prefix) |
| `start`,`end` | gene genomic coordinates (bp, **GRCh38**)                          |
| `chr_boundary`| `1` for the first gene of each chromosome, else `0`              |

REFCON reads **`chromosome` and `start`** to place every gene: each window carries its genes' positions in
megabases along GRCh38 (`refcon.model_binned_dev.gene_positions_mb`; chromosomes 1-22, X and Y laid end to end, with the
UCSC GRCh38 chromosome lengths). Coordinates of another assembly change these positions slightly; lift them over to
GRCh38 first.

REFCON uses only these positions and chromosome boundaries, never gene names or gene-set memberships.
A single trained model therefore applies across gene panels and platforms (10x, Smart-seq2, DNTR-seq,
scONE-seq, BD Rhapsody) without retraining.

Example (`data/a375/a375_dntr_per_cell.h5ad`):
```
adata.X          int32 raw UMI, (96, 14894)
adata.var cols   chromosome, start, end, gene_id, chr_boundary
```

## Output
`scripts/infer.py` writes `refcon_cn.npz` (the mean of the two REFCON models) with:
- `cn`     -  `(n_cells, n_genes)` per-gene **copy-number ratio**, renormalized so each cell's mean over
  valid genes is **1** (relative dosage, not absolute copy number / ploidy).
- `spread` -  `(n_cells, n_genes)` standard deviation of the two models' predictions per gene.
- `cells`  -  cell ids (= `adata.obs_names`)
- `genes`  -  gene symbols (= `adata.var_names`)

## Ground-truth convention (for scoring)
Per-cell **absolute** copy number, one value per gene. A gene/cell with no reliable call is marked
`NaN` **or** a negative sentinel (`-1`) and is excluded from scoring (valid = finite and `>= 0`)  - 
including the `-1` no-calls as if they were copy number would under-report Pearson. Metrics:
per-cell Pearson/Spearman, and AUROC with **loss = GT < 1.5**, **gain = GT > 2.5**. Correlation is
scale-invariant, so REFCON's mean-1 ratio scores directly against absolute-CN ground truth.

## Training data
Training (`scripts/train.py`) reads every `.h5ad` of `--train-data` and `--val-data`, each in the format above
plus its copy-number labels: `uns["cn_type"]` (`bulk`, `per_line` or `per_cell`), and `var["cn_ground_truth"]`
(bulk), `uns["cn_profiles"]` with `obs["cn_profile"]` (one profile per cell line), or `uns["cn_matrix"]` with
`uns["cn_cell_ids"]` (per cell). The healthy cells are one more training cohort with copy number 2 at every gene
(`var["cn_ground_truth"] = 2`, `uns["cn_type"] = "bulk"`). See [`../data/README.md`](../data/README.md).
