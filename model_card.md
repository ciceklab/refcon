# Model card: REFCON

REFCON predicts a per-cell, per-gene copy number profile from single-cell RNA-seq expression alone, without a
reference of normal cells. This card summarizes the released model. The paper gives the full description.

## Model details

| | |
|---|---|
| Developers | M. Mert Gençtürk and A. Ercüment Çiçek, Department of Computer Engineering, Bilkent University |
| Released model | Average of two models trained with different random seeds (2 and 3) |
| Checkpoints | `refcon_v3_s2.pt`, `refcon_v3_s3.pt` (3.9 MB each), at https://doi.org/10.5281/zenodo.23534511 |
| Architecture | log1p and per-cell masked InstanceNorm, 1D convolutional binner (8 genes per bin), two-layer convolutional block, 4-layer Transformer (dimension 128, 4 heads) with rotary positions in megabases, linear ratio head |
| Size | 975,299 parameters per model |
| Context | Windows of up to 2,048 genes in genomic order |
| Pretraining | None. Trained from scratch |
| License | Code: MIT. Checkpoints and processed data: CC BY 4.0 |
| Contact | mert.gencturk@bilkent.edu.tr, cicek@cs.bilkent.edu.tr |

The model reads only the order of the genes and the genomic distances between them. It uses no gene names and no gene
list, so one trained model applies to any gene panel and platform.

## Intended use

- Inferring relative copy number per cell and per gene from human scRNA-seq count matrices (GRCh38), including samples
  without normal cells such as cell lines and organoids.
- Labeling cells as aneuploid or diploid with the bundled classification script.
- Optional refinement of the profiles against normal cells of the same sample.

Input: raw UMI counts in an `.h5ad` file with genes sorted by genomic position (see `docs/input_format.md`).
Output: one copy number ratio per cell and gene, with each cell normalized to a mean of 1.

## Out of scope

- **Absolute copy number or ploidy.** The output is relative. A whole-genome doubling that scales every chromosome alike
  is not detected.
- **Sex chromosomes and the mitochondrial genome.** The model was trained and benchmarked on autosomal genes only.
- **Events of only a few genes.** Detection of such events is at chance level.
- **Other species or genome assemblies.** The model was trained and tested on human data with GRCh38 coordinates.
- **Clinical or diagnostic decisions.** REFCON is a research tool. A copy number label alone does not establish that a
  cell is malignant.

## Training data

64,379 cells from six public datasets:

- 60,179 cells of 140 cancer cell lines from five scRNA-seq datasets (MCF7, COLO320, MDA-MB-231, three gastric lines
  and 135 lines of the CCLE pan-cancer panel), each line labeled with its bulk DNA copy number from DepMap 24Q4.
- 4,200 healthy Tabula Sapiens cells (300 from each of 14 organs, six donors), labeled diploid.

Validation used seven CCLE cell lines absent from training (2,255 cells) and 1,200 Tabula Sapiens cells from two
further donors. No cell line, patient or donor is shared between training, validation and test data. Accessions are in
the paper's Data availability statement and in `data/README.md`.

## Training procedure

Huber loss on per-bin copy number ratios. AdamW (learning rate 1e-4, weight decay 0.01), batch size 64, gradient
clipping at 1.0, cosine decay of the learning rate to 1e-6 over 150 epochs, no early stopping. Half of the training
windows span the boundary of two adjacent chromosomes and half are presented in reversed gene order. Counts are randomly
thinned, and genes are randomly removed or set to zero, to mimic shallower sequencing and smaller gene panels. One
training run takes about 6 hours on one NVIDIA GeForce RTX 2080 Ti. The command and recipe are in the README.

## Evaluation

The model was evaluated on more than 410,000 held-out cells from cell lines, organoids and tumors, profiled with seven
sequencing technologies, none of them seen in training. Loss is a true copy number below 1.5 and gain above 2.5.

Per-cell agreement with paired single-cell DNA of the same cells, without a reference:

| cohort | cells | Pearson | Spearman | AUROC loss | AUROC gain |
|---|---:|---:|---:|---:|---:|
| A375 (DNTR-seq) | 64 | 0.470 | 0.479 | 0.734 | 0.759 |
| HCT116 (DNTR-seq) | 1,467 | 0.429 | 0.389 | 0.737 | 0.835 |
| Astrocytoma (scONE-seq), tumor cells alone | 390 | 0.360 | 0.345 | 0.694 | 0.657 |

Agreement with bulk DNA copy number on held-out cell lines (mean per cell line Pearson correlation): 0.553 on
7 esophageal lines, 0.467 on 3 mixed-tissue lines and 0.566 on 49 MIX-seq lines.

These numbers are reproduced by the commands in `docs/reproduce.md`. The paper reports the comparison with other
methods, the classification benchmark on tumors and healthy tissue, the organoid analyses and the ablations.

## Limitations and biases

- Gene expression is an indirect readout of copy number. Per-cell correlation with paired single-cell DNA is modest
  (0.36 to 0.47).
- Detection improves with event size. Arm- and chromosome-scale alterations are detected with AUROC 0.75 to 0.81.
  Shorter events are detected less well.
- The training labels are bulk copy number of cell lines, in which gains are far more common than losses. Training
  samples loss-bearing chromosomes twice as often to offset this.
- The aneuploid versus diploid labels are assigned to whole clusters. A tumor population too small or too weakly altered
  to form its own cluster is labeled diploid.
- Labeling relies on confident-normal cells found from gene expression. A sample with fewer than two such cells receives
  a single sample-level call.
- Specificity on healthy tissue varied by organ, from 0.42 to 1.00.

## Compute

Inference on HCT116 (about 17,700 genes) runs at about 17 cells per second on an NVIDIA GeForce RTX 2080 Ti, 6 cells per
second on an Apple M4 Pro GPU and 0.9 cells per second on a CPU, with 1 to 2.5 GB of memory. Runtime grows linearly
with the number of cells.

## Citation

Please cite the REFCON paper (see the README for the current reference) and the Zenodo record
(https://doi.org/10.5281/zenodo.23534511).
