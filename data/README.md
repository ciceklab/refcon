# REFCON data

The data objects are hosted on **Zenodo** (https://doi.org/10.5281/zenodo.23534511; too large for git) and fetched by
`python scripts/fetch_data.py`. This directory holds the held-out evaluation cohorts used in the
paper's reference-free copy-number benchmark. Every h5ad has **raw UMI counts** in `X`
and genomic annotations in `var` (`chromosome`, `start`, `end`, `gene_id`, `chr_boundary`); genes are ordered by
genomic position. REFCON is reference-free, so no normal/reference cells are required at inference.

Every cohort keeps the genes of its own source data (GRCh38, GENCODE 46): mitochondrial and sex-chromosome genes are
removed, and genes without counts in the cohort's cells are dropped. Nothing else is filtered; the cells are those of
the paper's analyses.

The cohorts fall into two ground-truth tiers.

## Tier 1  -  paired single-cell DNA (per-cell scWGS ground truth)
Each cell is scored against its **own** single-cell whole-genome DNA profile.

### `a375/`  -  A375 melanoma (DNTR-seq; Zachariadis et al. 2020, GSE144296)
- `a375_dntr_per_cell.h5ad`  -  96 cells x 14,894 genes, raw UMI.
- `a375_dntr_per_cell_cn.npy`  -  `(96, 14894)` AneuFinder per-cell absolute CN, row- and column-aligned to the
  h5ad. 64 cells evaluable; 32 (the HCA00101 plate) are all-NaN and excluded from scoring.

### `hct116/`  -  HCT116 colorectal (DNTR-seq; GSE144296)
- `hct116_dntr_per_cell.h5ad`  -  1,656 cells (all RNA cells, as the paper runs them) x 18,571 genes, raw UMI.
- `hct116_dntr_per_cell_cn.npy`  -  `(1656, 18571)` per-cell absolute CN, aligned. 1,468 cells have paired DNA
  (1,467 evaluable); the other 188 are all-NaN and excluded from scoring.

### `scone/`  -  scONE-seq astrocytoma (Yu et al. 2023, GSE185269)
- `scone_astrocytoma.h5ad`  -  840 cells x 29,978 genes, raw UMI (the deposited counts rounded to integers).
  `obs['cn_role']` splits the cells into **390 tumor** and **450 reference** (non-malignant); `obs['Clone_type']`
  gives the scWGS clone.
- `scone_astrocytoma_cn.npy`  -  `(840, 29978)` per-cell absolute CN (Yu et al.'s deposited normalized read depth x 2,
  mapped to genes), aligned to the h5ad. For the paper's reference-free benchmark, run the 390 tumor cells alone
  (their cells, with the genes that have counts in them; see [`../docs/reproduce.md`](../docs/reproduce.md)); the 450
  reference cells serve the optional refinement.

Tier-1 ground truth was called from the paired single-cell DNA reads with BWA-MEM + SAMtools +
AneuFinder (1 Mb bins, e-divisive segmentation) and projected to gene level (A375/HCT116);
scONE-seq provides paired scWGS from the source study.

## Tier 2  -  bulk-CN cell-line panels (per-line DepMap ground truth)
Panels of distinct cell lines; each cell is scored against the **bulk** DepMap copy number of its line
(`obs['cell_line']`). Each file holds all lines of a panel, on the union of the genes expressed in any line. The paper
runs **each line alone**: take a line's cells and keep the genes with counts in them, then run and score that line.

### `ccle_esophageal/`  -  7 CCLE esophageal lines (Kinker et al. 2020, GSE157220 / SCP542)
- `ccle_esophageal.h5ad`  -  2,543 cells x 16,390 genes, raw UMI, `obs['cell_line']`.
- `ccle_esophageal_cn.npy`  -  `(2543, 16390)` bulk DepMap absolute CN (24Q4), broadcast to each line's cells.

### `ccle_mixed/`  -  3 CCLE mixed-tissue lines (CL34, NCIH460, SKMEL5; GSE157220 / SCP542)
- `ccle_mixed.h5ad`  -  1,512 cells x 15,448 genes, raw UMI, `obs['cell_line']`.
- `ccle_mixed_cn.npy`  -  `(1512, 15448)` bulk DepMap absolute CN (24Q4), broadcast per line.

### `mixseq/`  -  MIX-seq pan-cancer panel (McFarland et al. 2020, figshare 10.6084/m9.figshare.10298696)
- `mixseq_pancancer.h5ad`  -  3,055 cells x 23,404 genes across **49 pan-cancer lines** (the DepMap-CN-covered
  subset of the frozen 73-line MIX-seq experiment), raw UMI, `obs['cell_line']`.
- `mixseq_pancancer_cn.npy`  -  `(3055, 23404)` bulk DepMap absolute CN (24Q4), broadcast per line (NaN where a gene
  is absent from DepMap).

## Scoring
`refcon.metrics.per_cell_metrics`  -  per-cell Pearson/Spearman and AUROC (loss `CN < 1.5`,
gain `CN > 2.5`) vs the absolute-CN profiles; correlation is scale-invariant, so REFCON's mean-1
ratios score directly. `per_line_metrics` aggregates the same over `obs['cell_line']` for Tier 2.

## Training data (not bundled)
The training cohorts are **not** included here; see the paper's Data availability for their
primary accessions. `scripts/train.py` reads them from `--train-data` and `--val-data` (format:
[`../docs/input_format.md`](../docs/input_format.md)).

### Healthy cells
The training corpus includes **4,200 healthy cells** as one more cohort, labelled diploid (copy number 2 at every gene:
`var["cn_ground_truth"] = 2`, `uns["cn_type"] = "bulk"`): 300 cells from each of 14 organs of Tabula Sapiens v2 (10x,
at least 200 detected genes), from the donors TSP1, TSP6, TSP14, TSP17, TSP20 and TSP30, with raw counts on the gene
axis of the CCLE training panel (7,351 genes; genes missing from Tabula Sapiens are 0). Donors held out for evaluation
(TSP2, 4, 8, 9, 12, 19, 21, 25, 26, 27, 28) are never used, and TSP7 and TSP10 are held out for validation (1,200
cells), so no donor is shared between training, validation and evaluation. Heart, Kidney, Mammary and Uterus
contribute no cells: all their 10x donors are held out. The cells come from these Tabula Sapiens v2 organ files of
CELLxGENE (`https://datasets.cellxgene.cziscience.com/<version>.h5ad`):

| organ | CELLxGENE dataset id | version (file) | size (GB) |
|---|---|---|---|
| Bladder | `e5c63d94-593c-4338-a489-e1048599e751` | `8ffc335d-36fa-461e-be47-cd6503833420` | 3.32 |
| Fat | `5e5e7a2f-8f1c-42ac-90dc-b4f80f38e84c` | `8115812f-9acf-4f62-b15f-bde18086a1c8` | 3.69 |
| Heart | `e6a11140-2545-46bc-929e-da243eed2cae` | `82197577-7914-45c0-9fde-6597a1da920f` | 1.16 |
| Kidney | `2423ce2c-3149-4cca-a2ff-cf682ea29b5f` | `76b6cf23-56db-4eac-8ee8-d84ce64c0395` | 0.45 |
| Large_Intestine | `7357cee7-9f7f-4ab0-8cec-90de8f047e38` | `81781bd2-7a71-43a8-ab31-2a9009c1aea6` | 1.02 |
| Liver | `6d41668c-168c-4500-b06a-4674ccf3e19d` | `a4b3a49e-062b-4e3f-8915-02f40607651f` | 0.83 |
| Lung | `0d2ee4ac-05ee-40b2-afb6-ebb584caa867` | `fddf43a3-b7c8-49e9-ac0a-5e24b390849f` | 3.20 |
| Lymph_Node | `18eb630b-a754-4111-8cd4-c24ec80aa5ec` | `6dd28080-1d36-4905-8b44-a9954a4438a2` | 4.09 |
| Mammary | `2ba40233-8576-4dec-a5f1-2adfa115e2dc` | `33b25f68-a9d6-4643-9283-653e937ae608` | 1.33 |
| Muscle | `1c9eb291-6d31-47e1-96b2-129b5e1ae64f` | `4919a450-62fb-4741-b750-5591443edee8` | 2.05 |
| Ovary | `bb78adbb-1428-4468-bad0-63602bb974e9` | `6ad78e84-6715-458d-95b4-69e8509cd785` | 1.82 |
| Pancreas | `ff45e623-7f5f-46e3-b47d-56be0341f66b` | `ba587dee-e050-4a31-bcb8-05e5418f9086` | 0.74 |
| Prostate | `d77ec7d6-ef2e-49d6-9e79-05b7f8881484` | `4a289bf1-97c2-4616-8e1c-585088d28c3e` | 0.90 |
| Skin | `0041b9c3-6a49-4bf7-8514-9bc7190067a7` | `c240d5bc-a521-4330-a76b-7d566e939d87` | 0.51 |
| Small_Intestine | `a357414d-2042-4eb5-95f0-c58604a18bdd` | `6db4dfe6-d421-4287-a9dc-22eaa59be966` | 1.81 |
| Stomach | `9ba03780-4b13-44bc-a7d3-ce532ea0a856` | `3e8bd2b2-fe53-4b95-abb2-ce971b4f1c14` | 1.07 |
| Tongue | `55cf0ea3-9d2b-4294-871e-bb4b49a79fc7` | `07cf9c5f-6e79-4422-a4de-4427f0f4a76d` | 2.01 |
| Uterus | `6ec405bb-4727-4c6d-ab4e-01fe489af7ea` | `ec18d0d1-3b1b-4413-b011-0078c8d69bd3` | 0.90 |
