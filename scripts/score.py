#!/usr/bin/env python3
"""Score REFCON per-cell predictions against per-cell copy-number ground truth.

Takes the refcon_cn.npz written by scripts/infer.py and a
ground-truth array (absolute copy number per gene, one row per cell, aligned to the same cells and genes
as the input .h5ad; NaN or a negative value marks a no-call). Prints per-cell Pearson, Spearman, and
AUROC (loss / gain), averaged over evaluable cells (refcon/metrics.py).

Usage
  python scripts/score.py --pred out/sample/refcon_cn.npz --gt sample_cn.npy
"""
import argparse, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                               # repo root -> `import refcon`
from refcon.metrics import per_cell_metrics, summarize      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="refcon_cn.npz from scripts/infer.py")
    ap.add_argument("--gt", required=True,
                    help="per-cell ground truth .npy (n_cells, n_genes), aligned to the input .h5ad")
    a = ap.parse_args()

    d = np.load(a.pred, allow_pickle=True)
    cn = np.asarray(d["cn"], dtype=float)
    gt = np.asarray(np.load(a.gt, allow_pickle=True), dtype=float)
    if gt.shape != cn.shape:
        sys.exit(f"shape mismatch: predictions {cn.shape} vs ground truth {gt.shape} "
                 "(the GT must be aligned to the same cells and genes as the input .h5ad)")

    r = per_cell_metrics(cn, gt)
    for name, arr in [("Pearson", r.P), ("Spearman", r.S),
                      ("AUROC_loss", r.AL), ("AUROC_gain", r.AG)]:
        s = summarize(arr)
        print(f"{name:11s} mean={s['mean']:.4f}   (n={s['n']} evaluable cells)")


if __name__ == "__main__":
    main()
