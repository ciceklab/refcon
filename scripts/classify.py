#!/usr/bin/env python3
"""Downstream aneuploid/diploid classification on REFCON's reference-free CN predictions.

Marker-based confident-normal detection on expression -> GMM clustering of the remaining cells on the CN
predictions -> per-cluster label against the confident-normal cells: a cluster is aneuploid if its median per-cell
dispersion is >= tau1 = 1.05 times that of the confident-normal cells. A sample with fewer than 2 confident-normal
cells gets one sample-level call: aneuploid if its mean per-cell dispersion is >= T_sigma = 0.125.

Usage
  python scripts/classify.py --data sample.h5ad --cn out/sample/refcon_cn.npz
    (--cn is the output of scripts/infer.py; genes must align to the h5ad.)
Prints per-cell labels summary + info; writes labels.npy next to --cn.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import anndata as ad
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                               # repo root -> `import refcon`
from refcon.downstream.classify import call_sample, TAU1, PURE_THR   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="raw-UMI .h5ad (expression, for marker detection)")
    ap.add_argument("--cn", required=True, help="refcon_cn.npz from scripts/infer.py")
    ap.add_argument("--tau1", type=float, default=TAU1, help=f"cluster dispersion-ratio threshold (default {TAU1})")
    ap.add_argument("--t-sigma", type=float, default=PURE_THR, help=f"pure-sample threshold (default {PURE_THR})")
    a = ap.parse_args()
    A = ad.read_h5ad(a.data)
    counts = A.X.toarray() if sp.issparse(A.X) else np.asarray(A.X)
    genes = np.asarray(A.var_names).astype(str)
    d = np.load(a.cn, allow_pickle=True)
    cn, cgenes = d["cn"], d["genes"].astype(str)
    # align CN genes to the h5ad gene order the marker detector uses
    gi = {g: i for i, g in enumerate(cgenes)}
    col = np.array([gi.get(g, -1) for g in genes])
    pred = np.full((cn.shape[0], len(genes)), np.nan)
    pred[:, col >= 0] = cn[:, col[col >= 0]]
    labels, info = call_sample(counts, genes, pred, tau1=a.tau1, pure_thr=a.t_sigma)
    n_aneu = int((labels == 1).sum()); n_dip = int((labels == 0).sum())
    print(f"status={info['status']} n_confident_normal={info['n_conf']}  "
          f"aneuploid={n_aneu} diploid={n_dip}  info={info}")
    np.save(Path(a.cn).parent / "labels.npy", labels)


if __name__ == "__main__":
    main()
