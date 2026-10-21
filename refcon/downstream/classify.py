"""Marker-anchored aneuploid/diploid caller (one sample at a time).

    from refcon.downstream.classify import call_sample
    labels, info = call_sample(raw_counts, gene_names, pred)
        labels : (n_cells,) int  -> 0 diploid, 1 aneuploid (a pure sample gets one label for all cells)
        info   : dict(n_conf, status, K, per-cluster dispersion ratios, thresholds)

Steps: confident-normal cells from markers on expression -> pure-sample gate -> the remaining cells clustered on the
CN predictions (sub1) -> each cluster labelled against the confident-normal cells.

Cluster labelling: per-cell dispersion d = standard deviation of the cell's predicted ratios over genes; a cluster is
aneuploid if median(d in the cluster) / median(d of the confident-normal cells) >= tau1 (1.05).
Pure-sample gate: with fewer than 2 confident-normal cells the whole sample gets one label, aneuploid if the mean
per-cell dispersion >= T_sigma (0.125).

Both thresholds were set on development data, separate from all evaluation data. Predictions are imputed (per-gene
mean) before clustering and statistics.
"""
import numpy as np, warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from .common import impute
from .markers import ssmww_confident_normal

TAU1 = 1.05       # cluster dispersion ratio at or above which a cluster is aneuploid
PURE_THR = 0.125  # T_sigma: mean per-cell dispersion at or above which a pure sample is aneuploid
MIN_CONF = 2      # samples with fewer confident-normal cells get the pure-sample call (a one-cell reference is too
                  # noisy to compare clusters against)


def sub1(X, n_pcs=5):
    """1-level GMM/BIC clustering: PCA(5) -> StandardScale -> GMM(full) k=1..8 argmin BIC."""
    Xi = impute(X)
    npc = min(n_pcs, Xi.shape[1], max(2, Xi.shape[0]-1))
    pcs = PCA(max(npc, 2), svd_solver='full').fit_transform(Xi)
    Xc = StandardScaler().fit_transform(pcs[:, :npc])
    bics, gms = [], []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for k in range(1, min(8, Xc.shape[0]-1)+1):
            try:
                gm = GaussianMixture(k, covariance_type='full', n_init=5, random_state=0).fit(Xc)
                bics.append(gm.bic(Xc)); gms.append(gm)
            except Exception:
                bics.append(np.inf); gms.append(None)
    g = gms[int(np.argmin(bics))]
    return g.predict(Xc) if g is not None else np.zeros(Xi.shape[0], int)


def call_sample(raw_counts, gene_names, pred, tau1=TAU1, pure_thr=PURE_THR):
    """Label every cell of one sample 0 (diploid) or 1 (aneuploid). `tau1` and `pure_thr` override the thresholds."""
    tau1, pure_thr = float(tau1), float(pure_thr)
    cn = ssmww_confident_normal(raw_counts, gene_names)['norm_mask'].astype(bool)
    nconf = int(cn.sum())
    p = impute(np.asarray(pred, float))
    disp = np.nanstd(p, axis=1)                       # per-cell dispersion over genes
    if nconf < MIN_CONF:
        # pure sample (no diploid anchor): one sample-level call from the mean per-cell dispersion
        score = float(np.mean(disp))
        call = 1 if score >= pure_thr else 0
        return (np.full(len(p), call, int),
                dict(n_conf=nconf, status='pure_sample', aneuploidy_score=round(score, 4), pure_thr=pure_thr))
    out = np.zeros(len(p), int)                       # confident-normal cells -> 0 (diploid)
    rest = ~cn; ridx = np.where(rest)[0]
    lab = sub1(p[rest])
    d_ref = float(np.median(disp[cn]))                # dispersion of the confident-normal cells
    stats = {}
    for c in range(lab.max()+1):
        s = lab == c
        ratio = float(np.median(disp[rest][s]) / d_ref)
        stats[int(c)] = ratio
        out[ridx[s]] = 1 if ratio >= tau1 else 0      # >= tau1 aneuploid, else diploid
    return out, dict(n_conf=nconf, status='labeled', K=int(lab.max()+1),
                     cluster_dispersion_ratios=stats, tau1=tau1)
