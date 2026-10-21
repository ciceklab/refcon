#!/usr/bin/env python3
"""Train one REFCON model (Conv-binned per-bin ratio model).

Usage (invoked per model by scripts/train.py):
    python -m refcon.train_binned_dev --train-data <dir> --val-data <dir> --device cuda:0 [--seed N] [--out-dir DIR]

Recipe:
  - training data: every cohort of --train-data (the cell lines and the healthy cells labelled diploid), one dataset
    after the other in a round-robin;
  - windows of up to 2,048 genes: bridge windows across a chromosome boundary (probability 0.5; each chromosome with
    its own start offset, so the boundary stays on a bin edge), reversed windows (0.5), a start offset of 0 to
    bin_size-1 genes; chromosomes carrying a loss drawn twice as often;
  - count perturbations that never change the targets: thinning (probability 0.5, rate log-uniform in [0.1, 1]), gene
    removal (0.25, a share U(0, 0.5) of the window's genes), gene zeroing (0.25, U(0, 0.3));
  - megabase rotary positions and chromosome separation (gap of 1,000 Mb between chromosomes);
  - Huber (smooth L1) loss on per-bin ratio targets; AdamW lr 1e-4, weight decay 0.01, gradient clipping 1.0,
    batch 64, with a fixed cosine decay to 1e-6 over 150 epochs and no early stopping;
  - every epoch, the complete inference pipeline (windows, bridges, both reading directions, joint stitching) is scored
    on monitoring cells of the validation lines (up to 50 per line, none of the fixed validation cells). Its Pearson is
    logged; it steers nothing;
  - a snapshot every 10 epochs (epNNN.pt); the run's weights are the snapshot at the peak of the 10-epoch moving
    average of the monitoring Pearson (centred on the snapshot: epochs e-5 ... e+4), copied to chosen.pt. Its Pearson
    on the fixed validation cells (150 per line, the complete pipeline again) is the run's validation score in
    summary.json, which scripts/train.py uses to choose the models.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# refcon/refcon/train_binned_dev.py -> repo root is 1 level up (refcon/).
# Set REFCON_ROOT to redirect training output/logs elsewhere.
import os
PROJECT_ROOT = Path(os.environ.get("REFCON_ROOT", str(Path(__file__).resolve().parents[1])))

from .model_binned_dev import build_model
from .data import build_val_windows, load_dataloaders, pipeline_cohort, selection_cells, val_cells
from .eval_cellwide import broadcast_bins_to_genes, gene_positions_for, run_cellwide_inference
from .metrics import per_line_metrics, summarize

# Training run outputs (checkpoints) and logs, repo-relative by default.
BASE_DIR = PROJECT_ROOT / "runs"
LOG_DIR = PROJECT_ROOT / "runs" / "logs"
CONFIG = {
    "seed": 1,
    "bin_size": 8,
    "max_chunk": 2048,
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 4,
    "dim_ff": 512,
    "dropout": 0.1,
    "conv_kernel": 7,
    "batch_size": 64,
    "lr": 1e-4,
    "lr_min": 1e-6,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "max_epochs": 150,
    "coverage": "random",
    "loss_oversample": 2.0,
    "flip_prob": 0.5,
    "bridge_prob": 0.5,
    "phase_aug": True,
    "depth_aug": {"prob": 0.5, "min_rate": 0.1},
    "gene_remove": {"prob": 0.25, "max_share": 0.5},
    "gene_zero": {"prob": 0.25, "max_share": 0.3},
    "pos_gap_mb": 1000.0,
    "val_per_line": 150,
    "monitor": {"cells_per_line": 50, "seed": 1, "exclude_per_line": 150, "exclude_seed": 0},
    "snapshot_every": 10,
    "choose_window": 10,
}


def bin_ratio_targets(cn, lengths, bin_size):
    """Compute per-bin ratio targets.

    Returns (bin_ratios, bin_valid, chunk_mean).
    """
    B, S = cn.shape
    n_bins = S // bin_size
    mask = torch.arange(S, device=cn.device).unsqueeze(0) < lengths.unsqueeze(1)
    valid = (cn > -0.5) & mask
    safe_cn = cn.clone()
    safe_cn[~valid] = 0.0
    counts = valid.float().sum(dim=1).clamp(min=1)
    chunk_mean = safe_cn.sum(dim=1) / counts

    ratios = safe_cn / chunk_mean.unsqueeze(1).clamp(min=0.1)
    ratios[~valid] = 0.0

    usable = n_bins * bin_size
    ratios_trim = ratios[:, :usable].view(B, n_bins, bin_size)
    valid_trim = valid[:, :usable].view(B, n_bins, bin_size)
    bin_count = valid_trim.float().sum(dim=2)
    bin_sum = (ratios_trim * valid_trim.float()).sum(dim=2)
    bin_ratios = bin_sum / bin_count.clamp(min=1)
    bin_valid = bin_count >= 3

    return bin_ratios, bin_valid, chunk_mean


def _rank(a: torch.Tensor) -> torch.Tensor:
    return a.argsort().argsort().double()


def _pearson_t(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double(); b = b.double()
    if a.numel() < 10 or a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std(unbiased=False) * b.std(unbiased=False)))


def _reverse_prefix(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
    Lc = lengths.unsqueeze(1)
    return torch.gather(x, 1, torch.where(idx < Lc, Lc - 1 - idx, idx))


@torch.no_grad()
def evaluate_fixed(model, V: dict, device, bin_size: int, batch: int = 256) -> dict:
    """Pooled bin-level Spearman / Pearson on fixed validation windows (data.build_val_windows), for the forward
    pass ("fwd") and for the forward + reversed average ("bidir"); logged only. Positions are reversed together with
    the window."""
    model.eval()
    fwd, avg = [], []
    for i in range(0, V["x"].shape[0], batch):
        x = V["x"][i:i + batch].to(device)
        L = V["len"][i:i + batch].to(device)
        p = V["pos"][i:i + batch].to(device)
        f, _ = model(x, L, p)
        r, _ = model(_reverse_prefix(x, L), L, _reverse_prefix(p, L))
        nb = L // bin_size                                        # windows are whole bins
        k = torch.arange(f.shape[1], device=device).unsqueeze(0)
        src = (nb.unsqueeze(1) - 1 - k).clamp(min=0)
        r_un = torch.gather(r, 1, src)
        fwd.append(f.cpu()); avg.append(((f + r_un) / 2).cpu())
    fwd = torch.cat(fwd); avg = torch.cat(avg)
    gt, valid, _ = bin_ratio_targets(V["cn"].float(), V["len"], bin_size)
    out = {}
    for name, p in (("fwd", fwd), ("bidir", avg)):
        pv, gv = p[valid], gt[valid]
        out[f"{name}_spearman"] = _pearson_t(_rank(pv), _rank(gv))
        out[f"{name}_pearson"] = _pearson_t(pv, gv)
    return out


@torch.no_grad()
def full_pipeline_score(model, cohorts, device, cell_batch: int = 512) -> dict:
    """The complete inference pipeline (windows, bridges, both reading directions, joint stitching) on each cohort (data.pipeline_cohort), scored per line: Pearson and loss AUROC of each line's
    mean profile with the line's copy number (refcon.metrics.per_line_metrics, genes without a prediction filled with
    the cell's median), averaged over the lines of all cohorts."""
    model.eval()
    P, AL, n_cells = [], [], 0
    for c in cohorts:
        res = run_cellwide_inference(model, c["X_log"], c["chr_ranges"], device,
                                     gene_positions_for(model, c["var"]), cell_batch=cell_batch)
        pred = broadcast_bins_to_genes(res["bin_profile"], res["bin_coords"], c["n_genes"]).astype(np.float32)
        lm = per_line_metrics(pred, c["gt"], c["lines"], None, fill_diploid=True)
        P += [m["P"] for m in lm.values()]
        AL += [m["AL"] for m in lm.values()]
        n_cells += len(c["cells"])
    return {"P": summarize(np.asarray(P, dtype=np.float64))["mean"],
            "AL": summarize(np.asarray(AL, dtype=np.float64))["mean"], "lines": len(P), "cells": n_cells}


def choose_snapshot(scores: dict, snapshot_epochs, window: int = 10):
    """The snapshot at the peak of the moving average of the monitoring score: for snapshot epoch e, the mean score
    of epochs e - window//2 ... e + window - window//2 - 1 (e-5 ... e+4 for 10), over those that exist. Ties go to
    the earlier snapshot. Returns (epoch, {epoch: moving average})."""
    ma = {}
    for e in snapshot_epochs:
        vals = [scores[k] for k in range(e - window // 2, e + window - window // 2)
                if k in scores and np.isfinite(scores[k])]
        ma[e] = float(np.mean(vals)) if vals else float("nan")
    ok = [e for e in snapshot_epochs if np.isfinite(ma[e])]
    if not ok:
        return max(snapshot_epochs), ma
    return max(ok, key=lambda e: (ma[e], -e)), ma


def train(cfg, device, train_data, val_data, out_dir=None, smoke_batches=0):
    bs = cfg["bin_size"]
    mc = cfg["max_chunk"]
    run_id = f"bdev_bs{bs}_mc{mc}_s{cfg['seed']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ckpt_dir = Path(out_dir) if out_dir else BASE_DIR / "checkpoints" / "dev" / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = ckpt_dir if out_dir else LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}.log"

    log = logging.getLogger(run_id)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    for h in [logging.FileHandler(log_path), logging.StreamHandler()]:
        h.setFormatter(fmt)
        log.addHandler(h)
    logging.getLogger("eval_cellwide").setLevel(logging.WARNING)   # the monitoring runs the inference every epoch

    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    log.info("Run: %s", run_id)
    log.info("=" * 70)
    log.info("BINNED DEV - bin_size=%d, max_chunk=%d, n_bins=%d", bs, mc, mc // bs)
    log.info("=" * 70)
    for k, v in sorted(cfg.items()):
        log.info("  %-25s = %s", k, v)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    window_opts = dict(
        flip_prob=cfg["flip_prob"], bridge_prob=cfg["bridge_prob"], phase_aug=cfg["phase_aug"], bin_size=bs,
        seed=cfg["seed"],
        depth_aug_prob=cfg["depth_aug"]["prob"], depth_aug_min=cfg["depth_aug"]["min_rate"],
        gene_remove_prob=cfg["gene_remove"]["prob"], gene_remove_max=cfg["gene_remove"]["max_share"],
        gene_zero_prob=cfg["gene_zero"]["prob"], gene_zero_max=cfg["gene_zero"]["max_share"],
        return_pos=True, pos_gap_mb=cfg["pos_gap_mb"], chrom_sep=True,
    )
    log.info("Loading data (max_chunk=%d)...", mc)
    train_sets = load_dataloaders(
        train_data, max_chunk=mc, batch_size=cfg["batch_size"], loss_oversample=cfg["loss_oversample"],
        coverage=cfg["coverage"], shuffle=True, exclude_lines=tuple(cfg.get("exclude_lines", ())), **window_opts,
    )
    for name, d in train_sets.items():
        if d["ds"].excluded_cells:
            log.info("  %s: excluded %d cells of %s", name, d["ds"].excluded_cells, cfg["exclude_lines"])
    val_sets = load_dataloaders(
        val_data, max_chunk=mc, batch_size=cfg["batch_size"], loss_oversample=1.0, coverage="random", shuffle=False,
        return_pos=True, pos_gap_mb=cfg["pos_gap_mb"],
    )
    val_fixed = [build_val_windows(d["ds"], bs, cfg["val_per_line"], seed=0) for d in val_sets.values()]
    for v in val_fixed:
        log.info("  fixed validation %s: %d cells, %d windows", v["name"], v["n_cells"], v["x"].shape[0])
    mon = cfg["monitor"]                              # complete pipeline on monitoring and validation cells
    rng = np.random.default_rng(mon["seed"])
    monitor = [pipeline_cohort(d["ds"], selection_cells(d["ds"], rng, mon["cells_per_line"],
                                                        mon["exclude_per_line"], mon["exclude_seed"]))
               for d in val_sets.values()]
    decision = [pipeline_cohort(d["ds"], val_cells(d["ds"], cfg["val_per_line"], 0)) for d in val_sets.values()]
    log.info("  monitoring cells (complete pipeline every epoch): %s",
             ", ".join(f"{c['name']} {len(c['cells'])}" for c in monitor))

    train_loaders = {name: d["loader"] for name, d in train_sets.items()}
    log.info("Training: %s | validation: %s", list(train_loaders.keys()), list(val_sets.keys()))

    model = build_model(cfg).to(device)
    log.info("Model: %s params, n_bins=%d", f"{model.count_parameters():,}", model.n_bins)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    snapshot_every = int(cfg["snapshot_every"])

    monitor_P, snapshots = {}, []
    hist_f = open(ckpt_dir / "history.csv", "w", newline="")
    hist = csv.writer(hist_f)
    hist.writerow(["epoch", "seconds", "batches", "loss", "pred_std", "lr", "val_fwd_spearman",
                   "val_bidir_spearman", "monitor_P", "monitor_AL"])

    for epoch in range(cfg["max_epochs"]):
        t0 = time.time()
        lo, hi = cfg["lr_min"], cfg["lr"]             # fixed cosine schedule, never set from validation
        for g in optimizer.param_groups:
            g["lr"] = lo + 0.5 * (hi - lo) * (1 + math.cos(math.pi * epoch / cfg["max_epochs"]))
        model.train()
        iterators = {n: iter(l) for n, l in train_loaders.items()}
        total_loss, bc = 0.0, 0
        finished = set()
        pstds = []

        while len(finished) < len(train_loaders):
            for name in list(train_loaders.keys()):
                if name in finished:
                    continue
                if smoke_batches and bc >= smoke_batches:        # test runs only
                    finished = set(train_loaders)
                    break
                try:
                    item = next(iterators[name])
                except StopIteration:
                    finished.add(name)
                    continue

                expr, cn, length, pos = item
                expr = expr.to(device)
                cn = cn.to(device).float()
                length = length.to(device)
                pos = pos.to(device)

                bin_pred, bin_lengths = model(expr, length, pos)
                bin_gt, bin_valid, _ = bin_ratio_targets(cn, length, bs)
                if not bin_valid.any():
                    continue

                loss = F.smooth_l1_loss(bin_pred[bin_valid], bin_gt[bin_valid])

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()

                total_loss += loss.item()
                pstds.append(bin_pred[bin_valid].std().item())
                bc += 1

        elapsed = time.time() - t0
        avg_loss = total_loss / max(bc, 1)
        lr = optimizer.param_groups[0]["lr"]
        log.info("")
        log.info("Epoch %3d/%d [%.1fs]", epoch + 1, cfg["max_epochs"], elapsed)
        log.info("TRAIN: loss=%.4f pred_std=%.4f lr=%.1e", avg_loss, sum(pstds) / len(pstds) if pstds else 0, lr)

        vm = [evaluate_fixed(model, v, device, bs) for v in val_fixed]
        fwd_s = float(np.mean([m["fwd_spearman"] for m in vm]))
        bidir_s = float(np.mean([m["bidir_spearman"] for m in vm]))
        log.info("VAL [fixed windows, pooled bin level]:")
        for v, m in zip(val_fixed, vm):
            log.info("  %-22s fwd_spear=%.4f fwd_pear=%.4f | bidir_spear=%.4f bidir_pear=%.4f",
                     v["name"], m["fwd_spearman"], m["fwd_pearson"], m["bidir_spearman"], m["bidir_pearson"])
        full = full_pipeline_score(model, monitor, device)
        monitor_P[epoch + 1] = full["P"]
        log.info("MONITOR [complete pipeline, %d cells of %d lines]: Pearson=%.4f loss AUROC=%.4f",
                 full["cells"], full["lines"], full["P"], full["AL"])
        hist.writerow([epoch + 1, round(elapsed, 1), bc, avg_loss, float(np.mean(pstds)) if pstds else 0.0, lr,
                       fwd_s, bidir_s, full["P"], full["AL"]])
        hist_f.flush()

        state = {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(), "config": cfg,
                 "monitor_P": full["P"], "monitor_AL": full["AL"]}
        torch.save(state, ckpt_dir / "last.pt")
        if (snapshot_every and (epoch + 1) % snapshot_every == 0) or epoch + 1 == cfg["max_epochs"]:
            torch.save(state, ckpt_dir / f"ep{epoch + 1:03d}.pt")
            snapshots.append(epoch + 1)
    hist_f.close()

    # the snapshot at the peak of the moving average of the monitoring score, then its validation score on the
    # fixed validation cells (complete pipeline), the score scripts/train.py chooses models on
    summary = {"seed": cfg["seed"], "epochs_run": epoch + 1, "snapshots": snapshots}
    chosen, ma = choose_snapshot(monitor_P, snapshots, int(cfg["choose_window"]))
    shutil.copyfile(ckpt_dir / f"ep{chosen:03d}.pt", ckpt_dir / "chosen.pt")
    summary.update(moving_average={str(k): v for k, v in ma.items()}, chosen_epoch=chosen,
                   chosen=str(ckpt_dir / "chosen.pt"))
    ck = torch.load(ckpt_dir / "chosen.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    val = full_pipeline_score(model, decision, device)
    summary.update(validation_P=val["P"], validation_AL=val["AL"], validation_cells=val["cells"])
    log.info("Chosen snapshot: epoch %d (moving average %.4f); validation (complete pipeline, %d cells): "
             "Pearson=%.4f loss AUROC=%.4f", chosen, ma.get(chosen, float("nan")), val["cells"], val["P"], val["AL"])
    with open(ckpt_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("=" * 70)
    log.info("DONE | %d epochs | chosen snapshot epoch %d -> %s", epoch + 1, chosen, ckpt_dir / "chosen.pt")
    return ckpt_dir / "chosen.pt", summary["validation_P"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--train-data", required=True, help="directory (or file) of processed training .h5ad(s)")
    ap.add_argument("--val-data", required=True, help="directory (or file) of processed validation .h5ad(s)")
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    ap.add_argument("--max-chunk", type=int, default=CONFIG["max_chunk"], help="window length in genes (2048)")
    ap.add_argument("--epochs", type=int, default=CONFIG["max_epochs"], help="epochs, the cosine period (150)")
    ap.add_argument("--snapshot-every", type=int, default=CONFIG["snapshot_every"], help="save epNNN.pt every N epochs")
    ap.add_argument("--exclude-lines", nargs="*", default=[], help="cell lines removed from the training data")
    ap.add_argument("--out-dir", default=None, help="folder for the checkpoints and the log (default runs/checkpoints/dev/<run>)")
    ap.add_argument("--smoke-batches", type=int, default=0, help=">0: stop each epoch after this many batches (tests only)")
    args = ap.parse_args()

    cfg = dict(CONFIG)
    cfg.update(seed=args.seed, max_chunk=args.max_chunk, max_epochs=args.epochs, snapshot_every=args.snapshot_every)
    if args.exclude_lines:
        cfg["exclude_lines"] = sorted(args.exclude_lines)
    train(cfg, torch.device(args.device), args.train_data, args.val_data,
          out_dir=args.out_dir, smoke_batches=args.smoke_batches)


if __name__ == "__main__":
    main()
