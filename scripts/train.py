#!/usr/bin/env python3
"""Train the REFCON models.

Three runs at context length 2048 (seeds 1, 2, 3), 150 epochs each with a fixed cosine learning rate. Each run's
weights are its snapshot at the peak of the 10-epoch moving average of the monitoring score (the complete inference
pipeline on monitoring cells of the validation lines, every epoch). REFCON averages the two runs with the higher
validation Pearson (the complete pipeline on the fixed validation cells, 150 per line); they are copied to
<out-dir>/refcon_v3_s<seed>.pt.

Recipe (refcon/train_binned_dev.py): every cohort of --train-data, the cell lines and the healthy cells labelled
diploid; bridge windows (probability 0.5, each chromosome with its own start offset so the boundary stays on a bin
edge), reversed windows (0.5), a start offset of 0 to 7 genes; thinning (0.5, rate log-uniform in [0.1, 1]), gene
removal (0.25, up to 50%), gene zeroing (0.25, up to 30%); megabase rotary positions and chromosome separation; Huber
loss; AdamW lr 1e-4 with cosine decay to 1e-6 over 150 epochs.

Model:
  architecture : 4-layer Transformer, d_model=128, 4 heads, ffn=512, dropout=0.1, pre-norm, rotary positions on q/k;
                 masked InstanceNorm on log1p input; Conv1d binner (bin_size=8); 975,299 parameters per model.
  optimizer    : AdamW, lr=1e-4, weight_decay=0.01, grad-clip 1.0, batch_size=64.
  loss         : SmoothL1(beta=1) on per-bin ratio targets; chromosomes carrying >=1 loss drawn 2x as often.
  input        : log1p(raw UMI counts); per-cell masked InstanceNorm handles library size/scale.

Training data is NOT bundled here (see Data availability). Point --train-data and --val-data at directories
(or single files) of processed cohorts, each a self-describing .h5ad (raw counts in X, chr_boundary, chromosome and
start in .var, copy-number ground truth and cn_type in .uns; see docs/input_format.md). The training directory holds
the healthy cells as one more cohort, with copy number 2 at every gene (data/README.md).

Usage
  python scripts/train.py --train-data /path/to/train_cohorts --val-data /path/to/val_cohorts \\
      --device cuda:0 --out-dir checkpoints/
"""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-data", required=True, help="dir (or file) of processed training .h5ad(s)")
    ap.add_argument("--val-data", required=True, help="dir (or file) of processed validation .h5ad(s)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=str(ROOT / "checkpoints"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="seeds to train (default 1 2 3)")
    ap.add_argument("--keep", type=int, default=2, help="runs kept, by validation score (default 2)")
    ap.add_argument("--exclude-lines", nargs="*", default=[], help="cell lines removed from the training data")
    ap.add_argument("--max-epochs", type=int, default=None, help="epochs (default 150)")
    ap.add_argument("--snapshot-every", type=int, default=None, help="snapshot every N epochs (default 10)")
    ap.add_argument("--smoke-batches", type=int, default=0, help=">0: stop each epoch after this many batches (tests only)")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    runs = []
    for seed in a.seeds:
        run_dir = out / "runs" / f"seed{seed}"
        cmd = [sys.executable, "-m", "refcon.train_binned_dev", "--seed", str(seed),
               "--train-data", a.train_data, "--val-data", a.val_data, "--device", a.device, "--out-dir", str(run_dir)]
        if a.max_epochs:
            cmd += ["--epochs", str(a.max_epochs)]
        if a.snapshot_every:
            cmd += ["--snapshot-every", str(a.snapshot_every)]
        if a.smoke_batches:
            cmd += ["--smoke-batches", str(a.smoke_batches)]
        if a.exclude_lines:
            cmd += ["--exclude-lines", *a.exclude_lines]
        print(f"\n=== training seed {seed} ===\n{' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, env=env).returncode
        if rc != 0:
            sys.exit(f"training seed {seed} failed (rc={rc})")
        s = json.load(open(run_dir / "summary.json"))   # the chosen snapshot and its validation Pearson
        runs.append((float(s["validation_P"]), seed, run_dir / "chosen.pt", s["chosen_epoch"]))
    ranked = sorted(runs, key=lambda r: (-r[0], r[1]))
    for score, seed, path, ep in ranked[:a.keep]:
        name = f"refcon_v3_s{seed}.pt"
        shutil.copyfile(path, out / name)
        print(f"kept seed {seed} (epoch {ep}, validation {score:.4f}) -> {out / name}", flush=True)
    print(f"validation score by seed {[(s, round(v, 4)) for v, s, _, _ in runs]}", flush=True)
    print("\nDone. REFCON is the per-cell mean of the models' predicted_cn_ratios (scripts/infer.py), which reads "
          "checkpoints/refcon_v3_s2.pt and refcon_v3_s3.pt; rename models trained here to use them.")


if __name__ == "__main__":
    main()
