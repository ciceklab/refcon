#!/usr/bin/env python3
"""REFCON inference.

Runs the two REFCON models on a processed .h5ad of RAW UMI counts, averages their per-cell copy-number ratios
(NaN-aware mean over the models per gene, then each cell rescaled to mean 1; this mean is the REFCON result), and
records runtime and peak memory. The input needs var 'chromosome' and 'start' (GRCh38) for every gene.

Usage
  python scripts/infer.py --data <sample>.h5ad --out out/sample --device cuda:0
  python scripts/infer.py --forward-only ...   # forward pass only: about half the model time, slightly lower accuracy

Outputs (in --out): refcon_cn.npz (keys cn, spread, cells, genes) and runtime.json.
`spread` is the per-gene standard deviation over the two models.
Reference-free by design: each cell is predicted independently; no reference panel is used.
"""
import argparse, json, os, resource, subprocess, sys, time, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent           # refcon/
CKPTS = [("seed2", "refcon_v3_s2.pt"), ("seed3", "refcon_v3_s3.pt")]


def run_checkpoint(ckpt, h5ad, out_pt, device, cell_batch, forward_only=False):
    # Run eval_cellwide as a package module so its relative imports resolve; put the repo root on
    # PYTHONPATH so this works from a fresh clone (no install) and after `pip install -e .` alike.
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    cmd = [sys.executable, "-m", "refcon.eval_cellwide",
           "--checkpoint", str(ROOT / "checkpoints" / ckpt),
           "--data", str(h5ad), "--output", str(out_pt),
           "--device", device, "--cell-batch", str(cell_batch)] + (["--forward-only"] if forward_only else [])
    t0 = time.time()
    rc = subprocess.run(cmd, env=env).returncode
    return rc, time.time() - t0


def load_pt(p):
    import torch
    d = torch.load(p, map_location="cpu", weights_only=False)
    return (np.asarray(d["predicted_cn_ratios"], float),
            np.asarray(d["cell_ids"]).astype(str),
            np.asarray(d["gene_symbols"]).astype(str))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="processed raw-UMI .h5ad (var has chromosome/start/chr_boundary)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--device", default="cuda:0", help="cuda / mps / cpu (falls back to cpu on failure)")
    ap.add_argument("--cell-batch", type=int, default=512,
                    help="cells per forward call (default 512; the output does not depend on it)")
    ap.add_argument("--forward-only", action="store_true",
                    help="forward pass only (no reversed pass): about half the model time, slightly lower accuracy")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    preds, cells, genes, timing, dev_used = [], None, None, {}, a.device
    for tag, ck in CKPTS:
        o = out / f"pred_{tag}.pt"
        rc, dt = run_checkpoint(ck, a.data, o, a.device, a.cell_batch, a.forward_only)
        if rc != 0 and a.device != "cpu":
            print(f"[{tag}] {a.device} failed; retrying on cpu", flush=True)
            rc, dt = run_checkpoint(ck, a.data, o, "cpu", a.cell_batch, a.forward_only)
            dev_used = "cpu"
        if rc != 0:
            sys.exit(f"[{tag}] inference FAILED (rc={rc})")
        p, cells, genes = load_pt(o)
        preds.append(p); timing[tag] = round(dt, 2)
        print(f"[{tag}] {p.shape} in {dt:.1f}s", flush=True)

    with warnings.catch_warnings():                      # genes outside every bin stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        stack = np.stack(preds)
        ens = np.nanmean(stack, axis=0)
        ens = (ens / np.nanmean(ens, axis=1, keepdims=True)).astype(np.float32)
        spread = np.nanstd(stack, axis=0).astype(np.float32)
    name = "refcon_cn.npz"
    np.savez(out / name, cn=ens, spread=spread, cells=cells, genes=genes)

    total = sum(timing.values())
    # ru_maxrss: bytes on macOS, kilobytes on Linux
    unit = 1e6 if sys.platform == "darwin" else 1e3
    runtime = dict(output=name, device=dev_used, n_cells=int(ens.shape[0]), n_genes=int(ens.shape[1]),
                   n_models=len(preds), cell_batch=a.cell_batch,
                   passes="forward only" if a.forward_only else "forward and reversed",
                   per_model_s=timing, total_inference_s=round(total, 2),
                   cells_per_s=round(ens.shape[0] / max(total, 1e-9), 2),
                   peak_rss_mb=round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / unit, 1))
    (out / "runtime.json").write_text(json.dumps(runtime, indent=2))
    print("RUNTIME", json.dumps(runtime))


if __name__ == "__main__":
    main()
