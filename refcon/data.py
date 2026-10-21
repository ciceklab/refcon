"""Chromosome-chunked dataset for REFCON training.

Key features:
  - Within-chromosome windows: small chromosomes padded, large ones sub-windowed at a random offset
  - Input: log1p of raw counts (the model's masked InstanceNorm handles library size)
  - Loss-region oversampling: chromosomes carrying a copy-number loss are drawn more often
  - Training windows (all off by default, as used for validation):
      bridge_prob  share of windows replaced by a bridge window (the end of one chromosome and the start of the
                   next, built like the inference bridges)
      phase_aug    every window starts 0 to bin_size-1 genes later, so the model sees every way of grouping genes
                   into bins
      flip_prob    share of windows reversed (expression, target and positions together)
      chrom_sep    bridge windows as two whole-bin chromosome segments: with phase_aug each side gets its own start
                   offset, so the boundary stays on a bin edge
      exclude_lines   cell lines removed as if absent from the data
  - Count perturbations (change a window's counts, never its targets):
      depth_aug_prob / depth_aug_min      thinning: every count of the window is thinned binomially at a rate drawn
                                          log-uniformly between depth_aug_min and 1
      gene_remove_prob / gene_remove_max  gene removal: a share U(0, max) of the window's genes is removed; the rest
                                          keep their order, positions and targets
      gene_zero_prob / gene_zero_max      gene zeroing: a share U(0, max) of the window's genes is set to 0, as genes
                                          present but not captured
  - return_pos / pos_gap_mb: items carry each gene's position in Mb relative to the window's first gene in reading
    order (refcon.model_binned_dev.gene_positions_mb; gap between chromosomes)
    Each random option draws from its own random stream, so switching one option does not shift the others' draws.
  - Fixed validation windows (`build_val_windows`): all chromosomes of up to 150 cells per line, the same every epoch.
  - Monitoring cells (`selection_cells`): up to 50 cells per validation line, none of the fixed validation cells, on
    which the complete inference pipeline is scored every epoch.
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .model_binned_dev import gene_positions_mb

logger = logging.getLogger(__name__)

# CN threshold of a loss (used for loss-chromosome oversampling)
CN_LOSS_THRESH = 1.5
MAX_CHUNK = 2048


# ======================================================================
# Chromosome-Chunked Dataset
# ======================================================================

class ChrDataset(Dataset):
    """Within-chromosome chunked dataset.

    Two coverage modes:
      - "full": __len__ = n_cells * n_chromosomes. Every (cell, chr) seen per epoch.
      - "random": __len__ = n_cells. Each cell picks a random chromosome per call.

    Args:
        h5ad_path: Path to a processed h5ad file.
        cn_type: CN label type (bulk, per_line, per_cell, pseudobulk). Defaults to the
                 value stored in ``adata.uns["cn_type"]``.
        max_chunk: Maximum chunk length. Chromosomes > max_chunk are sub-chunked.
        coverage: "full" (n_cells * n_chrs samples) or "random" (n_cells samples, random chr).
    """

    def __init__(
        self,
        h5ad_path: str | Path,
        cn_type: str | None = None,
        max_chunk: int = MAX_CHUNK,
        coverage: str = "full",
        flip_prob: float = 0.0,
        bridge_prob: float = 0.0,
        phase_aug: bool = False,
        bin_size: int = 8,
        seed: int = 0,
        exclude_lines=(),
        depth_aug_prob: float = 0.0,
        depth_aug_min: float = 1.0,
        gene_remove_prob: float = 0.0,
        gene_remove_max: float = 0.0,
        gene_zero_prob: float = 0.0,
        gene_zero_max: float = 0.0,
        return_pos: bool = False,
        pos_gap_mb: float = 0.0,
        chrom_sep: bool = False,
    ):
        self.max_chunk = max_chunk
        self.coverage = coverage
        if chrom_sep and not (return_pos and pos_gap_mb > 0):
            raise ValueError("chromosome separation needs positions with a gap between chromosomes (pos_gap_mb > 0)")

        adata = ad.read_h5ad(str(h5ad_path))
        self.dataset_name = adata.uns.get("dataset_name", Path(h5ad_path).stem)
        if cn_type is None:
            cn_type = adata.uns.get("cn_type", "per_cell")
        if adata.n_obs == 0:
            raise ValueError(f"No cells in {h5ad_path}")

        cn_profiles_dict = adata.uns.get("cn_profiles", {})
        cn_matrix_full = adata.uns.get("cn_matrix", None)
        cn_cell_ids_full = adata.uns.get("cn_cell_ids", None)

        cell_indices = np.arange(adata.n_obs)
        adata_sub = adata
        # Cell lines removed as if absent from the corpus (matched on obs cell_line / Cell_line / cn_profile):
        # their cells, their label profiles and their share of the loss-chromosome mask.
        self.excluded_cells = 0
        if exclude_lines:
            drop = np.zeros(adata.n_obs, dtype=bool)
            for col in ("cell_line", "Cell_line", "cn_profile"):
                if col in adata.obs.columns:
                    drop |= np.isin(adata.obs[col].astype(str).values, list(exclude_lines))
            if drop.all():
                raise ValueError(f"every cell of {h5ad_path} is in exclude_lines")
            if drop.any():
                cell_indices = np.nonzero(~drop)[0]
                adata_sub = adata[cell_indices]
                self.excluded_cells = int(drop.sum())

        self.n_cells = adata_sub.n_obs
        self.n_genes = adata_sub.n_vars
        # Line of each cell (for per-line scores): obs cell_line, else Cell_line, else cn_profile
        self.lines = None
        for col in ("cell_line", "Cell_line", "cn_profile"):
            if col in adata_sub.obs.columns:
                self.lines = adata_sub.obs[col].astype(str).values.copy()
                break

        # Expression matrix (keep sparse)
        self.X = adata_sub.X.copy() if sp.issparse(adata_sub.X) else adata_sub.X.copy()
        self.var = adata_sub.var.copy()

        # Chromosome boundaries
        self.chr_boundary = adata_sub.var["chr_boundary"].values.astype(np.float32).copy()
        self.chr_ranges = self._build_chromosome_ranges()
        self.n_chromosomes = len(self.chr_ranges)

        # Build CN labels
        self._build_cn(cn_type, adata_sub, cn_profiles_dict, cn_matrix_full,
                        cn_cell_ids_full, cell_indices)

        # gene positions (Mb along GRCh38, gap between chromosomes)
        self.return_pos = bool(return_pos)
        self.pos_gap_mb = float(pos_gap_mb)
        self.gene_pos = gene_positions_mb(adata_sub.var, self.pos_gap_mb) if self.return_pos else None

        # Precompute which chromosomes have loss genes (for oversampling)
        self._loss_chr_mask = self._build_loss_chr_mask()

        # training windows; separate random streams, so switching one option does not shift the others' draws
        self.flip_prob = float(flip_prob)
        self.bridge_prob = float(bridge_prob)
        self.phase_aug = bool(phase_aug)
        self.bin_size = int(bin_size)
        self._flip_rng = np.random.default_rng(seed + 1000)
        self._bridge_rng = np.random.default_rng(seed + 2000)
        self._phase_rng = np.random.default_rng(seed + 3000)
        # count perturbations, each with its own random stream
        self.chrom_sep = bool(chrom_sep)
        self.depth_aug_prob, self.depth_aug_min = float(depth_aug_prob), float(depth_aug_min)
        self.gene_remove_prob, self.gene_remove_max = float(gene_remove_prob), float(gene_remove_max)
        self.gene_zero_prob, self.gene_zero_max = float(gene_zero_prob), float(gene_zero_max)
        self._depth_rng = np.random.default_rng(seed + 4000)
        self._remove_rng = np.random.default_rng(seed + 5000)
        self._zero_rng = np.random.default_rng(seed + 6000)
        if self.depth_aug_prob > 0:                            # thinning needs counts
            if not 0 < self.depth_aug_min < 1:
                raise ValueError(f"depth_aug_min must be in (0, 1), not {self.depth_aug_min}")
            d = self.X.data if sp.issparse(self.X) else np.asarray(self.X)
            if not (np.all(d >= 0) and np.all(np.mod(d, 1) == 0)):
                raise ValueError(f"{h5ad_path}: thinning needs raw integer counts in X")
        # Allowed bridge windows per adjacent chromosome pair, exactly as the inference bridges: the boundary on a
        # bin edge, at least 2 bins on each side, at most max_chunk genes, cut at the last whole bin of the right
        # chromosome; the start is drawn among these, the pair uniformly.
        self._bridge_starts = []
        for i in range(self.n_chromosomes - 1):
            cs_i, ce_i = self.chr_ranges[i]
            cs_j, ce_j = self.chr_ranges[i + 1]
            b, bs = ce_i, self.bin_size
            lo = max(cs_i, b - max_chunk + bs)
            hi = min(ce_j - bs, b - bs)
            first = lo + ((b % bs - lo) % bs)
            ok = []
            for ws in range(first, hi + 1, bs):
                we = ws + max_chunk
                if we > ce_j:
                    we = cs_j + ((ce_j - cs_j) // bs) * bs
                if (b - ws) // bs >= 2 and (we - b) // bs >= 2:
                    ok.append((int(ws), int(we)))
            self._bridge_starts.append(ok)

        logger.info(
            "%s loaded: %d cells x %d genes x %d chrs = %d samples, coverage=%s",
            self.dataset_name, self.n_cells, self.n_genes,
            self.n_chromosomes, len(self), self.coverage,
        )

    # ------------------------------------------------------------------
    # Chromosome ranges
    # ------------------------------------------------------------------

    def _build_chromosome_ranges(self) -> list[tuple[int, int]]:
        """Precompute (start, end) for each chromosome from chr_boundary."""
        starts = np.where(self.chr_boundary > 0.5)[0]
        ranges = []
        for i, s in enumerate(starts):
            e = starts[i + 1] if i + 1 < len(starts) else self.n_genes
            ranges.append((s, e))
        return ranges

    def _build_loss_chr_mask(self) -> np.ndarray:
        """Boolean mask: which chromosomes contain loss genes.

        Returns (n_chromosomes,) bool array.
        """
        if self._cn_mode == "shared":
            cn = self._cn_continuous_shared
        elif self._cn_mode == "per_line":
            # Union across all profiles
            cn = self._cn_continuous_profiles.min(axis=0)
        else:
            cn = self._cn_continuous_matrix.min(axis=0)

        mask = np.zeros(self.n_chromosomes, dtype=bool)
        for ci, (s, e) in enumerate(self.chr_ranges):
            chr_cn = cn[s:e]
            valid = ~np.isnan(chr_cn) & (chr_cn >= 0)
            if valid.any() and (chr_cn[valid] < CN_LOSS_THRESH).any():
                mask[ci] = True
        return mask

    # ------------------------------------------------------------------
    # CN labels
    # ------------------------------------------------------------------

    def _build_cn(self, cn_type, adata_sub, cn_profiles_dict, cn_matrix_full,
                  cn_cell_ids_full, cell_indices):
        if cn_type in ("bulk", "pseudobulk"):
            self._cn_continuous_shared = adata_sub.var["cn_ground_truth"].values.astype(
                np.float32
            ).copy()
            self._cn_mode = "shared"

        elif cn_type == "per_line":
            profile_names = adata_sub.obs["cn_profile"].values
            unique_names = sorted(set(profile_names))
            name_to_idx = {n: i for i, n in enumerate(unique_names)}

            profiles_continuous = []
            for name in unique_names:
                raw = np.array(cn_profiles_dict[name], dtype=np.float32)
                profiles_continuous.append(raw.copy())
            self._cn_continuous_profiles = np.stack(profiles_continuous)
            self._cn_cell_idx = np.array(
                [name_to_idx[n] for n in profile_names], dtype=np.int32
            )
            self._cn_mode = "per_line"

        elif cn_type == "per_cell":
            if cn_matrix_full is not None and cn_cell_ids_full is not None:
                id_to_row = {cid: i for i, cid in enumerate(cn_cell_ids_full)}
                sub_ids = adata_sub.obs.index.values
                rows = np.array([id_to_row[cid] for cid in sub_ids], dtype=np.int64)
                cn_sub = cn_matrix_full[rows]
            elif cn_matrix_full is not None:
                cn_sub = cn_matrix_full[cell_indices]
            else:
                raise ValueError("per_cell cn_type but no cn_matrix in uns")
            self._cn_continuous_matrix = cn_sub.astype(np.float32).copy()
            self._cn_mode = "per_cell"
        else:
            raise ValueError(f"Unknown cn_type: {cn_type}")

    def _get_cn_continuous(self, idx: int) -> np.ndarray:
        """(n_genes,) float32 continuous CN values. NO capping."""
        if self._cn_mode == "shared":
            return self._cn_continuous_shared
        elif self._cn_mode == "per_line":
            return self._cn_continuous_profiles[self._cn_cell_idx[idx]]
        else:
            return self._cn_continuous_matrix[idx]

    # ------------------------------------------------------------------
    # __len__ and __getitem__
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self.coverage == "full":
            return self.n_cells * self.n_chromosomes
        else:
            return self.n_cells

    def __getitem__(self, idx: int):
        """Returns a sample tuple padded to max_chunk:
            (expression, cn, length), or with return_pos
            (expression, cn, length, positions) - each gene's position in Mb relative to the window's first gene
        """
        if self.coverage == "full":
            cell_idx = idx // self.n_chromosomes
            chr_idx = idx % self.n_chromosomes
        else:
            cell_idx = idx
            if hasattr(self, '_chr_sample_probs'):
                chr_idx = np.random.choice(self.n_chromosomes, p=self._chr_sample_probs)
            else:
                chr_idx = np.random.randint(0, self.n_chromosomes)
        chr_start, chr_end = self.chr_ranges[chr_idx]
        chr_size = chr_end - chr_start

        # Get raw expression for this cell
        row = self.X[cell_idx]
        if sp.issparse(row):
            x = row.toarray().ravel()
        else:
            x = np.asarray(row).ravel()
        x = x.astype(np.float32)

        # Get CN for this cell
        cn_full = self._get_cn_continuous(cell_idx).astype(np.float32)
        cn_full = np.where(np.isnan(cn_full), -1.0, cn_full)

        # Extract chromosome region
        if chr_size > self.max_chunk:
            # Random sub-chunk within chromosome
            max_start = chr_size - self.max_chunk
            offset = np.random.randint(0, max_start + 1)
            start = chr_start + offset
            end = start + self.max_chunk
        else:
            start = chr_start
            end = chr_end

        # bridge window instead, start shift; separated bridge (each side shifted on its own)
        segs = None
        if self.bridge_prob > 0 and self._bridge_rng.random() < self.bridge_prob:
            pairs = [k for k, ok in enumerate(self._bridge_starts) if ok]
            pk = pairs[self._bridge_rng.integers(len(pairs))]
            ok = self._bridge_starts[pk]
            start, end = ok[self._bridge_rng.integers(len(ok))]
            if self.chrom_sep:
                segs = self._sep_bridge(start, self.chr_ranges[pk][1], end)
        if segs is None:
            if self.phase_aug and end - start > 4 * self.bin_size:
                start = start + int(self._phase_rng.integers(0, self.bin_size))
            segs = [(start, end)]
        raw = np.concatenate([x[a:b] for a, b in segs])
        cn = np.concatenate([cn_full[a:b] for a, b in segs])
        pos = np.concatenate([self.gene_pos[a:b] for a, b in segs]) if self.return_pos else None

        # thinning, gene removal, gene zeroing (counts only; targets unchanged)
        raw, cn, pos = self._augment(raw, cn, pos, [b - a for a, b in segs])
        length = raw.shape[0]
        expr = np.log1p(raw)

        # reversal (expression, target and positions together)
        if self.flip_prob > 0 and self._flip_rng.random() < self.flip_prob:
            expr, cn = expr[::-1], cn[::-1]
            pos = pos[::-1] if pos is not None else None

        # Build output chunks with padding
        expr_chunk = np.zeros(self.max_chunk, dtype=np.float32)
        cn_chunk = np.full(self.max_chunk, -1.0, dtype=np.float32)
        expr_chunk[:length] = expr
        cn_chunk[:length] = cn
        if not self.return_pos:
            return (
                torch.from_numpy(expr_chunk),
                torch.from_numpy(cn_chunk),
                length,
            )
        pos_chunk = np.zeros(self.max_chunk, dtype=np.float32)
        pos_chunk[:length] = pos - pos[0]                     # window-relative (float64 before rounding)
        return (
            torch.from_numpy(expr_chunk),
            torch.from_numpy(cn_chunk),
            length,
            torch.from_numpy(pos_chunk),
        )

    def _sep_bridge(self, s, b, e):
        """Bridge window [s, e) over the chromosome boundary b as two whole-bin segments. With phase_aug each side
        gets its own grouping into bins: r genes next to the boundary are left out and that side is trimmed to whole
        bins at its far end (a side with fewer than 3 bins keeps its grouping). The boundary stays on a bin edge, as
        at inference, and no bin holds genes of two chromosomes."""
        bs = self.bin_size
        (ls, le), (rs, re) = (s, b), (b, e)
        assert (le - ls) % bs == 0 and (re - rs) % bs == 0
        if self.phase_aug:
            rl, rr = int(self._phase_rng.integers(0, bs)), int(self._phase_rng.integers(0, bs))
            if rl and (le - ls) // bs >= 3:
                ls, le = ls + (bs - rl), le - rl
            if rr and (re - rs) // bs >= 3:
                rs, re = rs + rr, re - (bs - rr)
        return [(ls, le), (rs, re)]

    def _augment(self, raw: np.ndarray, cn: np.ndarray, pos=None, seg_lens=None):
        """Thinning, then gene removal, then gene zeroing (each only when its option is on; targets are never
        changed). Removed genes leave the window with their targets and positions. In a window of two separated
        chromosomes each side is trimmed back to whole bins after removal (at its far end), so the boundary stays on
        a bin edge."""
        if self.depth_aug_prob > 0 and self._depth_rng.random() < self.depth_aug_prob:
            rate = float(np.exp(self._depth_rng.uniform(np.log(self.depth_aug_min), 0.0)))
            raw = self._depth_rng.binomial(np.rint(raw).astype(np.int64), rate).astype(np.float32)
        if self.gene_remove_prob > 0 and self._remove_rng.random() < self.gene_remove_prob:
            keep = self._remove_rng.random(raw.shape[0]) >= self._remove_rng.uniform(0.0, self.gene_remove_max)
            if seg_lens is not None and len(seg_lens) == 2:
                bs, nl = self.bin_size, seg_lens[0]
                kl, kr = np.nonzero(keep[:nl])[0], nl + np.nonzero(keep[nl:])[0]
                kl, kr = kl[len(kl) % bs:], kr[:len(kr) - len(kr) % bs]
                keep = np.zeros_like(keep)
                if len(kl) >= 2 * bs and len(kr) >= 2 * bs:
                    keep[kl] = True
                    keep[kr] = True
                else:                                          # too short to stay whole: keep every gene
                    keep[:] = True
            if keep.sum() >= 4 * self.bin_size:                # short windows keep all their genes
                raw, cn = raw[keep], cn[keep]
                pos = pos[keep] if pos is not None else None
        if self.gene_zero_prob > 0 and self._zero_rng.random() < self.gene_zero_prob:
            drop = self._zero_rng.random(raw.shape[0]) < self._zero_rng.uniform(0.0, self.gene_zero_max)
            raw = np.where(drop, np.float32(0.0), raw).astype(np.float32)
        return raw, cn, pos

    # ------------------------------------------------------------------
    # Oversampling support
    # ------------------------------------------------------------------

    def get_sample_weights(self, loss_oversample: float = 2.0) -> np.ndarray | None:
        """Per-sample weights for WeightedRandomSampler.

        Only works for coverage="full". For "random", chromosome selection
        is randomized per call - use oversample_chr_probs instead.

        Args:
            loss_oversample: Weight multiplier for loss-containing chromosomes.

        Returns:
            (len(self),) float array of weights, or None for "random" mode.
        """
        if self.coverage == "random":
            # For random mode, build chromosome sampling probabilities
            # that bias toward loss chromosomes
            probs = np.ones(self.n_chromosomes, dtype=np.float64)
            probs[self._loss_chr_mask] *= loss_oversample
            self._chr_sample_probs = probs / probs.sum()
            return None

        weights = np.ones(len(self), dtype=np.float64)
        for chr_idx in range(self.n_chromosomes):
            if self._loss_chr_mask[chr_idx]:
                indices = np.arange(chr_idx, len(self), self.n_chromosomes)
                weights[indices] *= loss_oversample
        return weights


# ======================================================================
# Fixed validation windows
# ======================================================================

def val_cells(ds: ChrDataset, per_line: int = 150, seed: int = 0) -> np.ndarray:
    """Fixed subset of a validation dataset: up to per_line cells of every cell line (per_line datasets) or of the
    whole dataset (bulk / per-cell labels); the same cells for every run with the same seed."""
    rng = np.random.default_rng(seed)
    lines = ds._cn_cell_idx if ds._cn_mode == "per_line" else np.zeros(ds.n_cells, dtype=np.int32)
    keep = []
    for li in np.unique(lines):
        idx = np.nonzero(lines == li)[0]
        keep.extend(sorted(rng.choice(idx, size=min(per_line, idx.size), replace=False).tolist()))
    return np.asarray(keep, dtype=np.int64)


def build_val_windows(ds: ChrDataset, bin_size: int, per_line: int = 150, seed: int = 0) -> dict:
    """All chromosomes of the fixed validation cells, one whole-chromosome window each (as in training, at most
    ds.max_chunk genes), truncated to a multiple of bin_size so that a reversed window maps bin k to bin nb-1-k.
    When the dataset carries gene positions (return_pos), each window also gets its genes' positions relative
    to its first gene ("pos")."""
    cells = val_cells(ds, per_line, seed)
    rows_x, rows_c, lens, rows_p = [], [], [], []
    for c in cells:
        row = ds.X[int(c)]
        x_full = np.log1p((row.toarray().ravel() if sp.issparse(row) else np.asarray(row).ravel()).astype(np.float32))
        cn_full = ds._get_cn_continuous(int(c))
        for s, e in ds.chr_ranges:
            L = (min(e - s, ds.max_chunk) // bin_size) * bin_size
            if L < 2 * bin_size:
                continue
            x = np.zeros(ds.max_chunk, np.float32)
            cn = np.full(ds.max_chunk, -1.0, np.float32)
            x[:L] = x_full[s:s + L]
            v = cn_full[s:s + L].astype(np.float32)
            cn[:L] = np.where(np.isnan(v), -1.0, v)
            rows_x.append(x); rows_c.append(cn); lens.append(L)
            if ds.gene_pos is not None:
                p = np.zeros(ds.max_chunk, np.float32)
                p[:L] = ds.gene_pos[s:s + L] - ds.gene_pos[s]
                rows_p.append(p)
    V = {"x": torch.from_numpy(np.stack(rows_x)), "cn": torch.from_numpy(np.stack(rows_c)),
         "len": torch.tensor(lens, dtype=torch.long), "n_cells": int(cells.size), "name": ds.dataset_name}
    if rows_p:
        V["pos"] = torch.from_numpy(np.stack(rows_p))
    return V


# ======================================================================
# Full-pipeline cohorts: monitoring cells and validation cells
# ======================================================================

def selection_cells(ds: ChrDataset, rng: np.random.Generator, per_line: int = 50,
                    exclude_per_line: int = 150, exclude_seed: int = 0) -> np.ndarray:
    """Monitoring cells of a validation dataset: up to per_line cells of every line, drawn with `rng` (shared
    across the validation datasets, in their order), none of them among the fixed validation cells
    (val_cells(ds, exclude_per_line, exclude_seed), the cells on which finished models are compared)."""
    used = set(val_cells(ds, exclude_per_line, exclude_seed).tolist())
    lines = ds._cn_cell_idx if ds._cn_mode == "per_line" else np.zeros(ds.n_cells, dtype=np.int32)
    keep = []
    for li in np.unique(lines):
        cand = np.array([i for i in np.nonzero(lines == li)[0] if i not in used])
        if cand.size:
            keep.extend(np.sort(rng.choice(cand, size=min(per_line, cand.size), replace=False)).tolist())
    return np.asarray(keep, dtype=np.int64)


def pipeline_cohort(ds: ChrDataset, cells: np.ndarray) -> dict:
    """The given cells of a dataset as a cohort for the complete inference pipeline: log1p expression of every gene,
    chromosome ranges, each cell's absolute copy number (-1 = no call) and line, the dataset's var."""
    cells = np.asarray(cells, dtype=np.int64)
    X = ds.X[cells]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    gt = np.stack([ds._get_cn_continuous(int(c)) for c in cells]).astype(np.float32)
    gt = np.where(np.isfinite(gt) & (gt >= 0), gt, -1.0).astype(np.float32)
    lines = ds.lines[cells] if ds.lines is not None else np.array([ds.dataset_name] * len(cells))
    chr_ranges = [(int(s), int(e)) for s, e in ds.chr_ranges]
    if chr_ranges and chr_ranges[0][0] != 0:                  # genes before the first flagged boundary: own range
        chr_ranges = [(0, chr_ranges[0][0])] + chr_ranges
    return {"name": ds.dataset_name, "cells": cells, "X_log": np.log1p(X.astype(np.float32)), "gt": gt,
            "lines": lines, "chr_ranges": chr_ranges, "var": ds.var, "n_genes": ds.n_genes}


# ======================================================================
# DataLoader factory
# ======================================================================

def load_dataloaders(
    path: str | Path,
    max_chunk: int = MAX_CHUNK,
    batch_size: int = 64,
    num_workers: int = 0,
    loss_oversample: float = 2.0,
    coverage: str = "full",
    shuffle: bool = True,
    **window_options,
) -> dict[str, dict]:
    """Load every ``.h5ad`` under ``path`` (a directory or a single file) as a ChrDataset + DataLoader.

    Each file is expected to be processed and self-describing: raw-count expression in ``X``, a
    ``chr_boundary`` column in ``.var``, and the copy-number ground truth and ``cn_type`` in ``.uns``
    (see docs/input_format.md). No dataset registry or split column is required; the caller decides
    which files are training vs. validation by passing different paths.

    ``window_options`` (the ChrDataset options above) go to every ChrDataset.

    Returns: ``{dataset_name: {"ds": ChrDataset, "loader": DataLoader}}``.
    """
    path = Path(path)
    files = sorted(path.glob("*.h5ad")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no .h5ad files found under {path}")

    result: dict = {}
    for h5ad_path in files:
        try:
            dataset = ChrDataset(
                h5ad_path=h5ad_path,
                max_chunk=max_chunk,
                coverage=coverage,
                **window_options,
            )
        except ValueError as e:
            logger.warning("Skipping %s: %s", h5ad_path.name, e)
            continue

        if shuffle and loss_oversample > 1.0 and coverage == "full":
            weights = dataset.get_sample_weights(loss_oversample)
            sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
            loader = DataLoader(
                dataset, batch_size=batch_size, sampler=sampler,
                num_workers=num_workers, pin_memory=True, drop_last=False,
            )
        else:
            if shuffle and loss_oversample > 1.0 and coverage == "random":
                dataset.get_sample_weights(loss_oversample)  # sets _chr_sample_probs
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers, pin_memory=True, drop_last=False,
            )

        result[dataset.dataset_name] = {"ds": dataset, "loader": loader}

    return result
