"""BinnedDevModel - per-bin CN ratio predictor with masked InstanceNorm.

Architecture:
  raw log1p expression (B, max_chunk), zero-padded after `lengths`
  -> masked InstanceNorm (per-sample, on the real-gene prefix only)
  -> Conv1d(1, d_conv, kernel=bin_size, stride=bin_size) + GELU
      -> n_bins = max_chunk // bin_size tokens of d_conv dims each
  -> MaskedConvStemTransformer backbone (d_model=128, 4 layers, rotary positions)
  -> ratio_head -> (B, n_bins) per-bin CN/chunk_mean ratios

Positions: the rotary position of a bin is its genomic position in megabases, the mean over its genes of each gene's
distance from the window's first gene in reading order (so a reversed window reads as the mirrored genome). The
forward pass therefore needs `gene_pos`, each gene's position in Mb (`gene_positions_mb` below; only
differences within a window matter).

Chromosome separation: in a window that spans two chromosomes, attention stays within a chromosome and the convolution
stem runs on each chromosome as if it were a window of its own. The model finds the boundary from the gap of
`pos_gap_mb` placed between chromosomes in the positions. The two chromosomes share only the window's input
normalisation and the target (each bin relative to the window's mean).

`predict_per_gene` broadcasts bin predictions back to gene resolution
for downstream gene-level evaluation.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .model import MaskedConvStemTransformer


# ---------------------------------------------------------------------------
# Gene positions: each gene's start coordinate in megabases along GRCh38 with the chromosomes 1-22, X and Y laid end
# to end (UCSC GRCh38 cytoband lengths). With `gap_mb` > 0 that many megabases are added between consecutive
# chromosomes; the model uses a gap of 1,000 Mb to mark where a chromosome ends (never used as a distance, because
# neither attention nor convolution crosses a chromosome boundary). Only differences inside a window reach the model.
# ---------------------------------------------------------------------------

CHROMOSOMES = [str(i) for i in range(1, 23)] + ["X", "Y"]

# GRCh38 chromosome lengths in bp (UCSC cytoBand hg38: end of each chromosome's last band)
GRCH38_LENGTHS = {
    "1": 248956422, "2": 242193529, "3": 198295559, "4": 190214555, "5": 181538259, "6": 170805979,
    "7": 159345973, "8": 145138636, "9": 138394717, "10": 133797422, "11": 135086622, "12": 133275309,
    "13": 114364328, "14": 107043718, "15": 101991189, "16": 90338345, "17": 83257441, "18": 80373285,
    "19": 58617616, "20": 64444167, "21": 46709983, "22": 50818468, "X": 156040895, "Y": 57227415,
}


def chromosome_offsets(gap_mb: float = 0.0) -> dict:
    """Start of each chromosome in bp on the concatenated genome (gap_mb megabases between chromosomes)."""
    off, acc = {}, 0.0
    for c in CHROMOSOMES:
        off[c] = acc
        acc += float(GRCH38_LENGTHS[c]) + gap_mb * 1e6
    return off


def gene_positions_mb(var, gap_mb: float = 0.0) -> np.ndarray:
    """Each gene's position in Mb (float64) from `var["chromosome"]` (chr1 ... chrY or 1 ... Y) and `var["start"]`
    (GRCh38 bp coordinates), with gap_mb megabases between consecutive chromosomes."""
    for col in ("chromosome", "start"):
        if col not in var.columns:
            raise ValueError(f"REFCON needs var['{col}'] (GRCh38) to place each gene; "
                             "see docs/input_format.md")
    off = chromosome_offsets(gap_mb)
    chrom = var["chromosome"].astype(str).str.replace("chr", "", regex=False).values
    unknown = sorted(set(chrom) - set(CHROMOSOMES))
    if unknown:
        raise ValueError(f"chromosomes outside 1-22, X, Y in var['chromosome']: {unknown}")
    return ((np.array([off[c] for c in chrom]) + var["start"].values.astype(np.float64)) / 1e6).astype(np.float64)


def bin_positions(gene_pos: torch.Tensor, lengths: torch.Tensor, bin_size: int) -> torch.Tensor:
    """Rotary position of each bin in Mb: the mean over its genes of |gene position - position of the window's
    first gene in reading order|. gene_pos (B, S) per gene, lengths (B,) -> (B, S // bin_size)."""
    B, S = gene_pos.shape
    nb = S // bin_size
    valid = (torch.arange(S, device=gene_pos.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
    if gene_pos.device.type == "mps":   # Apple GPUs have no float64; positions are window-relative, float32 suffices
        rel = (gene_pos.float() - gene_pos[:, :1].float()).abs() * valid
    else:
        rel = (gene_pos.double() - gene_pos[:, :1].double()).abs().float() * valid
    rel = rel[:, :nb * bin_size].view(B, nb, bin_size)
    v = valid[:, :nb * bin_size].view(B, nb, bin_size)
    return rel.sum(-1) / v.sum(-1).clamp(min=1.0)


def bin_segments(pos: torch.Tensor, valid: torch.Tensor, gap_mb: float) -> torch.Tensor:
    """Chromosome of each bin within its window (0, 1 in reading order; -1 for padding): a new chromosome starts
    where the bin position jumps by more than half the gap placed between chromosomes."""
    jump = torch.zeros_like(pos, dtype=torch.long)
    jump[:, 1:] = (((pos[:, 1:] - pos[:, :-1]) > gap_mb / 2) & valid[:, 1:]).long()
    return torch.where(valid, jump.cumsum(1), torch.full_like(jump, -1))


class BinnedDevModel(nn.Module):

    def __init__(
        self,
        bin_size: int = 8,
        max_chunk: int = 2048,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.1,
        conv_kernel_size: int = 7,
        d_conv: int = 64,
        pos_gap_mb: float = 1000.0,
    ):
        super().__init__()
        if pos_gap_mb <= 0:
            raise ValueError("pos_gap_mb must be positive: it marks the boundary between chromosomes")
        self.bin_size = bin_size
        self.max_chunk = max_chunk
        self.n_bins = max_chunk // bin_size
        self.pos_gap_mb = float(pos_gap_mb)

        # Learnable affine for masked InstanceNorm
        self.in_scale = nn.Parameter(torch.ones(1))
        self.in_bias = nn.Parameter(torch.zeros(1))

        # Conv binning at input
        self.binner = nn.Sequential(
            nn.Conv1d(1, d_conv, kernel_size=bin_size, stride=bin_size),
            nn.GELU(),
        )

        self.backbone = MaskedConvStemTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_ff=dim_ff,
            dropout=dropout,
            input_dim=d_conv,
            conv_kernel_size=min(conv_kernel_size, max(3, self.n_bins // 2 * 2 - 1)),
        )

        self.ratio_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, expression: torch.Tensor, lengths: torch.Tensor, gene_pos: torch.Tensor):
        """
        Args:
            expression: (B, max_chunk) raw log1p, zero-padded
            lengths:    (B,) real gene count
            gene_pos:   (B, max_chunk) each gene's genomic position in Mb

        Returns:
            bin_ratios: (B, n_bins) predicted ratio per bin
            bin_lengths: (B,) number of valid bins
        """
        B, S = expression.shape
        n_bins = S // self.bin_size

        # Masked InstanceNorm: per-sample normalize on real-gene prefix only (mean and unbiased std of each row's
        # first `length` values; rows of length <= 1 left as is)
        mask = torch.arange(S, device=expression.device).unsqueeze(0) < lengths.unsqueeze(1)
        mf = mask.float()
        L = mf.sum(dim=1)
        mean = (expression * mf).sum(dim=1) / L.clamp(min=1.0)
        xc = (expression - mean.unsqueeze(1)) * mf
        std = ((xc * xc).sum(dim=1) / (L - 1.0).clamp(min=1.0)).sqrt()
        x = xc / (std.unsqueeze(1) + 1e-5)
        small = lengths <= 1
        if small.any():
            x = torch.where(small.unsqueeze(1), expression, x)
        x = x * self.in_scale + self.in_bias
        x = x * mf

        # Conv binning: (B, 1, S) -> (B, d_conv, n_bins) -> (B, n_bins, d_conv)
        x = x.unsqueeze(1)
        x = self.binner(x)
        x = x.transpose(1, 2)

        # Bin-level mask
        bin_lengths = (lengths.float() / self.bin_size).ceil().long().clamp(min=1, max=n_bins)
        bin_mask = torch.arange(n_bins, device=x.device).unsqueeze(0) < bin_lengths.unsqueeze(1)

        pos = bin_positions(gene_pos, lengths, self.bin_size)
        seg = bin_segments(pos, bin_mask, self.pos_gap_mb)
        pad = self.binner[1](self.binner[0].bias)             # the feature of an empty (padded) bin
        out = self.backbone(x, pos, attn_mask=bin_mask, seg=seg, pad_feat=pad)
        bin_ratios = self.ratio_head(out).squeeze(-1)

        return bin_ratios, bin_lengths

    def predict_per_gene(self, expression: torch.Tensor, lengths: torch.Tensor, gene_pos: torch.Tensor):
        """Predict per-gene ratios by broadcasting bin predictions.

        Returns (B, max_chunk) where each gene gets its bin's prediction.
        """
        bin_ratios, bin_lengths = self.forward(expression, lengths, gene_pos)
        # Repeat each bin prediction for bin_size genes
        per_gene = bin_ratios.unsqueeze(2).expand(-1, -1, self.bin_size)
        per_gene = per_gene.reshape(bin_ratios.shape[0], -1)
        # Pad if max_chunk > n_bins * bin_size
        B, G = per_gene.shape
        if G < self.max_chunk:
            pad = torch.zeros(B, self.max_chunk - G, device=per_gene.device)
            per_gene = torch.cat([per_gene, pad], dim=1)
        return per_gene[:, :self.max_chunk]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: dict, dropout: float | None = None) -> BinnedDevModel:
    """The model described by a checkpoint's or a training run's config."""
    return BinnedDevModel(
        bin_size=cfg["bin_size"], max_chunk=cfg["max_chunk"],
        d_model=cfg.get("d_model", 128), n_heads=cfg.get("n_heads", 4),
        n_layers=cfg.get("n_layers", 4), dim_ff=cfg.get("dim_ff", 512),
        dropout=cfg.get("dropout", 0.1) if dropout is None else dropout,
        conv_kernel_size=cfg.get("conv_kernel", 7),
        pos_gap_mb=float(cfg.get("pos_gap_mb", 1000.0)),
    )


if __name__ == "__main__":
    m = BinnedDevModel()
    lengths = torch.tensor([2048, 1000])
    expr = torch.randn(2, 2048)
    pos = torch.cumsum(torch.rand(2, 2048), dim=1) / 10
    pos[:, 600:] += 1000.0                                      # a chromosome boundary at gene 600
    br, bl = m(expr, lengths, pos)
    print(f"params={m.count_parameters():,}, bin_ratios={br.shape}")
    print("Smoke test passed.")
