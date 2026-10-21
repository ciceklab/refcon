"""Backbone for the binned CN-ratio model.

Provides the mask-aware Transformer stack (`MaskedConvStemTransformer`)
used by `model_binned_dev.BinnedDevModel`. Self-contained; only depends
on PyTorch.

  - Rotary positions are given per token (`pos`): the genomic position of
    each bin in megabases.
  - Chromosome separation (`seg`): in a window that spans two chromosomes,
    attention stays within a chromosome and the convolution stem runs on
    each chromosome as if it were a window of its own.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================== #
# Rotary Position Embedding
# ====================================================================== #

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def at(self, pos: torch.Tensor):
        """cos and sin at the given positions, one row per sample: pos (B, S) -> (B, 1, S, dim / 2) each."""
        freqs = pos.float().unsqueeze(-1) * self.inv_freq
        return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.repeat(1, 1, 1, 2)
    sin = sin.repeat(1, 1, 1, 2)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


# ====================================================================== #
# Mask-Aware Attention
# ====================================================================== #

class MaskedGenomicSelfAttention(nn.Module):
    """Multi-head self-attention with rotary positions and an optional padding mask."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, attn_mask: torch.Tensor | None = None,
                seg: torch.Tensor | None = None):
        """
        x:         (B, S, d_model)
        pos:       (B, S) float - rotary position of each token (the bin's position in Mb).
        attn_mask: (B, S) bool - True = valid, False = padded. None = no masking.
        seg:       (B, S) long - chromosome of each token within its window (0, 1; -1 = padding). A token attends
                   only to tokens of its own chromosome (padded tokens to those of the first). None = no restriction.
        """
        B, S, _ = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        cos, sin = self.rope.at(pos)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        sdpa_mask = None
        if attn_mask is not None:
            sdpa_mask = attn_mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, S, -1)
        if seg is not None:
            sq = torch.where(seg >= 0, seg, torch.zeros_like(seg))
            same = (sq.unsqueeze(2) == seg.unsqueeze(1)).unsqueeze(1)
            sdpa_mask = same if sdpa_mask is None else sdpa_mask & same

        drop = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask, dropout_p=drop)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, self.d_model)
        return self.out_proj(attn_out)


# ====================================================================== #
# Transformer Block
# ====================================================================== #

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with optional attention mask."""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MaskedGenomicSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor, attn_mask: torch.Tensor | None = None,
                seg: torch.Tensor | None = None):
        x = x + self.attn(self.ln1(x), pos, attn_mask=attn_mask, seg=seg)
        x = x + self.ffn(self.ln2(x))
        return x


# ====================================================================== #
# Conv Stem + Transformer Backbone (mask-aware)
# ====================================================================== #

class MaskedConvStemTransformer(nn.Module):
    """Conv1d stem + Transformer stack with attention mask support."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 4,
                 dim_ff: int = 512, dropout: float = 0.1, input_dim: int = 1,
                 conv_kernel_size: int = 11):
        super().__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=conv_kernel_size,
                      padding=conv_kernel_size // 2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size,
                      padding=conv_kernel_size // 2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.post_stem_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dim_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, features: torch.Tensor, pos: torch.Tensor, attn_mask: torch.Tensor | None = None,
                seg: torch.Tensor | None = None, pad_feat: torch.Tensor | None = None):
        """
        features:  (B, S, input_dim)
        pos:       (B, S) rotary position of each token, see MaskedGenomicSelfAttention
        attn_mask: (B, S) bool - True = valid, False = padded
        seg:       see MaskedGenomicSelfAttention
        pad_feat:  (input_dim,) the feature of an empty bin; needed with seg (chromosome separation)
        """
        if seg is not None and bool((seg > 0).any()):
            x = self._stem_by_chromosome(features, seg, pad_feat)
        else:
            x = features.transpose(1, 2)       # (B, input_dim, S)
            x = self.conv_stem(x)              # (B, d_model, S)
            x = x.transpose(1, 2)              # (B, S, d_model)
        x = self.post_stem_norm(x)
        for block in self.blocks:
            x = block(x, pos, attn_mask=attn_mask, seg=seg)
        return self.final_norm(x)

    def _stem_by_chromosome(self, features, seg, pad_feat):
        """The convolution stem run on each chromosome of a window as if it were a window of its own: the first
        chromosome followed by empty bins, and the second moved to the front and followed by empty bins (an empty
        bin's feature is what the binner gives a padded bin). One pass over all views, so in training the batch
        statistics of the stem's batch norm are computed over all of them together."""
        if pad_feat is None:
            raise ValueError("chromosome separation needs pad_feat, the feature of an empty bin")
        if int(seg.max()) > 1:
            raise ValueError("a window spans at most two chromosomes")
        B, S, D = features.shape
        c = pad_feat.view(1, 1, D).expand(B, S, D)
        view0 = torch.where((seg == 1).unsqueeze(-1), c, features)
        two = seg.max(dim=1).values >= 1
        f1, is1 = features[two], seg[two] == 1
        start = is1.float().argmax(dim=1)
        j = torch.arange(S, device=features.device).unsqueeze(0)
        src = (start.unsqueeze(1) + j).clamp(max=S - 1)
        view1 = torch.gather(f1, 1, src.unsqueeze(-1).expand(-1, -1, D))
        view1 = torch.where((j < is1.sum(dim=1, keepdim=True)).unsqueeze(-1), view1, c[: view1.shape[0]])
        out = self.conv_stem(torch.cat([view0, view1]).transpose(1, 2)).transpose(1, 2)
        o0, o1 = out[:B], out[B:]
        back = (j - start.unsqueeze(1)).clamp(min=0, max=S - 1)
        o1b = torch.gather(o1, 1, back.unsqueeze(-1).expand(-1, -1, o1.shape[-1]))
        res = o0.clone()
        res[two] = torch.where(is1.unsqueeze(-1), o1b, o0[two])
        return res


if __name__ == "__main__":
    # Smoke test: variable-length padded batch through the backbone.
    B, S, D = 4, 32, 64
    lengths = torch.tensor([32, 20, 10, 28])
    x = torch.randn(B, S, D)
    mask = torch.arange(S).unsqueeze(0) < lengths.unsqueeze(1)
    pos = torch.cumsum(torch.rand(B, S), dim=1)
    m = MaskedConvStemTransformer(d_model=128, input_dim=D, conv_kernel_size=7)
    out = m(x, pos, attn_mask=mask)
    print(f"backbone out: {out.shape}  params: {sum(p.numel() for p in m.parameters()):,}")
    print("smoke test passed.")
