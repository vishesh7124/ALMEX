"""
mappers/audio_mapper.py

Mapper from OpenBEATs encoder output → SmolLM2 embedding space.

Design is grounded in three sources:
  1. Depthwise Separable Conv  → cheap token-mixing (temporal compression)
  2. MLP-Mixer paper           → token-mixing and channel-mixing are separate ops
  3. LLM MLP article           → expand-then-contract with GELU creates sparsity
                                  and richer linear separability

Architecture:
  Input: (B, 809, 1024)   [CLS + 808 patches from OpenBEATs-ICME]
  ↓
  Separate CLS token
  ↓
  TOKEN-MIXING: DepthwiseSepConv1d, stride=4
    → (B, 202, 1024)   temporal compression, learns what to compress
  ↓
  CHANNEL-MIXING: Expand → GELU → Contract
    Linear(1024 → 2048)   expand 2× (richer feature space)
    GELU                  sparsity: suppresses irrelevant features
    Linear(2048 → 896)    contract to Qwen2.5-0.5B embedding dim
    LayerNorm
    → (B, 202, 896)
  ↓
  Prepend projected CLS: (B, 203, 896)

Token count: 203 per 10s clip
Prefix contribution per audio: 203 tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSepConv1d(nn.Module):
    """
    1D Depthwise Separable Convolution.
    Implements token-mixing from MLP-Mixer: operates on temporal positions
    independently per channel (depthwise), then mixes channels (pointwise).

    Cost vs standard Conv1d:
        Standard:  C_in × C_out × kernel × T  operations
        Depthwise: C_in × kernel × T  +  C_in × C_out × T
        Saving:    ~kernel× cheaper (7× for kernel=7)
    """

    def __init__(self, channels: int, kernel_size: int = 7,
                 stride: int = 4, padding: int = 3):
        super().__init__()
        # Depthwise: one filter per channel — pure temporal mixing
        self.depthwise = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, groups=channels, bias=False,
        )
        # Pointwise: 1×1 conv — pure channel mixing
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(channels)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = x.transpose(1, 2)       # (B, C, T) — Conv1d needs channel-first
        x = self.depthwise(x)        # (B, C, T//stride)
        x = self.pointwise(x)        # (B, C, T//stride)
        x = x.transpose(1, 2)       # (B, T//stride, C)
        x = self.norm(x)
        x = self.act(x)
        return x


class ExpandContractMLP(nn.Module):
    """
    Expand → GELU → Contract projection.

    From the LLM MLP article: expanding dimensionality before contracting
    creates a richer space for linear separation. GELU suppresses most
    activations (sparsity), letting only relevant features pass through.
    This is the same pattern used inside every transformer block.

    expand_ratio=2 means: 1024 → 2048 → 896
    (we don't do 4× because we're going encoder_dim→lm_dim, not d→d)
    """

    def __init__(self, d_in: int, d_out: int,
                 expand_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        d_mid = int(d_in * expand_ratio)
        self.expand   = nn.Linear(d_in,  d_mid)
        self.act      = nn.GELU()
        self.drop     = nn.Dropout(dropout)
        self.contract = nn.Linear(d_mid, d_out)
        self.norm     = nn.LayerNorm(d_out)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)     # d_in → d_mid  (expand into richer space)
        x = self.act(x)        # GELU: sparsity, suppresses irrelevant features
        x = self.drop(x)
        x = self.contract(x)   # d_mid → d_out (contract to target dim)
        x = self.norm(x)
        return x


class AudioMapper(nn.Module):
    """
    Maps OpenBEATs-ICME output → SmolLM2 embedding space.

    Two-stage design (MLP-Mixer principle):
        Stage 1 — Token-mixing:   DepthwiseSepConv1d (temporal compression)
        Stage 2 — Channel-mixing: ExpandContractMLP  (feature projection)

    Output tokens per 10s clip: 203 (1 CLS + 202 compressed patches)

    Sliding window: for audio > 10s, pass concatenated chunks from
    SlidingWindowExtractor — this mapper handles them automatically.
    """

    TOTAL_PATCHES = 808   # 101 time × 8 freq patches from OpenBEATs for 10s

    def __init__(
        self,
        encoder_dim:  int   = 1024,   # OpenBEATs-ICME hidden size
        lm_dim:       int   = 896,    # Qwen2.5-0.5B embedding size
        conv_stride:  int   = 4,      # 808 patches → 202 compressed tokens
        conv_kernel:  int   = 7,      # receptive field (7 patches ≈ 91ms)
        expand_ratio: float = 2.0,    # MLP expand: 1024 → 2048 → 896
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.lm_dim      = lm_dim
        self.conv_stride = conv_stride

        # Stage 1: Token-mixing (temporal compression)
        self.token_mix = DepthwiseSepConv1d(
            channels    = encoder_dim,
            kernel_size = conv_kernel,
            stride      = conv_stride,
            padding     = conv_kernel // 2,
        )

        # Stage 2: Channel-mixing for patch tokens
        self.channel_mix = ExpandContractMLP(
            d_in         = encoder_dim,
            d_out        = lm_dim,
            expand_ratio = expand_ratio,
            dropout      = dropout,
        )

        # CLS token gets its own projection (different statistical properties)
        self.cls_proj = ExpandContractMLP(
            d_in         = encoder_dim,
            d_out        = lm_dim,
            expand_ratio = expand_ratio,
            dropout      = dropout,
        )

        n = sum(p.numel() for p in self.parameters())
        print(f"[AudioMapper] Parameters: {n:,}")
        print(f"[AudioMapper] Tokens per 10s clip: {self.tokens_per_clip}")

    @property
    def tokens_per_clip(self) -> int:
        """CLS (1) + compressed patches (808//stride)."""
        return 1 + (self.TOTAL_PATCHES // self.conv_stride)   # 1 + 202 = 203

    def _process_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        """Process one 10s chunk: (B, 809, 1024) → (B, 203, 896)"""
        cls     = chunk[:, :1, :]    # (B, 1, 1024)
        patches = chunk[:, 1:, :]    # (B, 808, 1024)

        if patches.shape[1] == 0:
            return self.cls_proj(cls)

        # Stage 1: token-mixing — compress 808 temporal patches to 202
        patches = self.token_mix(patches)          # (B, 202, 1024)

        # Stage 2: channel-mixing — project 1024 → 576
        patches = self.channel_mix(patches)        # (B, 202, 896)
        cls_out = self.cls_proj(cls)               # (B, 1, 576)

        return torch.cat([cls_out, patches], dim=1)  # (B, 203, 896)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, encoder_dim)
               N = 809           for single 10s clip
               N = 809 * chunks  for longer audio (from SlidingWindowExtractor)

        Returns:
            (B, 203 * chunks, lm_dim)
        """
        tokens_per_chunk = self.TOTAL_PATCHES + 1  # 809

        if x.shape[1] <= tokens_per_chunk:
            return self._process_chunk(x)

        chunks  = x.split(tokens_per_chunk, dim=1)
        outputs = [self._process_chunk(c) for c in chunks]
        return torch.cat(outputs, dim=1)