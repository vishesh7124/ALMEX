"""
mappers/speech_mapper.py

Mapper from Whisper encoder output → SmolLM2 embedding space.

Same two-stage design as AudioMapper (MLP-Mixer principle):
  Stage 1 — Token-mixing:   DepthwiseSepConv1d (temporal compression)
  Stage 2 — Channel-mixing: ExpandContractMLP  (feature projection)

Whisper always outputs 1500 frames for any input (padded to 30s).
We compress these with depthwise conv before projecting.

Stride choices and resulting token counts:
    stride=2 → 750 tokens  (high resolution, more VRAM)
    stride=4 → 375 tokens  (recommended — good balance)
    stride=8 → 188 tokens  (if VRAM is very tight)

With stride=4 and whisper-small (768-dim):
    Input:  (B, 1500, 768)
    Output: (B, 375, 896)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the same building blocks from audio_mapper
from mappers.audio_mapper import DepthwiseSepConv1d, ExpandContractMLP


class SpeechMapper(nn.Module):
    """
    Maps Whisper encoder output → SmolLM2 embedding space.

    Two-stage design (same principle as AudioMapper):
        Stage 1 — Token-mixing:   DepthwiseSepConv1d
        Stage 2 — Channel-mixing: ExpandContractMLP

    Token counts (whisper-small, 1500 input frames):
        stride=4 → 375 tokens per 30s Whisper output
    """

    def __init__(
        self,
        encoder_dim:  int   = 768,    # whisper-small; 384=tiny, 512=base, 1280=large-v3
        lm_dim:       int   = 896,    # Qwen2.5-0.5B embedding size
        conv_stride:  int   = 4,
        conv_kernel:  int   = 5,      # smaller kernel than AudioMapper — speech is denser
        expand_ratio: float = 2.0,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.lm_dim      = lm_dim
        self.conv_stride = conv_stride

        # Stage 1: token-mixing
        self.token_mix = DepthwiseSepConv1d(
            channels    = encoder_dim,
            kernel_size = conv_kernel,
            stride      = conv_stride,
            padding     = conv_kernel // 2,
        )

        # Stage 2: channel-mixing
        self.channel_mix = ExpandContractMLP(
            d_in         = encoder_dim,
            d_out        = lm_dim,
            expand_ratio = expand_ratio,
            dropout      = dropout,
        )

        n = sum(p.numel() for p in self.parameters())
        print(f"[SpeechMapper] Parameters: {n:,}")
        print(f"[SpeechMapper] Tokens per 30s (1500 frames, stride={conv_stride}): "
              f"{1500 // conv_stride}")

    @property
    def tokens_per_clip(self) -> int:
        return 1500 // self.conv_stride   # 375 for stride=4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1500, encoder_dim)
               For longer audio from SlidingWindowExtractor:
               (B, M*1500, encoder_dim) where M = number of 30s chunks

        Returns:
            (B, 1500//stride * M, lm_dim)
        """
        # Stage 1: token-mixing
        x = self.token_mix(x)      # (B, 1500//stride, encoder_dim)

        # Stage 2: channel-mixing
        x = self.channel_mix(x)    # (B, 1500//stride, lm_dim)

        return x