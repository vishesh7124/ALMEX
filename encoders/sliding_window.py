"""
encoders/sliding_window.py

Handles audio longer than each encoder's context window.

    OpenBEATs context: 10s  (160000 samples at 16kHz)
    Whisper context:   30s  (480000 samples at 16kHz)

Strategy: non-overlapping windows, concatenate features along token dim.
This is exactly what AF3 does. Simple and effective — the LLM learns
to attend across chunk boundaries during training.

Usage (standalone):
    from encoders.sliding_window import SlidingWindowExtractor
    extractor = SlidingWindowExtractor(beats_enc, whisper_enc)
    beats_out, whisper_out = extractor(waveform)   # waveform: (B, samples)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlidingWindowExtractor(nn.Module):
    """
    Runs both encoders on arbitrarily long audio using non-overlapping windows.

    OpenBEATs:  chunks of 10s → concatenate (B, N*809, 1024)
    Whisper:    chunks of 30s → concatenate (B, M*1500, 768)

    where N = ceil(duration / 10)
          M = ceil(duration / 30)

    For typical DRDO audio (10-60s):
        10s: beats=[1, 809, 1024]  whisper=[1, 1500, 768]   ← single chunk each
        30s: beats=[1, 2427, 1024] whisper=[1, 1500, 768]   ← 3 beats, 1 whisper
        60s: beats=[1, 4854, 1024] whisper=[1, 3000, 768]   ← 6 beats, 2 whisper
    """

    BEATS_CHUNK_SAMPLES   = 160000   # 10s at 16kHz
    WHISPER_CHUNK_SAMPLES = 480000   # 30s at 16kHz

    def __init__(self, beats_encoder, whisper_encoder):
        super().__init__()
        # Not registered as submodules — encoders are already instantiated
        # and owned elsewhere. This class just orchestrates them.
        self.beats_encoder   = beats_encoder
        self.whisper_encoder = whisper_encoder

    @staticmethod
    def _split_into_chunks(waveform: torch.Tensor, chunk_size: int):
        """
        Split (B, total_samples) into list of (B, chunk_size) tensors.
        Last chunk is zero-padded if shorter than chunk_size.
        """
        total   = waveform.shape[1]
        chunks  = []
        for start in range(0, max(total, 1), chunk_size):
            end   = start + chunk_size
            chunk = waveform[:, start:end]
            if chunk.shape[1] < chunk_size:
                chunk = F.pad(chunk, (0, chunk_size - chunk.shape[1]))
            chunks.append(chunk)
        return chunks

    def extract_beats(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Run OpenBEATs on all 10s chunks.

        Args:
            waveform: (B, samples) at 16kHz

        Returns:
            (B, N_chunks * 809, 1024)
        """
        chunks  = self._split_into_chunks(waveform, self.BEATS_CHUNK_SAMPLES)
        outputs = []
        for chunk in chunks:
            out = self.beats_encoder(chunk)       # calls OpenBEATsEncoder.forward()
            outputs.append(out['embedding'])      # (B, 809, 1024)
        return torch.cat(outputs, dim=1)          # (B, N*809, 1024)

    def extract_whisper(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Run Whisper on all 30s chunks.

        Args:
            waveform: (B, samples) at 16kHz

        Returns:
            (B, M_chunks * 1500, hidden_size)
        """
        chunks  = self._split_into_chunks(waveform, self.WHISPER_CHUNK_SAMPLES)
        outputs = []
        for chunk in chunks:
            out = self.whisper_encoder(chunk)     # (B, 1500, hidden_size)
            outputs.append(out)
        return torch.cat(outputs, dim=1)          # (B, M*1500, hidden_size)

    def forward(self, waveform: torch.Tensor):
        """
        Args:
            waveform: (B, samples) at 16kHz — any length

        Returns:
            tuple:
                beats_features:   (B, N_beats_chunks * 809, 1024)
                whisper_features: (B, M_whisper_chunks * 1500, hidden_size)
        """
        beats_out   = self.extract_beats(waveform)
        whisper_out = self.extract_whisper(waveform)
        return beats_out, whisper_out

    def describe(self, duration_sec: float) -> dict:
        """
        Returns token counts for a given audio duration.
        Useful for computing prefix_length before training.
        """
        import math
        n_beats   = math.ceil(duration_sec / 10)
        n_whisper = math.ceil(duration_sec / 30)
        return {
            'duration_sec':      duration_sec,
            'beats_chunks':      n_beats,
            'whisper_chunks':    n_whisper,
            'beats_tokens':      n_beats   * 809,
            'whisper_tokens':    n_whisper * 1500,
        }