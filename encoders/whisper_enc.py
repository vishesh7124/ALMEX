"""
encoders/whisper_enc.py

Whisper encoder wrapper for the speech branch.

Recommended for your setup:
    "openai/whisper-tiny"   384-dim,  37M params
    "openai/whisper-base"   512-dim,  74M params
    "openai/whisper-small"  768-dim, 244M params  ← recommended
    "openai/whisper-medium" 1024-dim, 769M params
    "openai/whisper-large-v3" 1280-dim, 1540M params

Input:  (B, samples) waveform at 16kHz
Output: (B, 1500, d_model)
    1500 = Whisper's fixed output frames for any input (padded/trimmed to 30s)
    d_model = 768 for small, 384 for tiny, 512 for base, 1280 for large-v3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel, WhisperFeatureExtractor


class WhisperEncoder(nn.Module):

    SAMPLE_RATE = 16000
    MAX_SAMPLES = 480000   # 30s — Whisper's context window

    def __init__(
        self,
        model_name: str  = "openai/whisper-small",
        freeze:     bool = True,
    ):
        super().__init__()
        print(f"[WhisperEncoder] Loading {model_name}")

        # Load full model then keep only the encoder
        full_model             = WhisperModel.from_pretrained(model_name)
        self.encoder           = full_model.encoder
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.hidden_size       = self.encoder.config.d_model
        del full_model

        print(f"[WhisperEncoder] hidden_size={self.hidden_size}")

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()
            print("[WhisperEncoder] Frozen.")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, samples) float32 at 16kHz
                      Any length — will be padded/trimmed to 30s internally.

        Returns:
            (B, 1500, hidden_size)
            Always 1500 frames regardless of input length.
        """
        device = waveform.device

        # Pad or trim to exactly 30s
        T = waveform.shape[1]
        if T < self.MAX_SAMPLES:
            waveform = F.pad(waveform, (0, self.MAX_SAMPLES - T))
        else:
            waveform = waveform[:, :self.MAX_SAMPLES]

        # WhisperFeatureExtractor is not batched — process each sample
        mels = []
        for i in range(waveform.shape[0]):
            mel = self.feature_extractor(
                waveform[i].cpu().numpy(),
                sampling_rate=self.SAMPLE_RATE,
                return_tensors="pt",
            ).input_features.squeeze(0)   # (80, 3000)
            mels.append(mel)
        mel_batch = torch.stack(mels).to(device)   # (B, 80, 3000)

        is_frozen = not any(p.requires_grad for p in self.encoder.parameters())
        with torch.set_grad_enabled(self.training and not is_frozen):
            out = self.encoder(input_features=mel_batch, return_dict=True)

        return out.last_hidden_state   # (B, 1500, hidden_size)