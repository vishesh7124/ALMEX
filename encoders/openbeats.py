"""
encoders/openbeats.py

Standalone OpenBEATs encoder wrapper.
BEATs.py and its dependencies must be in the parent directory (audio_lm/).
"""

import os
import sys
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download, list_repo_files


def _load_beats_module():
    """
    BEATs.py uses bare imports (from backbone import ...)
    so its directory must be in sys.path when loaded.
    """
    beats_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    beats_py  = os.path.join(beats_dir, "BEATs.py")

    if not os.path.exists(beats_py):
        raise FileNotFoundError(
            f"\nBEATs.py not found in {beats_dir}\n"
            "Download with:\n"
            "  wget https://raw.githubusercontent.com/microsoft/unilm/master/beats/BEATs.py\n"
            "  wget https://raw.githubusercontent.com/microsoft/unilm/master/beats/backbone.py\n"
            "  wget https://raw.githubusercontent.com/microsoft/unilm/master/beats/modules.py\n"
            "  wget https://raw.githubusercontent.com/microsoft/unilm/master/beats/Tokenizers.py\n"
            "  wget https://raw.githubusercontent.com/microsoft/unilm/master/beats/quantizer.py\n"
        )

    inserted = beats_dir not in sys.path
    if inserted:
        sys.path.insert(0, beats_dir)
    try:
        spec   = importlib.util.spec_from_file_location("BEATs_ms", beats_py)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.BEATs, module.BEATsConfig
    finally:
        if inserted and beats_dir in sys.path:
            sys.path.remove(beats_dir)


class OpenBEATsEncoder(nn.Module):
    """
    Input:  (B, samples) waveform at 16kHz
    Output: {
        'embedding':       (B, N+1, 1024)  CLS at [0], patches at [1:]
        'clipwise_output': (B, 1024)        CLS token
    }

    Patch math for 10s audio (160000 samples):
        fbank: (1024 time frames, 128 mel bins)
        BEATs Conv2d kernel=(16,16), stride=(10,16)
        time patches:  floor((1024-16)/10) + 1 = 101
        freq patches:  floor((128-16)/16)  + 1 = 8
        N = 808 patches → embedding shape = (B, 809, 1024)
    """

    SAMPLE_RATE   = 16000
    TARGET_LENGTH = 1024
    NUM_MEL_BINS  = 128
    FRAME_SHIFT   = 10     # ms

    def __init__(
        self,
        model_name:  str  = "shikhar7ssu/OpenBEATs-ICME",
        hidden_size: int  = 1024,
        freeze:      bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        print(f"[OpenBEATsEncoder] Loading {model_name} (hidden_size={hidden_size})")
        self.beats = self._load(model_name, hidden_size)
        print("[OpenBEATsEncoder] Loaded.")

        if freeze:
            for p in self.beats.parameters():
                p.requires_grad_(False)
            self.beats.eval()
            print("[OpenBEATsEncoder] Frozen.")

    @staticmethod
    def _find_ckpt(model_name):
        files = list(list_repo_files(model_name))
        cands = [f for f in files if f.endswith(".pt") or f.endswith(".pth")]
        if not cands:
            raise FileNotFoundError(f"No checkpoint found in {model_name}. Files: {files}")
        def score(f):
            if "valid.acc.ave" in f: return 0
            if "valid.loss.best" in f: return 1
            if "epoch_latest" in f: return 2
            if f.endswith(".pth"): return 3
            return 4
        best = sorted(cands, key=score)[0]
        print(f"[OpenBEATsEncoder] Checkpoint: {best}")
        return hf_hub_download(repo_id=model_name, filename=best)

    def _load(self, model_name, hidden_size):
        ckpt_path  = self._find_ckpt(model_name)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print(f"[OpenBEATsEncoder] Keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}")

        BEATs, BEATsConfig = _load_beats_module()

        # Build config manually — never pass checkpoint['cfg'] directly.
        # BEATsConfig defaults input_patch_size=-1 → Conv2d crash.
        ckpt_cfg = {}
        if isinstance(checkpoint, dict) and 'cfg' in checkpoint:
            raw = checkpoint['cfg']
            ckpt_cfg = raw if isinstance(raw, dict) else {}

        cfg = BEATsConfig({
            'input_patch_size':        16,
            'embed_dim':               512,
            'conv_bias':               False,
            'encoder_embed_dim':       ckpt_cfg.get('encoder_embed_dim',       hidden_size),
            'encoder_layers':          ckpt_cfg.get('encoder_layers',          24 if hidden_size==1024 else 12),
            'encoder_attention_heads': ckpt_cfg.get('encoder_attention_heads', 16 if hidden_size==1024 else 12),
            'encoder_ffn_embed_dim':   ckpt_cfg.get('encoder_ffn_embed_dim',   hidden_size * 4),
        })
        print(f"[OpenBEATsEncoder] Config: embed_dim={cfg.encoder_embed_dim}, "
              f"layers={cfg.encoder_layers}, patch_size={cfg.input_patch_size}")

        model  = BEATs(cfg)
        model.patch_embedding.stride = (10, 16)
        state  = checkpoint.get('model', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        miss, unex = model.load_state_dict(state, strict=False)
        print(f"[OpenBEATsEncoder] {len(miss)} missing, {len(unex)} unexpected keys")
        return model

    def _fbank(self, waveform: torch.Tensor) -> torch.Tensor:
        device, batch = waveform.device, []
        for i in range(waveform.shape[0]):
            fb = torchaudio.compliance.kaldi.fbank(
                waveform[i].unsqueeze(0).cpu(),
                htk_compat=True, sample_frequency=self.SAMPLE_RATE,
                use_energy=False, window_type='hanning',
                num_mel_bins=self.NUM_MEL_BINS, dither=0.0,
                frame_shift=self.FRAME_SHIFT,
            )
            T  = fb.shape[0]
            fb = F.pad(fb, (0,0,0,self.TARGET_LENGTH-T)) if T < self.TARGET_LENGTH else fb[:self.TARGET_LENGTH]
            fb = fb - fb.mean()
            batch.append(fb)
        return torch.stack(batch).to(device)   # (B, TARGET_LENGTH, 128)

    def forward(self, waveform: torch.Tensor) -> dict:
        fbank = self._fbank(waveform)
        mask  = torch.zeros(fbank.shape[0], fbank.shape[1], dtype=torch.bool, device=fbank.device)
        is_frozen = not any(p.requires_grad for p in self.beats.parameters())
        with torch.set_grad_enabled(self.training and not is_frozen):
            out, _ = self.beats.extract_features_from_fbank(fbank, padding_mask=mask)

        cls = out.mean(dim=1, keepdim=True)
        emb = torch.cat([cls, out], dim=1)

        return {'embedding': emb, 'clipwise_output': emb[:, 0, :]}