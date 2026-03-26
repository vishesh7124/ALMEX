import os
import shutil
import subprocess

import torch
import torch.nn.functional as F


def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.dim() == 1:
        return waveform
    if waveform.dim() == 2:
        if waveform.shape[0] == 1:
            return waveform.squeeze(0)
        return waveform.mean(dim=0)
    raise ValueError(f"Unsupported waveform shape: {tuple(waveform.shape)}")


def _resample_if_needed(waveform: torch.Tensor, sample_rate: int, target_sample_rate: int) -> torch.Tensor:
    if sample_rate == target_sample_rate:
        return waveform
    import torchaudio

    return torchaudio.functional.resample(
        waveform.unsqueeze(0), sample_rate, target_sample_rate
    ).squeeze(0)


def _pad_or_trim(waveform: torch.Tensor, target_samples: int) -> torch.Tensor:
    waveform = waveform[:target_samples]
    if waveform.shape[0] < target_samples:
        waveform = F.pad(waveform, (0, target_samples - waveform.shape[0]))
    return waveform


def _load_with_torchaudio(path: str):
    import torchaudio

    wav, sr = torchaudio.load(path)
    return _to_mono(wav).to(torch.float32), int(sr), "torchaudio"


def _load_with_soundfile(path: str):
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav_t = torch.from_numpy(wav).transpose(0, 1)
    return _to_mono(wav_t).to(torch.float32), int(sr), "soundfile"


def _load_with_ffmpeg(path: str, target_sample_rate: int):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or not on PATH")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(target_sample_rate),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg decode failed: {stderr.strip()}")

    if len(proc.stdout) == 0:
        raise RuntimeError("ffmpeg returned empty audio stream")

    wav = torch.frombuffer(proc.stdout, dtype=torch.float32).clone()
    return wav, target_sample_rate, "ffmpeg"


def load_audio_10s_mono16k(path: str, target_sample_rate: int = 16000, target_samples: int = 160000):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    errors = []

    for loader in (_load_with_torchaudio, _load_with_soundfile):
        try:
            wav, sr, backend = loader(path)
            wav = _resample_if_needed(wav, sr, target_sample_rate)
            wav = _pad_or_trim(wav, target_samples)
            return wav.unsqueeze(0), backend
        except Exception as exc:
            detail = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
            errors.append(f"{loader.__name__}: {detail}")

    try:
        wav, sr, backend = _load_with_ffmpeg(path, target_sample_rate=target_sample_rate)
        wav = _pad_or_trim(wav, target_samples)
        return wav.unsqueeze(0), backend
    except Exception as exc:
        detail = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
        errors.append(f"_load_with_ffmpeg: {detail}")

    msg = "Could not load audio file with available backends.\n" + "\n".join(errors)
    raise RuntimeError(msg)