"""
test_mappers.py
Run from audio_lm/ directory:
    python test_mappers.py
    python test_mappers.py --audio test_files/7-cmp.wav

Tests ONLY the mappers (encoders are run to produce real inputs).
"""

import sys, os, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LINE = "=" * 55


def get_encoder_outputs(audio_path=None):
    """Run both encoders to get real inputs for the mappers."""
    from encoders.openbeats import OpenBEATsEncoder
    from encoders.whisper_enc import WhisperEncoder

    beats_enc   = OpenBEATsEncoder(freeze=True)
    whisper_enc = WhisperEncoder(freeze=True)
    beats_enc.eval()
    whisper_enc.eval()

    if audio_path:
        from utils.audio_loader import load_audio_10s_mono16k
        waveform, _ = load_audio_10s_mono16k(audio_path)
    else:
        waveform = torch.randn(1, 160000)

    with torch.no_grad():
        beats_out   = beats_enc(waveform)['embedding']   # (1, 809, 1024)
        whisper_out = whisper_enc(waveform)              # (1, 1500, 768)

    return beats_out, whisper_out, beats_enc, whisper_enc


def test_audio_mapper(beats_emb):
    print(f"\n{LINE}")
    print("  AudioMapper")
    print(LINE)

    from mappers.audio_mapper import AudioMapper
    mapper = AudioMapper(
        encoder_dim  = 1024,
        lm_dim       = 576,
        conv_stride  = 4,
        conv_kernel  = 7,
        expand_ratio = 2.0,
    )
    mapper.eval()

    with torch.no_grad():
        out = mapper(beats_emb)

    print(f"\n  Input  : {list(beats_emb.shape)}")
    print(f"  Output : {list(out.shape)}")
    print()
    print(f"  Expected: (1, 203, 576)")
    print(f"    1 CLS + 808//4=202 compressed patches = 203")
    print()

    # Verify internal structure
    print(f"  Internal breakdown:")
    print(f"    Stage 1 (token-mixing):   808 patches → 202  via DepthwiseSepConv1d")
    print(f"    Stage 2 (channel-mixing): 1024-dim    → 576  via ExpandContractMLP")
    print(f"    Expand ratio: 1024 → 2048 → 576  (GELU in between)")

    n_params = sum(p.numel() for p in mapper.parameters())
    print(f"\n  Total parameters: {n_params:,}")

    assert out.shape == (1, 203, 576), f"Unexpected shape: {out.shape}"
    print("  PASS ✓")
    return mapper


def test_speech_mapper(whisper_out):
    print(f"\n{LINE}")
    print("  SpeechMapper")
    print(LINE)

    from mappers.speech_mapper import SpeechMapper
    mapper = SpeechMapper(
        encoder_dim  = 768,
        lm_dim       = 576,
        conv_stride  = 4,
        conv_kernel  = 5,
        expand_ratio = 2.0,
    )
    mapper.eval()

    with torch.no_grad():
        out = mapper(whisper_out)

    print(f"\n  Input  : {list(whisper_out.shape)}")
    print(f"  Output : {list(out.shape)}")
    print()
    print(f"  Expected: (1, 375, 576)")
    print(f"    1500 Whisper frames // stride=4 = 375")
    print()
    print(f"  Internal breakdown:")
    print(f"    Stage 1 (token-mixing):   1500 frames → 375  via DepthwiseSepConv1d")
    print(f"    Stage 2 (channel-mixing): 768-dim     → 576  via ExpandContractMLP")
    print(f"    Expand ratio: 768 → 1536 → 576  (GELU in between)")

    n_params = sum(p.numel() for p in mapper.parameters())
    print(f"\n  Total parameters: {n_params:,}")

    assert out.shape == (1, 375, 576), f"Unexpected shape: {out.shape}"
    print("  PASS ✓")
    return mapper


def test_sliding_window_through_mappers(beats_enc, whisper_enc, audio_mapper, speech_mapper):
    print(f"\n{LINE}")
    print("  Sliding Window → Mapper (30s audio)")
    print(LINE)

    from encoders.sliding_window import SlidingWindowExtractor
    extractor = SlidingWindowExtractor(beats_enc, whisper_enc)

    waveform = torch.randn(1, 480000)   # 30s

    with torch.no_grad():
        beats_feats, whisper_feats = extractor(waveform)
        audio_mapped  = audio_mapper(beats_feats)
        speech_mapped = speech_mapper(whisper_feats)

    print(f"\n  30s audio:")
    print(f"    BEATs raw:       {list(beats_feats.shape)}    (3 chunks × 809)")
    print(f"    Whisper raw:     {list(whisper_feats.shape)}  (1 chunk × 1500)")
    print(f"    After AudioMapper:  {list(audio_mapped.shape)}  (3 × 203 = 609)")
    print(f"    After SpeechMapper: {list(speech_mapped.shape)} (1 × 375)")

    assert audio_mapped.shape  == (1, 609, 576), f"Audio: {audio_mapped.shape}"
    assert speech_mapped.shape == (1, 375, 576), f"Speech: {speech_mapped.shape}"
    print("  PASS ✓")


def test_prefix_length(audio_mapper, speech_mapper):
    print(f"\n{LINE}")
    print("  Prefix Length Summary")
    print(LINE)

    text_len   = 129   # text_tokenization_len from config
    a_tok      = audio_mapper.tokens_per_clip    # 203
    s_tok      = speech_mapper.tokens_per_clip   # 375

    # prefix = 2 audios × (sound + sep + speech + sep) + text
    prefix_len = 2 * (a_tok + 1 + s_tok + 1) + text_len

    print(f"\n  Per 10s audio clip:")
    print(f"    Sound tokens  (AudioMapper):  {a_tok}")
    print(f"    Speech tokens (SpeechMapper): {s_tok}")
    print()
    print(f"  Full prefix (2 clips):")
    print(f"    2 × ({a_tok} + 1 sep + {s_tok} + 1 sep) + {text_len} text")
    print(f"    = {prefix_len} tokens total")
    print()
    print(f"  → Set in config/v0.yaml:  prefix_length: {prefix_len}")
    print()
    print(f"  SmolLM2-135M context window: 8192 tokens → fits ✓")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=None)
    args = ap.parse_args()

    print("Loading encoders (needed to produce real inputs for mapper tests)...")
    beats_emb, whisper_out, beats_enc, whisper_enc = get_encoder_outputs(args.audio)

    audio_mapper  = test_audio_mapper(beats_emb)
    speech_mapper = test_speech_mapper(whisper_out)
    test_sliding_window_through_mappers(beats_enc, whisper_enc, audio_mapper, speech_mapper)
    test_prefix_length(audio_mapper, speech_mapper)

    print(f"\n{LINE}")
    print("  All mapper tests passed.")
    print("  Next: create model/dual_encoder_lm.py, then test_model.py")
    print(LINE + "\n")


if __name__ == "__main__":
    main()