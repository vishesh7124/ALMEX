"""
test_encoders.py
Run from audio_lm/ directory:
    python test_encoders.py
    python test_encoders.py --audio path/to/file.wav
    python test_encoders.py --audio path/to/file.wav --duration 30
"""

import sys, os, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_audio(path, duration_sec=10):
    from utils.audio_loader import load_audio_10s_mono16k
    target_samples = int(duration_sec * 16000)
    # audio_loader always returns 10s — reuse its loaders but with custom length
    # We load then pad/trim to the requested duration
    wav, backend = load_audio_10s_mono16k(path)   # (1, 160000)
    wav = wav.squeeze(0)   # (160000,)

    # Extend by repeating if longer duration requested
    if target_samples > wav.shape[0]:
        reps = (target_samples // wav.shape[0]) + 1
        wav  = wav.repeat(reps)
    wav = wav[:target_samples]
    return wav.unsqueeze(0), backend   # (1, target_samples)


def test_openbeats(waveform_10s):
    print(f"\n{'='*55}")
    print("  OpenBEATs Encoder")
    print(f"{'='*55}")

    from encoders.openbeats import OpenBEATsEncoder
    enc = OpenBEATsEncoder(model_name="shikhar7ssu/OpenBEATs-ICME",
                           hidden_size=1024, freeze=True)
    enc.eval()

    with torch.no_grad():
        out = enc(waveform_10s)

    emb = out['embedding']
    cls = out['clipwise_output']

    print(f"\n  Input  : {list(waveform_10s.shape)}  (B, samples)")
    print(f"  embedding       : {list(emb.shape)}")
    print(f"  clipwise_output : {list(cls.shape)}")
    N = emb.shape[1] - 1
    print(f"  N patches = {N}  (expect 808 = 101 time × 8 freq)")

    assert emb.shape == (1, 809, 1024), f"Unexpected: {emb.shape}"
    assert cls.shape == (1, 1024)
    print("  PASS ✓")
    return enc


def test_whisper(waveform_10s):
    print(f"\n{'='*55}")
    print("  Whisper Encoder")
    print(f"{'='*55}")

    from encoders.whisper_enc import WhisperEncoder
    enc = WhisperEncoder(model_name="openai/whisper-small", freeze=True)
    enc.eval()

    with torch.no_grad():
        out = enc(waveform_10s)

    print(f"\n  Input  : {list(waveform_10s.shape)}  (10s → padded to 30s inside Whisper)")
    print(f"  Output : {list(out.shape)}")
    print(f"  Always 1500 frames regardless of input length")

    assert out.shape == (1, 1500, enc.hidden_size), f"Unexpected: {out.shape}"
    print("  PASS ✓")
    return enc


def test_batch(beats_enc, whisper_enc):
    print(f"\n{'='*55}")
    print("  Batch size = 2 test")
    print(f"{'='*55}")
    batch = torch.randn(2, 160000)
    with torch.no_grad():
        b = beats_enc(batch)
        w = whisper_enc(batch)
    print(f"  OpenBEATs (B=2): {list(b['embedding'].shape)}")
    print(f"  Whisper   (B=2): {list(w.shape)}")
    assert b['embedding'].shape[0] == 2
    assert w.shape[0] == 2
    print("  PASS ✓")


def test_sliding_window(beats_enc, whisper_enc, duration_sec):
    print(f"\n{'='*55}")
    print(f"  Sliding Window — {duration_sec}s audio")
    print(f"{'='*55}")

    import math
    from encoders.sliding_window import SlidingWindowExtractor

    extractor = SlidingWindowExtractor(beats_enc, whisper_enc)

    waveform = torch.randn(1, int(duration_sec * 16000))

    with torch.no_grad():
        beats_out, whisper_out = extractor(waveform)

    n_beats   = math.ceil(duration_sec / 10)
    n_whisper = math.ceil(duration_sec / 30)
    expected_beats_tokens   = n_beats   * 809
    expected_whisper_tokens = n_whisper * 1500

    print(f"\n  Duration:        {duration_sec}s")
    print(f"  BEATs chunks:    {n_beats}  ×  10s")
    print(f"  Whisper chunks:  {n_whisper}  ×  30s")
    print()
    print(f"  beats_out   : {list(beats_out.shape)}")
    print(f"              expected [{1}, {expected_beats_tokens}, 1024]")
    print(f"  whisper_out : {list(whisper_out.shape)}")
    print(f"              expected [{1}, {expected_whisper_tokens}, 768]")

    assert beats_out.shape   == (1, expected_beats_tokens, 1024), \
        f"BEATs mismatch: {beats_out.shape}"
    assert whisper_out.shape == (1, expected_whisper_tokens, 768), \
        f"Whisper mismatch: {whisper_out.shape}"

    # Show token counts for common durations
    print()
    print(f"  Token count reference table:")
    print(f"  {'Duration':<10} {'BEATs chunks':<15} {'BEATs tokens':<15} {'Whisper chunks':<16} {'Whisper tokens'}")
    print(f"  {'-'*70}")
    for d in [10, 20, 30, 60]:
        nb = math.ceil(d / 10)
        nw = math.ceil(d / 30)
        print(f"  {str(d)+'s':<10} {nb:<15} {nb*809:<15} {nw:<16} {nw*1500}")

    print("  PASS ✓")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio",    default=None, help="Path to audio file (optional)")
    ap.add_argument("--duration", type=int, default=30,
                    help="Duration in seconds for sliding window test (default: 30)")
    args = ap.parse_args()

    if args.audio:
        waveform, backend = load_audio(args.audio, duration_sec=10)
        print(f"Loaded: {args.audio}  (backend={backend})")
    else:
        waveform = torch.randn(1, 160000)
        print("Using synthetic noise (no --audio provided)")

    beats_enc   = test_openbeats(waveform)
    whisper_enc = test_whisper(waveform)
    test_batch(beats_enc, whisper_enc)
    test_sliding_window(beats_enc, whisper_enc, duration_sec=args.duration)

    print(f"\n{'='*55}")
    print("  All encoder tests passed.")
    print("  Proceed to: create mappers/, then test_mappers.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()