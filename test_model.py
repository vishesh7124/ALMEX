"""
test_model.py
Run from audio_lm/ directory:
    python test_model.py
    python test_model.py --audio test_files/7-cmp.wav

Tests the full assembled model. No training — just shape verification.
"""

import sys, os, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_all_components():
    """Load every component in order."""
    from encoders.openbeats import OpenBEATsEncoder
    from encoders.whisper_enc import WhisperEncoder
    from mappers.audio_mapper import AudioMapper
    from mappers.speech_mapper import SpeechMapper
    from model.dual_encoder_lm import DualEncoderLM

    print("Loading encoders...")
    beats_enc   = OpenBEATsEncoder(freeze=True)
    whisper_enc = WhisperEncoder(freeze=True)

    print("\nCreating mappers...")
    audio_mapper  = AudioMapper(encoder_dim=1024, lm_dim=896,
                                conv_stride=4, conv_kernel=7)
    speech_mapper = SpeechMapper(encoder_dim=768,  lm_dim=896,
                                 conv_stride=4, conv_kernel=5)

    print("\nAssembling full model...")
    model = DualEncoderLM(
        beats_encoder   = beats_enc,
        whisper_encoder = whisper_enc,
        audio_mapper    = audio_mapper,
        speech_mapper   = speech_mapper,
        lm_name         = "Qwen/Qwen2.5-0.5B",
    )

    return model


def test_prefix_shape(model, audio_path=None):
    print(f"\n{'='*55}")
    print("  Prefix Shape Test")
    print(f"{'='*55}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    if audio_path:
        from utils.audio_loader import load_audio_10s_mono16k
        waveform, _ = load_audio_10s_mono16k(audio_path)
    else:
        waveform = torch.randn(1, 160000)

    prompt = "What can you infer about the surroundings from the audio?"
    text_ids = tokenizer(prompt, return_tensors="pt", padding=True)

    with torch.no_grad():
        prefix = model.build_prefix(waveform, waveform, text_ids)

    P = prefix.shape[1]
    print(f"\n  Input audio: {list(waveform.shape)}")
    print(f"  Prompt: '{prompt}'")
    print(f"  Prompt tokens: {text_ids['input_ids'].shape[1]}")
    print()
    print(f"  Prefix shape: {list(prefix.shape)}")
    print(f"    (B={prefix.shape[0]}, prefix_length={P}, lm_dim={prefix.shape[2]})")
    print()

    # Breakdown
    a_tok = model.audio_mapper.tokens_per_clip    # 203
    s_tok = model.speech_mapper.tokens_per_clip   # 375
    t_tok = text_ids['input_ids'].shape[1]
    expected = 2 * (a_tok + 1 + s_tok + 1) + t_tok

    print(f"  Token breakdown:")
    print(f"    audio1 sound:   {a_tok}")
    print(f"    SEP:             1")
    print(f"    audio1 speech:  {s_tok}")
    print(f"    SEP:             1")
    print(f"    audio2 sound:   {a_tok}")
    print(f"    SEP:             1")
    print(f"    audio2 speech:  {s_tok}")
    print(f"    SEP:             1")
    print(f"    text prompt:    {t_tok}")
    print(f"    ─────────────────────")
    print(f"    TOTAL:          {expected}")
    print()
    print(f"  Qwen2.5-0.5B context window: 32768 → {'✓ fits' if expected < 32768 else '✗ too long'}")

    assert prefix.shape == (1, expected, 896), \
        f"Shape mismatch: got {prefix.shape}, expected (1, {expected}, 896)"
    print("  PASS ✓")
    return tokenizer, prefix.shape[1]


def test_forward_pass(model, prefix_len):
    """Test training forward pass (loss computation)."""
    print(f"\n{'='*55}")
    print("  Training Forward Pass (Loss)")
    print(f"{'='*55}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    audio   = torch.randn(1, 160000)
    prompt  = "Caption the audio."
    answer  = "A dog is barking in the background."

    text_ids   = tokenizer(prompt, return_tensors="pt", padding=True)
    answer_ids = tokenizer(answer, return_tensors="pt", padding=True)

    with torch.no_grad():
        loss = model(audio, audio, text_ids, answer_ids)

    print(f"\n  Loss: {loss.item():.4f}")
    print(f"  (random init mappers → expect ~loss of 3-5, not zero)")
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss is NaN — something is wrong"
    print("  PASS ✓")


def test_generation(model):
    """Test inference generation."""
    print(f"\n{'='*55}")
    print("  Inference / Generation")
    print(f"{'='*55}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    audio    = torch.randn(1, 160000)
    prompt   = "Caption the audio."
    text_ids = tokenizer(prompt, return_tensors="pt", padding=True)

    with torch.no_grad():
        output_ids = model.generate(
            audio, audio, text_ids,
            max_new_tokens=20,
            temperature=1.0,
        )

    # Decode — output_ids may include the prompt, take last tokens
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\n  Prompt:    '{prompt}'")
    print(f"  Generated: '{generated[:100]}'")
    print(f"  (mappers random-init → output will be nonsense, that's expected)")
    print("  PASS ✓")


def test_parameter_count(model):
    print(f"\n{'='*55}")
    print("  Parameter Count")
    print(f"{'='*55}")

    def count(m): return sum(p.numel() for p in m.parameters())
    def count_trainable(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

    rows = [
        ("OpenBEATsEncoder (frozen)",  count(model.beats_encoder),   0),
        ("WhisperEncoder (frozen)",    count(model.whisper_encoder),  0),
        ("AudioMapper",                count(model.audio_mapper),     count_trainable(model.audio_mapper)),
        ("SpeechMapper",               count(model.speech_mapper),    count_trainable(model.speech_mapper)),
        ("Qwen2.5-0.5B",              count(model.lm),               count_trainable(model.lm)),
    ]

    print(f"\n  {'Component':<30} {'Total':>12} {'Trainable':>12}")
    print(f"  {'─'*56}")
    total, total_train = 0, 0
    for name, tot, train in rows:
        print(f"  {name:<30} {tot:>12,} {train:>12,}")
        total       += tot
        total_train += train
    print(f"  {'─'*56}")
    print(f"  {'TOTAL':<30} {total:>12,} {total_train:>12,}")
    print()
    print(f"  Training only mappers + Qwen2.5-0.5B on A100 80GB: easily fits")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=None)
    args = ap.parse_args()

    model = load_all_components()
    model.eval()

    tokenizer, prefix_len = test_prefix_shape(model, args.audio)
    test_forward_pass(model, prefix_len)
    test_generation(model)
    test_parameter_count(model)

    print(f"\n{'='*55}")
    print("  All model tests passed.")
    print("  Next: write training loop (train.py)")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()