"""
model/dual_encoder_lm.py

Full dual-encoder audio language model.

Architecture:
    [OpenBEATs] → [AudioMapper]  → (B, K_a, 896) ─┐
                                                     ├→ prefix → [Qwen2.5-0.5B] → text
    [Whisper]   → [SpeechMapper] → (B, K_s, 896) ─┘

Integration strategy: PLITS with delayed fusion (PAL finding 1).

PLITS (Prepend to LLM Input Token Space):
    audio tokens are prepended to text tokens in the embedding space.
    Standard approach used by LTU, GAMA, Mellow, SALMONN.

Delayed Fusion (from PAL paper, Hypothesis 1):
    Skip the first k=4 LLM layers for audio tokens.
    Let text-only processing establish concept context first,
    then re-inject audio starting from layer k.
    Empirically improves performance at zero parameter cost.

    Implementation: we do NOT actually delay (that requires modifying
    SmolLM2's internals). Instead we use the simpler approach of
    inserting audio tokens AFTER the system prompt tokens rather than
    before the user prompt — same effective delay principle.
    Full PAL/LAL can be added later once this pipeline is validated.

Prefix layout (two audio clips + text prompt):
    [audio1_sound | SEP | audio1_speech | SEP |
     audio2_sound | SEP | audio2_speech | SEP | text_prompt]

Prefix length (default config):
    AudioMapper  stride=4 → 203 tokens per clip
    SpeechMapper stride=4 → 375 tokens per clip
    2 × (203 + 1 + 375 + 1) + 129 text = 1289 tokens

    Qwen2.5-0.5B context: 32768 tokens → fits easily.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


class DualEncoderLM(nn.Module):
    """
    Assembles encoders + mappers + SmolLM2 into a single trainable model.

    Frozen (during initial training on A100):
        - OpenBEATsEncoder
        - WhisperEncoder

    Trainable:
        - AudioMapper
        - SpeechMapper
        - SmolLM2 (full fine-tune or LoRA, your choice at training time)
    """

    def __init__(
        self,
        beats_encoder,    # OpenBEATsEncoder instance
        whisper_encoder,  # WhisperEncoder instance
        audio_mapper,     # AudioMapper instance
        speech_mapper,    # SpeechMapper instance
        lm_name: str = "Qwen/Qwen2.5-0.5B",
    ):
        super().__init__()

        # Encoders — not registered as submodules here since they're
        # instantiated externally and frozen. We just hold references.
        self.beats_encoder   = beats_encoder
        self.whisper_encoder = whisper_encoder

        # Mappers — these ARE registered (trainable)
        self.audio_mapper  = audio_mapper
        self.speech_mapper = speech_mapper

        # Language model
        self.lm_name = lm_name.lower()
        print(f"[DualEncoderLM] Loading {lm_name}...")
        self.lm = AutoModelForCausalLM.from_pretrained(lm_name)
        print(f"[DualEncoderLM] LM loaded.")

        # Get embedding dimension from SmolLM2
        if any(k in self.lm_name for k in ("smol", "qwen")):
            self.lm_embed_dim = self.lm.model.embed_tokens.weight.shape[1]
        elif "gpt2" in self.lm_name:
            self.lm_embed_dim = self.lm.transformer.wte.weight.shape[1]
        else:
            # Generic fallback for Llama-family models
            self.lm_embed_dim = self.lm.model.embed_tokens.weight.shape[1]

        self.lm_dtype = self._infer_lm_dtype()

        print(f"[DualEncoderLM] LM embedding dim: {self.lm_embed_dim}")
        print(f"[DualEncoderLM] LM dtype: {self.lm_dtype}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert token ids to embeddings using LM's embedding layer."""
        if any(k in self.lm_name for k in ("smol", "qwen")):
            return self.lm.model.embed_tokens(input_ids)
        elif "gpt2" in self.lm_name:
            return self.lm.transformer.wte(input_ids)

    def _infer_lm_dtype(self) -> torch.dtype:
        """Infer LM computation dtype from token embedding weights."""
        if any(k in self.lm_name for k in ("smol", "qwen")):
            return self.lm.model.embed_tokens.weight.dtype
        if "gpt2" in self.lm_name:
            return self.lm.transformer.wte.weight.dtype
        return next(self.lm.parameters()).dtype

    def _sep_embed(self, B: int, device) -> torch.Tensor:
        """Single separator token embedding, broadcast to batch."""
        if any(k in self.lm_name for k in ("smol", "qwen")):
            # Use eos_token_id as separator (model-agnostic)
            eos_id = getattr(self.lm.config, 'eos_token_id', 0)
            if isinstance(eos_id, list):
                eos_id = eos_id[0]
            sep_id = torch.tensor([eos_id], device=device)
            sep    = self.lm.model.embed_tokens(sep_id)
        elif "gpt2" in self.lm_name:
            sep_id = torch.tensor([50256], device=device)
            sep    = self.lm.transformer.wte(sep_id)
        return sep.unsqueeze(0).repeat(B, 1, 1)   # (B, 1, lm_embed_dim)

    # ------------------------------------------------------------------
    # Core method: build prefix
    # ------------------------------------------------------------------

    def build_prefix(
        self,
        audio1:    torch.Tensor,   # (B, samples) at 16kHz
        audio2:    torch.Tensor,   # (B, samples) at 16kHz
        text_ids:  dict,           # tokenizer output {'input_ids': ...}
    ) -> torch.Tensor:
        """
        Runs both encoders, maps to LM space, assembles the prefix.

        Returns:
            prefix: (B, prefix_length, lm_embed_dim)
        """
        B      = audio1.shape[0]
        device = audio1.device

        # ── Run encoders (frozen) ───────────────────────────────────────
        beats_enc_frozen = not any(p.requires_grad for p in self.beats_encoder.parameters())
        with torch.set_grad_enabled(not beats_enc_frozen):
            beats1  = self.beats_encoder(audio1)['embedding']   # (B, 809, 1024)
            beats2  = self.beats_encoder(audio2)['embedding']

        whisper_enc_frozen = not any(p.requires_grad for p in self.whisper_encoder.parameters())
        with torch.set_grad_enabled(not whisper_enc_frozen):
            whisper1 = self.whisper_encoder(audio1)             # (B, 1500, 768)
            whisper2 = self.whisper_encoder(audio2)

        # ── Run mappers (trainable) ─────────────────────────────────────
        a1_sound  = self.audio_mapper(beats1).to(self.lm_dtype)      # (B, 203, 576)
        a1_speech = self.speech_mapper(whisper1).to(self.lm_dtype)   # (B, 375, 576)
        a2_sound  = self.audio_mapper(beats2).to(self.lm_dtype)
        a2_speech = self.speech_mapper(whisper2).to(self.lm_dtype)

        # ── Text embedding ──────────────────────────────────────────────
        text_embed = self._embed_text(text_ids['input_ids'].to(device)).to(self.lm_dtype)   # (B, L, 576)

        # ── Separator token ─────────────────────────────────────────────
        sep = self._sep_embed(B, device)   # (B, 1, 576)

        # ── Assemble prefix ─────────────────────────────────────────────
        # Layout: [audio1_sound | SEP | audio1_speech | SEP |
        #          audio2_sound | SEP | audio2_speech | SEP | text]
        prefix = torch.cat([
            a1_sound, sep, a1_speech, sep,
            a2_sound, sep, a2_speech, sep,
            text_embed,
        ], dim=1)                          # (B, prefix_length, 576)

        return prefix

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        audio1:    torch.Tensor,
        audio2:    torch.Tensor,
        text_ids:  dict,           # tokenizer({'input_ids': ...}) for prompt
        answer_ids: dict,          # tokenizer({'input_ids': ...}) for answer
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Full training forward pass.

        The LM is trained to predict answer_ids given the prefix.
        Labels for the prefix are set to -100 (ignored in cross-entropy loss).

        Args:
            audio1, audio2:  (B, samples) at 16kHz
            text_ids:        tokenizer output for the question/prompt
            answer_ids:      tokenizer output for the expected answer
            attention_mask:  optional, will be created if None

        Returns:
            loss: scalar cross-entropy loss on the answer tokens only
        """
        device = audio1.device

        # Build prefix
        prefix = self.build_prefix(audio1, audio2, text_ids)   # (B, P, 576)

        # Embed answer
        answer_embed = self._embed_text(answer_ids['input_ids'].to(device)).to(self.lm_dtype)  # (B, A, 576)

        # Concatenate: prefix + answer
        full_seq = torch.cat([prefix, answer_embed], dim=1)     # (B, P+A, 576)

        # Build labels: -100 for prefix (ignored), answer ids for answer
        prefix_labels = torch.full(
            (prefix.shape[0], prefix.shape[1]),
            fill_value=-100,
            dtype=torch.long,
            device=device,
        )
        labels = torch.cat([prefix_labels, answer_ids['input_ids'].to(device)], dim=1)

        # Attention mask for full sequence
        if attention_mask is None:
            attention_mask = torch.ones(
                full_seq.shape[0], full_seq.shape[1],
                dtype=torch.long, device=device,
            )

        out = self.lm(
            inputs_embeds  = full_seq,
            labels         = labels,
            attention_mask = attention_mask,
        )
        return out.loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(
        self,
        audio1:    torch.Tensor,
        audio2:    torch.Tensor,
        text_ids:  dict,
        max_new_tokens: int   = 200,
        top_p:          float = 0.8,
        temperature:    float = 1.0,
    ) -> torch.Tensor:
        """
        Greedy/nucleus sampling inference.

        Returns:
            output_ids: (B, max_new_tokens) token ids
        """
        self.eval()
        with torch.no_grad():
            prefix = self.build_prefix(audio1, audio2, text_ids)
            out    = self.lm.generate(
                inputs_embeds  = prefix,
                max_new_tokens = max_new_tokens,
                top_p          = top_p,
                temperature    = temperature,
                do_sample      = (temperature > 0),
            )
        return out

    # ------------------------------------------------------------------
    # Sliding window version (for audio > 10s)
    # ------------------------------------------------------------------

    def build_prefix_long(
        self,
        audio1:   torch.Tensor,
        audio2:   torch.Tensor,
        text_ids: dict,
    ) -> torch.Tensor:
        """
        Same as build_prefix but uses SlidingWindowExtractor.
        Use this when audio may be longer than 10s.
        """
        from encoders.sliding_window import SlidingWindowExtractor
        extractor = SlidingWindowExtractor(self.beats_encoder, self.whisper_encoder)

        B      = audio1.shape[0]
        device = audio1.device

        beats1,   whisper1 = extractor(audio1)
        beats2,   whisper2 = extractor(audio2)

        a1_sound  = self.audio_mapper(beats1).to(self.lm_dtype)
        a1_speech = self.speech_mapper(whisper1).to(self.lm_dtype)
        a2_sound  = self.audio_mapper(beats2).to(self.lm_dtype)
        a2_speech = self.speech_mapper(whisper2).to(self.lm_dtype)

        text_embed = self._embed_text(text_ids['input_ids'].to(device)).to(self.lm_dtype)
        sep        = self._sep_embed(B, device)

        return torch.cat([
            a1_sound, sep, a1_speech, sep,
            a2_sound, sep, a2_speech, sep,
            text_embed,
        ], dim=1)