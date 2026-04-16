"""
train_stage1.py

Stage 1: Mapper Pre-Alignment
─────────────────────────────
Goal:   Train AudioMapper + SpeechMapper (~7M params) to project
        frozen encoder features into Qwen2.5-0.5B's 896-dim space.

Frozen: OpenBEATs, Whisper, Qwen2.5-0.5B (all frozen, .eval())
Train:  AudioMapper + SpeechMapper ONLY

Loss:   Autoregressive cross-entropy on ANSWER tokens only.
        Audio prefix + question tokens masked to -100.

Optimizer: AdamW (lr=1e-3, weight_decay=0.01)
Scheduler: Cosine annealing with 500-step linear warmup
Precision: bf16 mixed precision

Logging:
    TensorBoard: always enabled (tensorboard --logdir runs/)
    WandB:       optional (--use_wandb)

Usage:
    # Full training (GPU machine):
    python train_stage1.py

    # Dry run (laptop, no audio files needed):
    python train_stage1.py --dry_run --epochs 1 --batch_size 2 --grad_accum 1
"""

import os
import sys
import json
import math
import time
import random
import argparse
import datetime
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import torchaudio


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Stage 1: Mapper Pre-Alignment")
    # Data
    p.add_argument("--train_jsonl", default="data/stage1/stage1_train.jsonl")
    p.add_argument("--val_jsonl", default="data/stage1/stage1_val.jsonl")
    # Model
    p.add_argument("--lm_name", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--beats_model", default="shikhar7ssu/OpenBEATs-ICME")
    p.add_argument("--whisper_model", default="openai/whisper-small")
    # Training
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=8,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_audio_sec", type=float, default=10.0)
    p.add_argument("--max_answer_tokens", type=int, default=128)
    p.add_argument("--max_question_tokens", type=int, default=64)
    # Output
    p.add_argument("--output_dir", default="checkpoints/stage1")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--save_every_epoch", action="store_true", default=True)
    # Logging
    p.add_argument("--log_dir", default="runs/stage1",
                   help="TensorBoard log directory")
    p.add_argument("--use_wandb", action="store_true",
                   help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", default="ALM-stage1",
                   help="WandB project name")
    p.add_argument("--run_name", default=None,
                   help="Run name for logging (auto-generated if not set)")
    # Hardware
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=4)
    # Testing
    p.add_argument("--dry_run", action="store_true",
                   help="Use synthetic data for testing (no audio files needed)")
    p.add_argument("--dry_run_samples", type=int, default=64,
                   help="Number of synthetic samples for dry run")
    # Resume
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# Logging Setup
# ═══════════════════════════════════════════════════════════════════════

class Logger:
    """
    Unified logging interface for TensorBoard + optional WandB.

    TensorBoard: Writes to runs/stage1/ — view with:
        tensorboard --logdir runs/
        Then open http://localhost:6006

    WandB: Enable with --use_wandb. View at wandb.ai.
    """

    def __init__(self, args):
        self.global_step = 0
        self.use_wandb = args.use_wandb

        # ── TensorBoard (always on) ──────────────────────────────────
        from torch.utils.tensorboard import SummaryWriter

        run_name = args.run_name or (
            f"stage1_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        log_dir = os.path.join(args.log_dir, run_name)
        self.tb = SummaryWriter(log_dir=log_dir)
        print(f"  📊 TensorBoard: {log_dir}")
        print(f"     View with: tensorboard --logdir {args.log_dir}")

        # Log hyperparameters
        self.tb.add_text("hyperparameters", json.dumps(vars(args), indent=2))

        # ── WandB (optional) ─────────────────────────────────────────
        if self.use_wandb:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
            )
            self.wandb = wandb
            print(f"  📊 WandB: {args.wandb_project}/{run_name}")
        else:
            self.wandb = None
            print(f"  📊 WandB: disabled (use --use_wandb to enable)")

    def log_scalar(self, tag, value, step=None):
        """Log a single scalar value."""
        step = step if step is not None else self.global_step
        self.tb.add_scalar(tag, value, step)
        if self.wandb:
            self.wandb.log({tag: value}, step=step)

    def log_scalars(self, main_tag, tag_value_dict, step=None):
        """Log multiple related scalars (e.g., loss per encoder type)."""
        step = step if step is not None else self.global_step
        self.tb.add_scalars(main_tag, tag_value_dict, step)
        if self.wandb:
            self.wandb.log(
                {f"{main_tag}/{k}": v for k, v in tag_value_dict.items()},
                step=step,
            )

    def log_histogram(self, tag, values, step=None):
        """Log parameter or gradient distributions."""
        step = step if step is not None else self.global_step
        self.tb.add_histogram(tag, values, step)

    def log_text(self, tag, text, step=None):
        """Log text (e.g., generation samples)."""
        step = step if step is not None else self.global_step
        self.tb.add_text(tag, text, step)
        if self.wandb:
            self.wandb.log({tag: self.wandb.Html(f"<pre>{text}</pre>")}, step=step)

    def close(self):
        self.tb.close()
        if self.wandb:
            self.wandb.finish()


# ═══════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════

SAMPLE_RATE = 16000

def load_audio(path: str, max_sec: float = 10.0) -> torch.Tensor:
    """Load audio file, resample to 16kHz, trim/pad to max_sec."""
    try:
        waveform, sr = torchaudio.load(path)
    except Exception:
        # Return silence on load failure
        return torch.zeros(1, int(max_sec * SAMPLE_RATE))

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    # Trim or pad
    max_samples = int(max_sec * SAMPLE_RATE)
    T = waveform.shape[1]
    if T > max_samples:
        waveform = waveform[:, :max_samples]
    elif T < max_samples:
        waveform = F.pad(waveform, (0, max_samples - T))

    return waveform  # (1, max_samples)


class Stage1Dataset(Dataset):
    """
    Loads JSONL with fields: audio_path, question, answer, task_type, encoder_type, source.
    Returns: waveform, question, answer, encoder_type
    """

    def __init__(self, jsonl_path: str, max_audio_sec: float = 10.0):
        self.max_audio_sec = max_audio_sec
        self.examples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

        # Print source distribution
        sources = Counter(ex.get("source", "unknown") for ex in self.examples)
        enc_types = Counter(ex.get("encoder_type", "unknown") for ex in self.examples)
        print(f"  Loaded {len(self.examples):,} examples from {jsonl_path}")
        print(f"    Encoder types: {dict(enc_types)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        waveform = load_audio(ex["audio_path"], self.max_audio_sec)
        return {
            "waveform": waveform.squeeze(0),   # (samples,)
            "question": ex["question"],
            "answer": ex["answer"],
            "encoder_type": ex["encoder_type"],  # "audio", "speech", "both"
        }


class DryRunDataset(Dataset):
    """
    Synthetic dataset for testing the training loop without audio files.
    Generates random waveforms + dummy Q/A pairs.
    """

    DUMMY_TASKS = [
        ("audio", "Describe the sounds in this audio.",
         "Birds chirping and wind blowing through trees."),
        ("speech", "What is the person saying?",
         "THE QUICK BROWN FOX JUMPED OVER THE LAZY DOG"),
        ("audio", "What type of sound is this?",
         "Dog barking, Crowd cheering"),
        ("both", "Transcribe the speech in this audio.",
         "No speech is present in this audio clip. Only environmental sounds."),
        ("speech", "Transcribe the speech.",
         "GOOD MORNING EVERYONE AND WELCOME TO THE CONFERENCE"),
        ("audio", "Write a caption for this audio.",
         "A car engine revving followed by tires screeching on pavement."),
    ]

    def __init__(self, n_samples: int = 64, max_audio_sec: float = 10.0):
        self.n_samples = n_samples
        self.max_samples = int(max_audio_sec * SAMPLE_RATE)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        task = self.DUMMY_TASKS[idx % len(self.DUMMY_TASKS)]
        # Random waveform (simulates audio)
        waveform = torch.randn(self.max_samples) * 0.1
        return {
            "waveform": waveform,
            "question": task[1],
            "answer": task[2],
            "encoder_type": task[0],
        }


def collate_fn(batch):
    """Collate variable-length audio into padded batch."""
    max_len = max(b["waveform"].shape[0] for b in batch)
    waveforms = []
    for b in batch:
        wav = b["waveform"]
        if wav.shape[0] < max_len:
            wav = F.pad(wav, (0, max_len - wav.shape[0]))
        waveforms.append(wav)

    return {
        "waveform": torch.stack(waveforms),    # (B, samples)
        "question": [b["question"] for b in batch],
        "answer": [b["answer"] for b in batch],
        "encoder_type": [b["encoder_type"] for b in batch],
    }


# ═══════════════════════════════════════════════════════════════════════
# Learning rate scheduler with warmup
# ═══════════════════════════════════════════════════════════════════════

class CosineWarmupScheduler:
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            scale = self.step_count / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = 0.5 * (1 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, base_lr * scale)

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


# ═══════════════════════════════════════════════════════════════════════
# Training utilities
# ═══════════════════════════════════════════════════════════════════════

def compute_grad_norm(parameters):
    """Compute total L2 gradient norm across all parameters."""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def gpu_mem_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


# ═══════════════════════════════════════════════════════════════════════
# Core Training Logic
# ═══════════════════════════════════════════════════════════════════════

def build_model(args):
    """Build all components, freeze encoders + LM, return trainable parts."""
    print("=" * 70)
    print("  Building Model Components")
    print("=" * 70)

    # ── Encoders (frozen) ─────────────────────────────────────────────
    from encoders.openbeats import OpenBEATsEncoder
    from encoders.whisper_enc import WhisperEncoder

    beats_encoder = OpenBEATsEncoder(
        model_name=args.beats_model, freeze=True
    )
    whisper_encoder = WhisperEncoder(
        model_name=args.whisper_model, freeze=True
    )

    # ── Mappers (trainable) ──────────────────────────────────────────
    from mappers.audio_mapper import AudioMapper
    from mappers.speech_mapper import SpeechMapper

    audio_mapper = AudioMapper(
        encoder_dim=beats_encoder.hidden_size,
        lm_dim=896,   # Qwen2.5-0.5B
    )
    speech_mapper = SpeechMapper(
        encoder_dim=whisper_encoder.hidden_size,
        lm_dim=896,
    )

    # ── Language Model (frozen) ──────────────────────────────────────
    print(f"\n[LM] Loading {args.lm_name}...")
    lm = AutoModelForCausalLM.from_pretrained(
        args.lm_name, torch_dtype=torch.bfloat16
    )
    for p in lm.parameters():
        p.requires_grad_(False)
    lm.eval()
    print(f"[LM] Frozen. Params: {sum(p.numel() for p in lm.parameters()):,}")

    # ── Tokenizer ────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Move to device ───────────────────────────────────────────────
    device = torch.device(args.device)
    beats_encoder = beats_encoder.to(device)
    whisper_encoder = whisper_encoder.to(device)
    audio_mapper = audio_mapper.to(device)
    speech_mapper = speech_mapper.to(device)
    lm = lm.to(device)

    # ── Summary ──────────────────────────────────────────────────────
    trainable = sum(p.numel() for p in audio_mapper.parameters()) + \
                sum(p.numel() for p in speech_mapper.parameters())
    frozen = sum(p.numel() for p in beats_encoder.parameters()) + \
             sum(p.numel() for p in whisper_encoder.parameters()) + \
             sum(p.numel() for p in lm.parameters())
    print(f"\n  Trainable: {trainable:,} params ({trainable/1e6:.1f}M)")
    print(f"  Frozen:    {frozen:,} params ({frozen/1e6:.1f}M)")
    print(f"  Total:     {trainable + frozen:,} params")

    if torch.cuda.is_available():
        print(f"  GPU memory: {gpu_mem_mb():.0f} MB")

    return {
        "beats_encoder": beats_encoder,
        "whisper_encoder": whisper_encoder,
        "audio_mapper": audio_mapper,
        "speech_mapper": speech_mapper,
        "lm": lm,
        "tokenizer": tokenizer,
        "device": device,
    }


def forward_step(batch, components, args):
    """
    Single forward pass for a batch.

    Routes audio to correct encoder based on encoder_type.
    Builds prefix: [mapper_output | question_embeds]
    Computes loss on answer tokens only (prefix masked to -100).

    Returns: (loss, per_type_losses dict)
    """
    device = components["device"]
    lm = components["lm"]
    tokenizer = components["tokenizer"]
    beats_encoder = components["beats_encoder"]
    whisper_encoder = components["whisper_encoder"]
    audio_mapper = components["audio_mapper"]
    speech_mapper = components["speech_mapper"]

    waveform = batch["waveform"].to(device)             # (B, samples)
    questions = batch["question"]                        # list of str
    answers = batch["answer"]                            # list of str
    encoder_types = batch["encoder_type"]                # list of str
    B = waveform.shape[0]

    # ── Get LM embedding function & dtype ────────────────────────────
    embed_tokens = lm.model.embed_tokens
    lm_dtype = embed_tokens.weight.dtype

    # ── Run frozen encoders ──────────────────────────────────────────
    with torch.no_grad():
        beats_features = beats_encoder(waveform)["embedding"]  # (B, 809, 1024)
        whisper_features = whisper_encoder(waveform)            # (B, 1500, 768)

    # ── Run trainable mappers ────────────────────────────────────────
    audio_projected = audio_mapper(beats_features).to(lm_dtype)    # (B, 203, 896)
    speech_projected = speech_mapper(whisper_features).to(lm_dtype) # (B, 375, 896)

    # ── Tokenize questions and answers ───────────────────────────────
    q_tok = tokenizer(
        questions, padding=True, truncation=True,
        max_length=args.max_question_tokens, return_tensors="pt"
    ).to(device)
    a_tok = tokenizer(
        answers, padding=True, truncation=True,
        max_length=args.max_answer_tokens, return_tensors="pt"
    ).to(device)

    q_embeds = embed_tokens(q_tok.input_ids).to(lm_dtype)  # (B, Lq, 896)
    a_embeds = embed_tokens(a_tok.input_ids).to(lm_dtype)  # (B, La, 896)

    # ── Build per-sample prefix based on encoder_type ────────────────
    # We process each sample individually because encoder_type varies within batch
    losses = []
    per_type_losses = defaultdict(list)

    for i in range(B):
        enc_type = encoder_types[i]

        # Select mapper output based on encoder routing
        if enc_type == "audio":
            prefix = audio_projected[i:i+1]       # (1, 203, 896)
        elif enc_type == "speech":
            prefix = speech_projected[i:i+1]       # (1, 375, 896)
        elif enc_type == "both":
            # Concatenate both mapper outputs
            prefix = torch.cat([
                audio_projected[i:i+1],
                speech_projected[i:i+1]
            ], dim=1)                              # (1, 578, 896)
        else:
            prefix = audio_projected[i:i+1]        # fallback

        # Get question and answer embeddings for this sample
        # Trim padding from question
        q_mask = q_tok.attention_mask[i]           # (Lq,)
        q_len = q_mask.sum().item()
        q_emb = q_embeds[i:i+1, :q_len]           # (1, q_len, 896)

        # Trim padding from answer
        a_mask = a_tok.attention_mask[i]           # (La,)
        a_len = a_mask.sum().item()
        a_emb = a_embeds[i:i+1, :a_len]           # (1, a_len, 896)
        a_ids = a_tok.input_ids[i, :a_len]         # (a_len,)

        # Assemble: [prefix | question | answer]
        full_embeds = torch.cat([prefix, q_emb, a_emb], dim=1)  # (1, P+Q+A, 896)

        # Labels: -100 for prefix+question, actual ids for answer
        prefix_q_len = prefix.shape[1] + q_len
        ignore_labels = torch.full(
            (1, prefix_q_len), -100, dtype=torch.long, device=device
        )
        answer_labels = a_ids.unsqueeze(0)         # (1, a_len)
        labels = torch.cat([ignore_labels, answer_labels], dim=1)  # (1, P+Q+A)

        # Attention mask
        attn_mask = torch.ones(
            1, full_embeds.shape[1], dtype=torch.long, device=device
        )

        # Forward through frozen LM
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = lm(
                inputs_embeds=full_embeds,
                attention_mask=attn_mask,
                labels=labels,
            )
        losses.append(outputs.loss)
        per_type_losses[enc_type].append(outputs.loss.item())

    # Average loss across batch
    loss = torch.stack(losses).mean()

    # Average per-type losses
    per_type_avg = {k: sum(v) / len(v) for k, v in per_type_losses.items()}

    return loss, per_type_avg


@torch.no_grad()
def generate_samples(components, args, n_samples=3):
    """
    Generate text from random audio to check mapper quality.
    Logged to TensorBoard as text at validation time.
    """
    device = components["device"]
    lm = components["lm"]
    tokenizer = components["tokenizer"]
    audio_mapper = components["audio_mapper"]
    speech_mapper = components["speech_mapper"]
    beats_encoder = components["beats_encoder"]
    whisper_encoder = components["whisper_encoder"]

    audio_mapper.eval()
    speech_mapper.eval()

    samples = []
    prompts = [
        ("Describe the sounds in this audio.", "audio"),
        ("What is the person saying?", "speech"),
        ("What type of sound is this?", "audio"),
    ]

    for prompt_text, enc_type in prompts[:n_samples]:
        # Random noise as audio (will produce garbage — that's expected initially)
        fake_audio = torch.randn(1, int(args.max_audio_sec * SAMPLE_RATE)).to(device)

        # Encode
        beats_feat = beats_encoder(fake_audio)["embedding"]
        whisper_feat = whisper_encoder(fake_audio)

        if enc_type == "audio":
            prefix = audio_mapper(beats_feat).to(lm.model.embed_tokens.weight.dtype)
        else:
            prefix = speech_mapper(whisper_feat).to(lm.model.embed_tokens.weight.dtype)

        # Question embeddings
        q_tok = tokenizer(prompt_text, return_tensors="pt").to(device)
        q_emb = lm.model.embed_tokens(q_tok.input_ids).to(prefix.dtype)

        # Prefix + question
        input_embeds = torch.cat([prefix, q_emb], dim=1)

        # Generate
        attn_mask = torch.ones(1, input_embeds.shape[1], dtype=torch.long, device=device)
        output_ids = lm.generate(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        samples.append(f"**Q:** {prompt_text}\n**A:** {generated}\n")

    audio_mapper.train()
    speech_mapper.train()
    return "\n---\n".join(samples)


@torch.no_grad()
def validate(val_loader, components, args, logger=None, max_batches=50):
    """Run validation, return average loss. Logs per-type losses."""
    components["audio_mapper"].eval()
    components["speech_mapper"].eval()

    total_loss = 0.0
    n_batches = 0
    all_type_losses = defaultdict(list)

    for batch in val_loader:
        if n_batches >= max_batches:
            break
        loss, per_type = forward_step(batch, components, args)
        total_loss += loss.item()
        for k, v in per_type.items():
            all_type_losses[k].append(v)
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)

    # Log per-type validation losses
    if logger:
        type_avgs = {k: sum(v) / len(v) for k, v in all_type_losses.items()}
        logger.log_scalars("val/loss_by_encoder", type_avgs)

    components["audio_mapper"].train()
    components["speech_mapper"].train()

    return avg_loss


def save_checkpoint(components, optimizer, scheduler, epoch, step, loss, args):
    """Save mapper checkpoints (not the frozen models)."""
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "audio_mapper": components["audio_mapper"].state_dict(),
        "speech_mapper": components["speech_mapper"].state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler_step": scheduler.step_count,
        "args": vars(args),
    }
    path = os.path.join(args.output_dir, f"stage1_epoch{epoch}_step{step}.pt")
    torch.save(ckpt, path)
    print(f"  💾 Saved checkpoint: {path}")

    # Also save as "latest"
    latest = os.path.join(args.output_dir, "stage1_latest.pt")
    torch.save(ckpt, latest)


def train(args):
    """Main training loop."""
    print("=" * 70)
    print("  Stage 1: Mapper Pre-Alignment")
    print("=" * 70)

    # ── Logger ───────────────────────────────────────────────────────
    logger = Logger(args)

    # ── Build model ──────────────────────────────────────────────────
    components = build_model(args)

    # ── Data ─────────────────────────────────────────────────────────
    print(f"\nLoading datasets...")
    if args.dry_run:
        print("  ⚠️  DRY RUN MODE — using synthetic data")
        train_ds = DryRunDataset(args.dry_run_samples, args.max_audio_sec)
        val_ds = DryRunDataset(max(8, args.dry_run_samples // 8), args.max_audio_sec)
    else:
        train_ds = Stage1Dataset(args.train_jsonl, args.max_audio_sec)
        val_ds = Stage1Dataset(args.val_jsonl, args.max_audio_sec)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers if not args.dry_run else 0,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers if not args.dry_run else 0,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ── Optimizer (mappers only) ─────────────────────────────────────
    trainable_params = (
        list(components["audio_mapper"].parameters()) +
        list(components["speech_mapper"].parameters())
    )
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # ── Scheduler ────────────────────────────────────────────────────
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=min(args.warmup_steps, total_steps // 2),
        total_steps=total_steps,
        min_lr=1e-6,
    )

    # ── Resume from checkpoint ───────────────────────────────────────
    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")

    if args.resume:
        print(f"\n  Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=components["device"])
        components["audio_mapper"].load_state_dict(ckpt["audio_mapper"])
        components["speech_mapper"].load_state_dict(ckpt["speech_mapper"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.step_count = ckpt.get("scheduler_step", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("loss", float("inf"))
        print(f"  Resumed: epoch={start_epoch}, step={global_step}, "
              f"loss={best_val_loss:.4f}")

    print(f"\n{'='*70}")
    print(f"  Training Config")
    print(f"{'='*70}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Grad accumulation:{args.grad_accum}")
    print(f"  Effective batch:  {args.batch_size * args.grad_accum}")
    print(f"  Learning rate:    {args.lr}")
    print(f"  Warmup steps:     {scheduler.warmup_steps}")
    print(f"  Steps/epoch:      {steps_per_epoch}")
    print(f"  Total steps:      {total_steps}")
    print(f"  Device:           {args.device}")
    if args.dry_run:
        print(f"  Mode:             DRY RUN ({args.dry_run_samples} samples)")
    print(f"{'='*70}\n")

    # Log initial GPU memory
    if torch.cuda.is_available():
        logger.log_scalar("system/gpu_memory_mb", gpu_mem_mb(), step=0)

    # ── Training ─────────────────────────────────────────────────────
    components["audio_mapper"].train()
    components["speech_mapper"].train()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_loss = 0.0
        epoch_batches = 0
        t_start = time.time()

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            # Forward
            loss, per_type_losses = forward_step(batch, components, args)
            loss_scaled = loss / args.grad_accum  # Scale for accumulation

            # Backward
            loss_scaled.backward()
            epoch_loss += loss.item()
            epoch_batches += 1

            # Optimizer step (every grad_accum batches)
            if (batch_idx + 1) % args.grad_accum == 0:
                # Compute gradient norm BEFORE clipping (for logging)
                grad_norm = compute_grad_norm(trainable_params)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                logger.global_step = global_step

                # ── Logging ──────────────────────────────────────────
                if global_step % args.log_every == 0:
                    avg_loss = epoch_loss / epoch_batches
                    lr = scheduler.get_lr()
                    elapsed = time.time() - t_start
                    samples_sec = (epoch_batches * args.batch_size) / elapsed

                    # Console
                    print(
                        f"  [{epoch}/{args.epochs}] step {global_step:>6d} | "
                        f"loss {avg_loss:.4f} | lr {lr:.2e} | "
                        f"‖∇‖ {grad_norm:.2f} | "
                        f"{samples_sec:.1f} samples/s"
                    )

                    # TensorBoard + WandB
                    logger.log_scalar("train/loss", avg_loss)
                    logger.log_scalar("train/loss_step", loss.item())
                    logger.log_scalar("train/lr", lr)
                    logger.log_scalar("train/grad_norm", grad_norm)
                    logger.log_scalar("train/samples_per_sec", samples_sec)
                    logger.log_scalar("train/epoch", epoch)

                    # Per-encoder-type loss
                    if per_type_losses:
                        logger.log_scalars("train/loss_by_encoder", per_type_losses)

                    # GPU memory
                    if torch.cuda.is_available():
                        logger.log_scalar("system/gpu_memory_mb", gpu_mem_mb())

                # ── Periodic validation ──────────────────────────────
                if global_step % args.val_every == 0:
                    val_loss = validate(
                        val_loader, components, args, logger, max_batches=50
                    )
                    print(f"  ── val loss: {val_loss:.4f} ──")

                    logger.log_scalar("val/loss", val_loss)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            components, optimizer, scheduler,
                            epoch, global_step, val_loss, args
                        )
                        print(f"  ★ New best val loss: {val_loss:.4f}")
                        logger.log_scalar("val/best_loss", best_val_loss)

                    # Generate sample text to check mapper quality
                    gen_text = generate_samples(components, args)
                    logger.log_text("val/generation_samples", gen_text)

                    components["audio_mapper"].train()
                    components["speech_mapper"].train()

        # ── End of epoch ─────────────────────────────────────────────
        epoch_avg_loss = epoch_loss / max(epoch_batches, 1)
        elapsed = time.time() - t_start
        print(f"\n  Epoch {epoch} complete | avg loss: {epoch_avg_loss:.4f} | "
              f"time: {elapsed/60:.1f}min")

        # Epoch validation + checkpoint
        val_loss = validate(val_loader, components, args, logger)
        print(f"  Epoch {epoch} val loss: {val_loss:.4f}")

        logger.log_scalar("val/loss_epoch", val_loss, step=epoch)
        logger.log_scalar("train/loss_epoch", epoch_avg_loss, step=epoch)

        if args.save_every_epoch:
            save_checkpoint(
                components, optimizer, scheduler,
                epoch, global_step, val_loss, args
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  ★ New best val loss: {val_loss:.4f}")

        # Log mapper weight distributions (once per epoch)
        for name, param in components["audio_mapper"].named_parameters():
            logger.log_histogram(f"weights/audio_mapper/{name}", param.data, step=epoch)
        for name, param in components["speech_mapper"].named_parameters():
            logger.log_histogram(f"weights/speech_mapper/{name}", param.data, step=epoch)

    # ── Cleanup ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints:   {args.output_dir}")
    print(f"  TensorBoard:   tensorboard --logdir {args.log_dir}")
    print(f"{'='*70}")
    logger.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
