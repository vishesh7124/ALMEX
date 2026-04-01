"""
prepare_stage1_data.py
Run from audio_lm/ directory:
    python prepare_stage1_data.py

Stage 1 data pipeline:
  Downloads AudioCaps + Clotho + FSD50K metadata (no audio yet).
  Converts existing captions/labels → closed-ended QA format.
  Injects 5-8% unanswerable queries.
  Outputs: data/stage1_train.jsonl, data/stage1_val.jsonl

Each line in the JSONL:
{
  "audio_path": "data/audio/audiocaps/train/Yxxxxxx.wav",
  "audio2_path": null,         # null = use same audio twice (single-audio tasks)
  "question": "Describe what you hear in the audio.",
  "answer": "A dog is barking followed by a door slamming.",
  "task_type": "captioning",   # captioning | classification | binary | unanswerable
  "source": "audiocaps"
}

Does NOT download audio yet — just builds the metadata JSONL.
Audio download is a separate step (download_audio.py) since it takes hours.
"""

import json
import os
import random
import csv
from pathlib import Path

random.seed(42)

DATA_DIR   = Path("data")
AUDIO_DIR  = DATA_DIR / "audio"
META_DIR   = DATA_DIR / "metadata"
OUT_DIR    = DATA_DIR / "stage1"

for d in [DATA_DIR, AUDIO_DIR, META_DIR, OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# Task templates — closed-ended only (Stage 1)
# ─────────────────────────────────────────────────────────────────────

CAPTION_QUESTIONS = [
    "Describe what you hear in the audio.",
    "Generate a caption describing the sounds in this audio clip.",
    "What sounds are present in the audio?",
    "Describe the auditory scene in this audio.",
    "What can you hear in this audio recording?",
]

CLASSIFICATION_QUESTIONS = [
    "What type of sound is this?",
    "What category best describes the sounds in this audio?",
    "Identify the primary sound event in this audio.",
]

BINARY_QUESTIONS_TEMPLATES = [
    ("Is there {event} in the audio?", "{answer}"),
    ("Can you hear {event} in this recording?", "{answer}"),
    ("Does this audio contain {event}?", "{answer}"),
]

# Unanswerable templates — injected at 5-8% of total
UNANSWERABLE_QUESTIONS = [
    ("What is the person in the audio saying?", "This audio does not contain any speech."),
    ("What song is playing in the audio?", "There is no music playing in this audio."),
    ("What language is being spoken?", "No speech is present in this audio clip."),
    ("What instrument is being played?", "This audio does not contain any musical instruments."),
    ("Who is the speaker in this audio?", "This audio contains no speech or speaking."),
]


def make_caption_example(audio_path: str, caption: str, source: str) -> dict:
    return {
        "audio_path":  audio_path,
        "audio2_path": None,
        "question":    random.choice(CAPTION_QUESTIONS),
        "answer":      caption.strip(),
        "task_type":   "captioning",
        "source":      source,
    }


def make_classification_example(audio_path: str, label: str, source: str) -> dict:
    return {
        "audio_path":  audio_path,
        "audio2_path": None,
        "question":    random.choice(CLASSIFICATION_QUESTIONS),
        "answer":      label.strip(),
        "task_type":   "classification",
        "source":      source,
    }


def make_unanswerable_example(audio_path: str) -> dict:
    q, a = random.choice(UNANSWERABLE_QUESTIONS)
    return {
        "audio_path":  audio_path,
        "audio2_path": None,
        "question":    q,
        "answer":      a,
        "task_type":   "unanswerable",
        "source":      "synthetic_unanswerable",
    }


# ─────────────────────────────────────────────────────────────────────
# 1. AudioCaps
# ─────────────────────────────────────────────────────────────────────

def process_audiocaps(csv_path: str, split: str) -> list:
    """
    AudioCaps CSV format:
        audiocap_id, youtube_id, start_time, caption
    Audio path convention: data/audio/audiocaps/{split}/Y{youtube_id}.wav
    """
    examples = []
    if not os.path.exists(csv_path):
        print(f"  [AudioCaps] {csv_path} not found — skipping.")
        print(f"  Download from: https://github.com/cdjkim/audiocaps/tree/master/dataset")
        return examples

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            youtube_id = row.get('youtube_id')
            caption    = row.get('caption')
            # Skip malformed rows (3 known bad rows with None fields)
            if not youtube_id or not caption or not caption.strip():
                continue
            audio_path = str(AUDIO_DIR / "audiocaps" / split / f"Y{youtube_id}.wav")
            examples.append(make_caption_example(audio_path, caption, f"audiocaps_{split}"))

    print(f"  [AudioCaps] {split}: {len(examples)} examples")
    return examples


# ─────────────────────────────────────────────────────────────────────
# 2. Clotho
# ─────────────────────────────────────────────────────────────────────

def process_clotho(csv_path: str, split: str) -> list:
    """
    Clotho CSV format:
        file_name, caption_1, caption_2, caption_3, caption_4, caption_5
    Audio path: data/audio/clotho/{split}/{file_name}
    """
    examples = []
    if not os.path.exists(csv_path):
        print(f"  [Clotho] {csv_path} not found — skipping.")
        print(f"  Download from: https://zenodo.org/record/4783391")
        return examples

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['file_name']
            audio_path = str(AUDIO_DIR / "clotho" / split / filename)
            # Use all 5 captions as separate examples
            for i in range(1, 6):
                cap_key = f'caption_{i}'
                if cap_key in row and row[cap_key].strip():
                    examples.append(
                        make_caption_example(audio_path, row[cap_key], f"clotho_{split}")
                    )

    print(f"  [Clotho] {split}: {len(examples)} examples (5 captions × files)")
    return examples


# ─────────────────────────────────────────────────────────────────────
# 3. FSD50K
# ─────────────────────────────────────────────────────────────────────

def process_fsd50k(csv_path: str, split: str) -> list:
    """
    FSD50K ground truth CSV format:
        fname, labels, mids, split
    labels = comma-separated class names (e.g. "Bark,Dog,Animal")
    Audio path: data/audio/fsd50k/{split}/{fname}.wav

    Note: the dev CSV has internal train/val splits — we use ALL rows
    from whichever file is passed in (the filename itself is the split).
    """
    examples = []
    if not os.path.exists(csv_path):
        print(f"  [FSD50K] {csv_path} not found — skipping.")
        print(f"  Download from: https://zenodo.org/record/4835end")
        return examples

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname      = row['fname']
            labels     = row['labels'].replace(',', ', ')
            audio_path = str(AUDIO_DIR / "fsd50k" / split / f"{fname}.wav")
            examples.append(
                make_classification_example(audio_path, labels, f"fsd50k_{split}")
            )

    print(f"  [FSD50K] {split}: {len(examples)} examples")
    return examples


# ─────────────────────────────────────────────────────────────────────
# 4. ClothoAQA binary QA (unanimous subset)
# ─────────────────────────────────────────────────────────────────────

def process_clotho_aqa(csv_path: str, audio_split: str) -> list:
    """
    ClothoAQA format:
        file_name, Question, answer, confidence

    Only use rows where confidence is unanimous (all annotators agreed).

    ClothoAQA split → Clotho audio folder mapping:
        clotho_aqa_train.csv → clotho/development/
        clotho_aqa_val.csv   → clotho/validation/
        clotho_aqa_test.csv  → clotho/evaluation/  ← NEVER use for training

    Args:
        csv_path:    path to the ClothoAQA CSV
        audio_split: Clotho audio subfolder ("development", "validation", "evaluation")
    """
    examples = []
    if not os.path.exists(csv_path):
        print(f"  [ClothoAQA] {csv_path} not found — skipping.")
        print(f"  Download from: https://zenodo.org/record/6473207")
        return examples

    unanimous_count = 0
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only unanimous answers (all 3 annotators agreed)
            if row.get('confidence', '').lower() not in ('yes', 'unanimous', '3'):
                continue

            filename   = row['file_name']
            question   = row['QuestionText'].strip()
            answer     = row['answer'].strip()
            # Audio lives in the Clotho folder matching the split
            audio_path = str(AUDIO_DIR / "clotho" / audio_split / filename)

            examples.append({
                "audio_path":  audio_path,
                "audio2_path": None,
                "question":    question,
                "answer":      answer,
                "task_type":   "binary",
                "source":      f"clotho_aqa_{audio_split}",
            })
            unanimous_count += 1

    print(f"  [ClothoAQA {audio_split}] unanimous: {unanimous_count} examples")
    return examples


# ─────────────────────────────────────────────────────────────────────
# 5. Inject unanswerable queries
# ─────────────────────────────────────────────────────────────────────

def inject_unanswerable(examples: list, rate: float = 0.06) -> list:
    """
    Inject unanswerable queries at `rate` of total examples.
    Uses audio paths from existing examples (real audio, wrong question).
    From LTU: this reduces hallucination rate from 48% to 31%.
    """
    n_inject = int(len(examples) * rate)
    audio_paths = [e['audio_path'] for e in examples]

    injected = []
    for _ in range(n_inject):
        audio_path = random.choice(audio_paths)
        injected.append(make_unanswerable_example(audio_path))

    print(f"  [Unanswerable] Injected {len(injected)} examples ({rate*100:.0f}%)")
    return examples + injected


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("\nStage 1 Data Preparation")
    print("=" * 55)

    train_examples = []
    val_examples   = []

    # ── AudioCaps ────────────────────────────────────────────────
    print("\nProcessing AudioCaps...")
    train_examples += process_audiocaps(
        str(META_DIR / "audiocaps_train.csv"), "train"
    )
    val_examples   += process_audiocaps(
        str(META_DIR / "audiocaps_val.csv"), "val"
    )

    # ── Clotho ───────────────────────────────────────────────────
    print("\nProcessing Clotho...")
    train_examples += process_clotho(
        str(META_DIR / "clotho_captions_development.csv"), "development"
    )
    val_examples   += process_clotho(
        str(META_DIR / "clotho_captions_validation.csv"), "validation"
    )

    # ── FSD50K ───────────────────────────────────────────────────
    print("\nProcessing FSD50K...")
    train_examples += process_fsd50k(
        str(META_DIR / "fsd50k_ground_truth_dev.csv"), "dev"
    )
    val_examples   += process_fsd50k(
        str(META_DIR / "fsd50k_ground_truth_eval.csv"), "eval"
    )

    # ── ClothoAQA binary ────────────────────────────────────────
    # train CSV  → clotho/development/ audio
    # val CSV    → clotho/validation/  audio
    # test CSV   → clotho/evaluation/  audio  ← HELD OUT, never train on this
    print("\nProcessing ClothoAQA...")
    train_examples += process_clotho_aqa(
        str(META_DIR / "clotho_aqa_train.csv"), "development"
    )
    val_examples += process_clotho_aqa(
        str(META_DIR / "clotho_aqa_val.csv"), "validation"
    )
    # test split is benchmark — save separately, do NOT add to train or val
    test_aqa = process_clotho_aqa(
        str(META_DIR / "clotho_aqa_test.csv"), "evaluation"
    )
    if test_aqa:
        import json as _json
        test_path = OUT_DIR / "stage1_test_aqa.jsonl"
        with open(test_path, "w") as _f:
            for ex in test_aqa:
                _f.write(_json.dumps(ex) + "\n")
        print(f"  [ClothoAQA test] {len(test_aqa)} examples → {test_path}  (eval only, not trained on)")


    # ── Unanswerable injection ───────────────────────────────────
    print("\nInjecting unanswerable queries...")
    if train_examples:
        train_examples = inject_unanswerable(train_examples, rate=0.06)
    if val_examples:
        val_examples = inject_unanswerable(val_examples, rate=0.06)

    # ── Shuffle ──────────────────────────────────────────────────
    random.shuffle(train_examples)
    random.shuffle(val_examples)

    # ── Write JSONL ──────────────────────────────────────────────
    train_path = OUT_DIR / "stage1_train.jsonl"
    val_path   = OUT_DIR / "stage1_val.jsonl"

    with open(train_path, 'w') as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + '\n')

    with open(val_path, 'w') as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + '\n')

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Stage 1 Data Summary")
    print(f"{'='*55}")
    print(f"  Train: {len(train_examples):,} examples → {train_path}")
    print(f"  Val:   {len(val_examples):,} examples → {val_path}")
    print()

    # Task type breakdown
    for split_name, exs in [("Train", train_examples), ("Val", val_examples)]:
        from collections import Counter
        counts = Counter(e['task_type'] for e in exs)
        print(f"  {split_name} breakdown:")
        for task, count in sorted(counts.items()):
            print(f"    {task:<20} {count:>8,}")
        print()

    print(f"  Metadata files needed (put in {META_DIR}/):")
    print(f"    audiocaps_train.csv          → https://github.com/cdjkim/audiocaps/tree/master/dataset")
    print(f"    audiocaps_val.csv")
    print(f"    clotho_captions_development.csv → https://zenodo.org/record/4783391")
    print(f"    clotho_captions_validation.csv")
    print(f"    fsd50k_ground_truth_dev.csv    → https://zenodo.org/record/4835end")
    print(f"    fsd50k_ground_truth_eval.csv")
    print(f"    clotho_aqa_train.csv           → https://zenodo.org/record/6473207 clotho_aqa_val.csv             → (same Zenodo record) clotho_aqa_test.csv            → (same Zenodo record, eval only — do NOT train on)")
    print(f"  Next: python download_audio.py   (downloads actual .wav files)")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()