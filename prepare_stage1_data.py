"""
prepare_stage1_data.py — Stage 1: Mapper Pre-Alignment Data
Run from ALM/ directory:
    python prepare_stage1_data.py

Builds metadata JSONL for Stage 1 training (~335K examples).
Does NOT download audio — just creates the metadata mapping.
Audio download is a separate step per dataset.

Stage 1 datasets (per training plan v6):
  Sound:   AudioCaps (~91K), Clotho (~14.5K), FSD50K (~41K), ClothoAQA (~14K)
  Music:   MusicCaps (~5.5K), JamendoMaxCaps (~5K curated subset)
  Speech:  LibriSpeech clean-100+360 (~160K with augmentation)
  Safety:  Unanswerable injection (~6% of total)

Output: data/stage1/stage1_train.jsonl, data/stage1/stage1_val.jsonl

Each JSONL line:
{
  "audio_path": "data/audio/audiocaps/train/Y<youtube_id>.wav",
  "question":   "Describe what you hear in the audio.",
  "answer":     "A dog is barking followed by a door slamming.",
  "task_type":  "captioning",
  "encoder_type": "audio",     # "audio" | "speech" | "both"
  "source":     "audiocaps_train"
}

encoder_type tells the training script which encoder path to use:
  "audio"  → OpenBEATs → AudioMapper  (sound / music examples)
  "speech" → Whisper → SpeechMapper   (speech examples)
  "both"   → both encoders            (unanswerable, mixed)
"""

import json
import os
import random
import csv
import re
from pathlib import Path
from collections import Counter

random.seed(42)

DATA_DIR  = Path("data")
AUDIO_DIR = DATA_DIR / "audio"
META_DIR  = DATA_DIR / "metadata"
OUT_DIR   = DATA_DIR / "stage1"

for d in [DATA_DIR, AUDIO_DIR, META_DIR, OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# QA Templates
# ═══════════════════════════════════════════════════════════════════════

CAPTION_QUESTIONS = [
    "Describe what you hear in the audio.",
    "Generate a caption describing the sounds in this audio clip.",
    "What sounds are present in the audio?",
    "Describe the auditory scene in this audio.",
    "What can you hear in this audio recording?",
]

MUSIC_CAPTION_QUESTIONS = [
    "Describe the music in this audio.",
    "What kind of music is playing?",
    "Describe the musical content of this audio.",
    "What instruments and genre can you identify in this music?",
    "Provide a description of this music piece.",
]

CLASSIFICATION_QUESTIONS = [
    "What type of sound is this?",
    "What category best describes the sounds in this audio?",
    "Identify the primary sound event in this audio.",
]

TRANSCRIPTION_QUESTIONS = [
    "What is the person saying?",
    "Transcribe the speech in this audio.",
    "What words are being spoken in this recording?",
]

# Varied speech-description templates (avoid near-duplicates)
# Uses LibriSpeech SPEAKERS.TXT metadata: gender
SPEECH_DESCRIPTION_TEMPLATES = [
    "A {gender} speaker reads an audiobook excerpt in clear English.",
    "A {gender} voice speaking at a {pace} pace with clear pronunciation.",
    "An audiobook narration by a {gender} speaker with {register} tone.",
    "Clear English speech by a {gender} narrator reading prose.",
    "{gender_cap} speaker delivers a passage with {quality} enunciation.",
]
PACE_OPTIONS = ["measured", "steady", "moderate", "deliberate"]
REGISTER_OPTIONS = ["neutral", "calm", "warm", "composed"]
QUALITY_OPTIONS = ["clear", "precise", "careful", "distinct"]

# ── 3-Type Stratified Unanswerable Taxonomy ──────────────────────────

# Type 1: Existence contradictions — ask about events that DON'T exist
UNANSWERABLE_TYPE1_SOUND = [
    ("What is the person saying?", "This audio does not contain any speech. It contains only environmental sounds."),
    ("Transcribe the speech in this audio.", "No speech is present in this audio clip. Only non-speech sounds are audible."),
    ("What language is being spoken?", "No speech is present in this audio. It contains only environmental sounds."),
    ("Who is the speaker in this audio?", "This audio contains no speech. Only non-speech audio events are present."),
]
UNANSWERABLE_TYPE1_SPEECH = [
    ("What instrument is being played?", "This audio does not contain any musical instruments. Only speech is present."),
    ("What animal can you hear?", "No animal sounds are present. This audio contains only human speech."),
    ("Describe the environmental sounds.", "This audio contains only speech with no notable environmental sounds."),
]
UNANSWERABLE_TYPE1_MUSIC = [
    ("What is the person saying?", "This audio contains music, not speech. No words are being spoken."),
    ("What type of vehicle is in this audio?", "No vehicle sounds are present. This audio contains only music."),
]

# Type 2: Attribute ambiguity — correct domain but impossible specificity
UNANSWERABLE_TYPE2 = [
    ("What is the exact make and model of the vehicle?", "The audio provides insufficient acoustic detail to identify the specific make and model. Only general engine sounds are present."),
    ("How old is the person speaking?", "The audio does not provide enough information to determine the speaker's exact age."),
    ("What brand of instrument is being played?", "The audio does not contain sufficient detail to identify the specific brand of the instrument."),
    ("What is the exact location where this was recorded?", "The recording location cannot be determined from audio content alone."),
    ("How many people are in the room?", "The exact number of people cannot be reliably determined from this audio."),
]

# Type 3: Logical traps / false premises
UNANSWERABLE_TYPE3 = [
    ("Which of the two guitarists is playing the melody?", "This audio does not contain guitar sounds. The premise of the question is incorrect."),
    ("After the piano stops, what does the violinist play?", "This audio does not contain both piano and violin. The premise of the question is incorrect."),
    ("Is the second explosion louder than the first?", "This audio does not contain multiple explosions. The premise of the question is incorrect."),
    ("The dog in this audio sounds aggressive, right?", "There is no dog in this audio. The premise of the question is incorrect."),
    ("Which bird species is singing in the left channel?", "The audio does not provide sufficient information to identify specific bird species or channel location. The premise may be incorrect."),
]


# ═══════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════

def make_example(audio_path, question, answer, task_type, encoder_type, source):
    """Create a single training example."""
    return {
        "audio_path":   str(audio_path),
        "question":     question,
        "answer":       answer.strip(),
        "task_type":    task_type,
        "encoder_type": encoder_type,
        "source":       source,
    }


def word_count(text):
    """Count words in a string."""
    return len(text.split())


def jaccard_similarity(s1, s2):
    """Compute Jaccard similarity between two strings (word-level)."""
    words1 = set(s1.lower().split())
    words2 = set(s2.lower().split())
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)


def generate_speech_description(gender):
    """Generate a varied speech description from templates."""
    template = random.choice(SPEECH_DESCRIPTION_TEMPLATES)
    gender_word = "male" if gender.lower() in ("m", "male") else "female"
    return template.format(
        gender=gender_word,
        gender_cap=gender_word.capitalize(),
        pace=random.choice(PACE_OPTIONS),
        register=random.choice(REGISTER_OPTIONS),
        quality=random.choice(QUALITY_OPTIONS),
    )


# ═══════════════════════════════════════════════════════════════════════
# 1. AudioCaps (~91K train, ~2.5K val)
# ═══════════════════════════════════════════════════════════════════════

def process_audiocaps(csv_path, split):
    """
    AudioCaps format: audiocap_id, youtube_id, start_time, caption
    Audio path: data/audio/audiocaps/{split}/Y{youtube_id}.wav
    Keep ALL rows — each caption is a separate valid training example.
    ~91K train rows from ~46K unique clips (avg ~2 captions/clip).
    """
    examples = []
    if not os.path.exists(csv_path):
        print(f"  [AudioCaps] {csv_path} not found — skipping.")
        print(f"  Download from: https://github.com/cdjkim/audiocaps/tree/master/dataset")
        return examples

    skipped = 0
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            youtube_id = row.get('youtube_id')
            caption    = row.get('caption')
            if not youtube_id or not caption or not caption.strip():
                skipped += 1
                continue
            # Drop captions < 5 words (per training plan)
            if word_count(caption) < 5:
                skipped += 1
                continue
            audio_path = AUDIO_DIR / "audiocaps" / split / f"Y{youtube_id}.wav"
            examples.append(make_example(
                audio_path, random.choice(CAPTION_QUESTIONS),
                caption, "captioning", "audio", f"audiocaps_{split}"
            ))

    print(f"  [AudioCaps] {split}: {len(examples):,} examples ({skipped} skipped)")
    return examples


# ═══════════════════════════════════════════════════════════════════════
# 2. Clotho (~14.5K dev, ~5K val)
# ═══════════════════════════════════════════════════════════════════════

def process_clotho(csv_path, split):
    """
    Clotho format: file_name, caption_1..caption_5
    Audio path: data/audio/clotho/{split}/{file_name}
    All 5 captions per file are separate examples.
    Drop captions shorter than 8 words (per training plan).
    Drop exact duplicate captions for same file.
    """
    examples = []
    if not os.path.exists(csv_path):
        print(f"  [Clotho] {csv_path} not found — skipping.")
        print(f"  Download from: https://zenodo.org/record/4783391")
        return examples

    dropped_short = 0
    dropped_dup = 0
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['file_name']
            audio_path = AUDIO_DIR / "clotho" / split / filename
            seen_captions = set()
            for i in range(1, 6):
                cap_key = f'caption_{i}'
                if cap_key in row and row[cap_key].strip():
                    caption = row[cap_key].strip()
                    # Drop short captions
                    if word_count(caption) < 8:
                        dropped_short += 1
                        continue
                    # Drop exact duplicates for same file
                    if caption.lower() in seen_captions:
                        dropped_dup += 1
                        continue
                    seen_captions.add(caption.lower())
                    examples.append(make_example(
                        audio_path, random.choice(CAPTION_QUESTIONS),
                        caption, "captioning", "audio", f"clotho_{split}"
                    ))

    print(f"  [Clotho] {split}: {len(examples):,} examples "
          f"(dropped: {dropped_short} short, {dropped_dup} dup)")
    return examples


# ═══════════════════════════════════════════════════════════════════════
# 3. FSD50K (~41K dev, ~11K eval for val)
# ═══════════════════════════════════════════════════════════════════════

def process_fsd50k(csv_path, split):
    """
    FSD50K format: fname, labels, mids, split
    labels = comma-separated class names (e.g. "Bark,Dog,Animal")
    Audio path: data/audio/fsd50k/{split}/{fname}.wav
    Keep ALL rows — empty labels are dropped.
    """
    examples = []
    if not os.path.exists(csv_path):
        print(f"  [FSD50K] {csv_path} not found — skipping.")
        print(f"  Download from: https://zenodo.org/records/4835end")
        return examples

    skipped = 0
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname  = row['fname']
            labels = row.get('labels', '').strip()
            if not labels:
                skipped += 1
                continue
            # Prettify: "Electric_guitar,Guitar" → "Electric guitar, Guitar"
            labels_pretty = ", ".join(
                label.replace('_', ' ') for label in labels.split(',')
            )
            audio_path = AUDIO_DIR / "fsd50k" / split / f"{fname}.wav"
            examples.append(make_example(
                audio_path, random.choice(CLASSIFICATION_QUESTIONS),
                labels_pretty, "classification", "audio", f"fsd50k_{split}"
            ))

    print(f"  [FSD50K] {split}: {len(examples):,} examples ({skipped} skipped)")
    return examples


# ═══════════════════════════════════════════════════════════════════════
# 4. ClothoAQA — Binary QA (unanimous only)
# ═══════════════════════════════════════════════════════════════════════

def process_clotho_aqa(csv_path, audio_split):
    """
    ClothoAQA format: file_name, QuestionText, answer, confidence
    Only keep rows where confidence == 'yes' (case-insensitive) = unanimous.
    Audio lives in Clotho folders: clotho/{audio_split}/

    ClothoAQA split → Clotho audio mapping:
        clotho_aqa_train.csv → clotho/development/
        clotho_aqa_val.csv   → clotho/validation/
        clotho_aqa_test.csv  → NEVER train on
    """
    examples = []
    if not os.path.exists(csv_path):
        print(f"  [ClothoAQA] {csv_path} not found — skipping.")
        print(f"  Download from: https://zenodo.org/record/6473207")
        return examples

    total = 0
    unanimous = 0
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            # Fix: case-insensitive check (CSV has 'yes' and 'Yes')
            confidence = row.get('confidence', '').strip().lower()
            if confidence != 'yes':
                continue
            unanimous += 1

            filename = row['file_name']
            question = row['QuestionText'].strip()
            answer   = row['answer'].strip()
            audio_path = AUDIO_DIR / "clotho" / audio_split / filename

            examples.append(make_example(
                audio_path, question, answer,
                "binary_qa", "audio", f"clotho_aqa_{audio_split}"
            ))

    print(f"  [ClothoAQA] {audio_split}: {unanimous:,} unanimous / {total:,} total")
    return examples


# ═══════════════════════════════════════════════════════════════════════
# 5. MusicCaps (~5.5K) — via HuggingFace datasets
# ═══════════════════════════════════════════════════════════════════════

def process_musiccaps():
    """
    MusicCaps: 5,521 human-written music captions (Google).
    Audio sourced from YouTube via ytid.
    Audio path: data/audio/musiccaps/{ytid}.wav
    Keep ALL — small, high quality, human-written.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [MusicCaps] 'datasets' library not installed. pip install datasets")
        return [], []

    print("  [MusicCaps] Loading from HuggingFace...")
    try:
        ds = load_dataset("google/MusicCaps", split="train")
    except Exception as e:
        print(f"  [MusicCaps] Failed to load: {e}")
        return [], []

    examples = []
    for row in ds:
        ytid = row.get("ytid", "")
        caption = row.get("caption", "")
        if not ytid or not caption or not caption.strip():
            continue

        audio_path = AUDIO_DIR / "musiccaps" / f"{ytid}.wav"
        examples.append(make_example(
            audio_path, random.choice(MUSIC_CAPTION_QUESTIONS),
            caption, "music_captioning", "audio", "musiccaps"
        ))

    # 90/10 split for train/val
    random.shuffle(examples)
    split_idx = int(len(examples) * 0.9)
    train = examples[:split_idx]
    val   = examples[split_idx:]

    print(f"  [MusicCaps] {len(train):,} train + {len(val):,} val = {len(examples):,} total")
    return train, val


# ═══════════════════════════════════════════════════════════════════════
# 6. JamendoMaxCaps (362K → 5K genre-stratified subset)
# ═══════════════════════════════════════════════════════════════════════
#
# Data layout on HuggingFace (amaai-lab/JamendoMaxCaps):
#   data/*.parquet          → audio files only (1.1TB) — DO NOT TOUCH
#   final_caption30sec.jsonl → captions (id, caption, start_time, end_time)
#   YYYY-MM-DD.jsonl         → metadata (id, duration, genres, speed, etc.)
#
# We download only the caption + metadata JSONL files (pure text, ~200MB total).

JAMENDO_CACHE = META_DIR / "jamendo_metadata_cache.csv"
JAMENDO_HF_BASE = "https://huggingface.co/datasets/amaai-lab/JamendoMaxCaps/resolve/main"

def _download_jamendo_metadata():
    """Download JamendoMaxCaps caption + metadata JSONL files from HuggingFace.

    1. Downloads final_caption30sec.jsonl (captions by Qwen2-Audio)
    2. Lists and downloads YYYY-MM-DD.jsonl metadata files
    3. Joins on track ID → saves combined CSV cache
    """
    import urllib.request
    from huggingface_hub import HfApi

    # ── Step 1: Download captions ──
    caption_url = f"{JAMENDO_HF_BASE}/final_caption30sec.jsonl"
    print(f"    Downloading captions from {caption_url}...")

    captions_by_id = {}  # {id: caption}
    try:
        with urllib.request.urlopen(caption_url) as resp:
            for line_bytes in resp:
                line = line_bytes.decode('utf-8').strip()
                if not line:
                    continue
                obj = json.loads(line)
                track_id = str(obj.get("id", ""))
                caption = obj.get("caption", "")
                if track_id and caption:
                    # Keep first caption per track (30sec segment)
                    if track_id not in captions_by_id:
                        captions_by_id[track_id] = caption
    except Exception as e:
        print(f"    Failed to download captions: {e}")
        return False

    print(f"    Loaded {len(captions_by_id):,} unique captions")

    # ── Step 2: List metadata JSONL files ──
    print("    Listing metadata JSONL files...")
    api = HfApi()
    try:
        all_files = api.list_repo_tree(
            "amaai-lab/JamendoMaxCaps",
            repo_type="dataset",
        )
        jsonl_files = sorted([
            f.path for f in all_files
            if f.path.endswith('.jsonl') and f.path != 'final_caption30sec.jsonl'
        ])
    except Exception as e:
        print(f"    Failed to list repo files: {e}")
        return False

    print(f"    Found {len(jsonl_files)} metadata JSONL files")

    # ── Step 3: Download metadata and join with captions ──
    metadata_by_id = {}  # {id: {duration, genres, speed, vartags}}

    for i, jsonl_path in enumerate(jsonl_files):
        url = f"{JAMENDO_HF_BASE}/{jsonl_path}"
        try:
            with urllib.request.urlopen(url) as resp:
                for line_bytes in resp:
                    line = line_bytes.decode('utf-8').strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    track_id = str(obj.get("id", ""))
                    if not track_id:
                        continue

                    duration = obj.get("duration", 0)
                    musicinfo = obj.get("musicinfo", {})
                    speed = musicinfo.get("speed", "")
                    tags = musicinfo.get("tags", {})
                    genres = ",".join(tags.get("genres", []))
                    vartags = ",".join(tags.get("vartags", []))

                    metadata_by_id[track_id] = {
                        "duration": duration,
                        "genres": genres,
                        "speed": speed,
                        "vartags": vartags,
                    }
        except Exception as e:
            print(f"    Warning: failed to download {jsonl_path}: {e}")
            continue

        if (i + 1) % 50 == 0 or i == len(jsonl_files) - 1:
            print(f"    Read {i+1}/{len(jsonl_files)} metadata files ({len(metadata_by_id):,} tracks)")

    # ── Step 4: Join captions + metadata → CSV ──
    import csv as csv_module
    fieldnames = ["track_id", "caption", "duration", "genres", "speed", "vartags"]
    rows_written = 0

    with open(JAMENDO_CACHE, 'w', newline='', encoding='utf-8') as f:
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for track_id, caption in captions_by_id.items():
            meta = metadata_by_id.get(track_id, {})
            writer.writerow({
                "track_id": track_id,
                "caption": caption,
                "duration": meta.get("duration", 0),
                "genres": meta.get("genres", ""),
                "speed": meta.get("speed", ""),
                "vartags": meta.get("vartags", ""),
            })
            rows_written += 1

    print(f"    Cached {rows_written:,} tracks → {JAMENDO_CACHE}")
    return True


def process_jamendo(musiccaps_captions=None):
    """
    JamendoMaxCaps: 362K CC-licensed instrumental tracks from Jamendo.
    Captions are synthetic (Qwen2-Audio generated).

    Two-step approach:
      1. First run: downloads text-only metadata via pyarrow (no audio),
         caches to data/metadata/jamendo_metadata_cache.csv
      2. Subsequent runs: reads from cache instantly

    Filtering (per training plan):
      1. Skip tracks < 15s
      2. Genre-stratified: 1K Jazz + 1K Classical + 1K World/Folk + 1K Ambient + 1K random
      3. Caption quality: 15-100 words, mention ≥2 music attributes
      4. Drop hallucinated vocals (keyword filter)
      5. Dedup against MusicCaps (Jaccard > 0.6)
    """
    # Step 1: Get metadata (from cache or download)
    if not JAMENDO_CACHE.exists():
        print("  [JamendoMaxCaps] No local cache found. Downloading text metadata...")
        try:
            success = _download_jamendo_metadata()
            if not success:
                print("  [JamendoMaxCaps] Download failed — skipping. (MusicCaps 5.5K still available)")
                return []
        except ImportError as e:
            print(f"  [JamendoMaxCaps] Missing dependency: {e}")
            print("  [JamendoMaxCaps] pip install pyarrow huggingface_hub")
            return []
        except Exception as e:
            print(f"  [JamendoMaxCaps] Error: {e} — skipping.")
            return []
    else:
        print(f"  [JamendoMaxCaps] Reading from cache: {JAMENDO_CACHE}")

    # Step 2: Read from CSV and apply filters
    rows = []
    with open(JAMENDO_CACHE, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"    Loaded {len(rows):,} rows from cache")

    # Vocal hallucination keywords
    vocal_keywords = re.compile(r'\b(singing|vocalist|vocals|singer|sings|sung|lyric|lyrics)\b', re.IGNORECASE)
    # Music attribute keywords (need ≥2 for quality)
    music_attrs = re.compile(
        r'\b(instrument|guitar|piano|drum|bass|violin|flute|saxophone|trumpet|'
        r'genre|rock|jazz|classical|electronic|ambient|folk|blues|'
        r'mood|calm|energetic|melancholy|upbeat|peaceful|'
        r'rhythm|tempo|beat|BPM|slow|fast|moderate)\b', re.IGNORECASE
    )

    # Collect by genre bucket
    genre_buckets = {
        "jazz":      [],
        "classical": [],
        "world_folk": [],
        "ambient":   [],
        "other":     [],
    }
    TARGET_PER_BUCKET = 1000

    skipped_stats = Counter()

    for row in rows:
        caption  = (row.get("caption") or "").strip()
        genres   = (row.get("genres") or "").lower()
        track_id = row.get("track_id", "")
        try:
            duration = float(row.get("duration", 30))
        except (ValueError, TypeError):
            duration = 30

        # Filter 1: Skip short tracks
        if duration < 15:
            skipped_stats["short"] += 1
            continue

        # Filter 3: Caption length 15-100 words
        wc = word_count(caption)
        if wc < 15 or wc > 100:
            skipped_stats["caption_length"] += 1
            continue

        # Filter 4: Drop hallucinated vocals
        if vocal_keywords.search(caption):
            skipped_stats["vocal_hallucination"] += 1
            continue

        # Filter 3b: Must mention ≥2 music attributes
        attr_matches = music_attrs.findall(caption)
        if len(set(attr_matches)) < 2:
            skipped_stats["few_attributes"] += 1
            continue

        # Filter 5: Dedup against MusicCaps (sample check for speed)
        if musiccaps_captions:
            is_dup = False
            check_caps = random.sample(musiccaps_captions, min(100, len(musiccaps_captions)))
            for mc_cap in check_caps:
                if jaccard_similarity(caption, mc_cap) > 0.6:
                    is_dup = True
                    break
            if is_dup:
                skipped_stats["musiccaps_dup"] += 1
                continue

        # Determine genre bucket
        if "jazz" in genres:
            bucket = "jazz"
        elif "classical" in genres:
            bucket = "classical"
        elif any(g in genres for g in ["world", "folk", "ethnic", "traditional"]):
            bucket = "world_folk"
        elif any(g in genres for g in ["ambient", "chillout", "downtempo", "newage"]):
            bucket = "ambient"
        else:
            bucket = "other"

        # Only add if bucket not full yet
        if len(genre_buckets[bucket]) < TARGET_PER_BUCKET:
            audio_path = AUDIO_DIR / "jamendo" / f"{track_id}.wav"
            example = make_example(
                audio_path, random.choice(MUSIC_CAPTION_QUESTIONS),
                caption, "music_captioning", "audio", "jamendo"
            )
            genre_buckets[bucket].append(example)

    # Final selection
    selected = []
    for bucket_name, bucket_examples in genre_buckets.items():
        random.shuffle(bucket_examples)
        n_take = min(TARGET_PER_BUCKET, len(bucket_examples))
        selected.extend(bucket_examples[:n_take])
        print(f"    {bucket_name}: {n_take:,} / {len(bucket_examples):,} available")

    print(f"  [JamendoMaxCaps] Selected {len(selected):,} examples")
    print(f"    Skipped: {dict(skipped_stats)}")
    return selected


# ═══════════════════════════════════════════════════════════════════════
# 7. LibriSpeech clean-100 + clean-360
# ═══════════════════════════════════════════════════════════════════════
#
# Two download modes:
#   A) HuggingFace streaming with monkey-patched Audio decoder (preferred)
#      - Patches Audio.decode_example to skip torchcodec
#      - Streams text fields (id, text, speaker_id, chapter_id) only
#      - Caches to local CSV for instant re-runs
#   B) Local .trans.txt file parsing (fallback)
#      - Requires manual download from openslr.org

LIBRISPEECH_CACHE = META_DIR / "librispeech_metadata_cache.csv"

def _download_librispeech_metadata():
    """Download LibriSpeech text metadata via HuggingFace streaming.

    Monkey-patches Audio.decode_example to prevent torchcodec import.
    The audio column returns raw struct data (ignored), while we extract
    only text, speaker_id, chapter_id, and id.
    """
    import datasets.features.audio as audio_mod
    from datasets import load_dataset
    import csv as csv_module

    # Monkey-patch Audio decoder to skip torchcodec
    _orig_decode = audio_mod.Audio.decode_example
    audio_mod.Audio.decode_example = lambda self, value, *args, **kwargs: value

    fieldnames = ["id", "text", "speaker_id", "chapter_id", "split"]
    all_rows = []

    splits = {
        "train.clean.100": 28539,
        "train.clean.360": 104014,
    }

    try:
        for split_name, expected_count in splits.items():
            print(f"    Streaming {split_name} (~{expected_count:,} rows)...")
            ds = load_dataset(
                "openslr/librispeech_asr",
                split=split_name,
                streaming=True,
            )
            count = 0
            for row in ds:
                all_rows.append({
                    "id": row.get("id", ""),
                    "text": row.get("text", ""),
                    "speaker_id": str(row.get("speaker_id", "")),
                    "chapter_id": str(row.get("chapter_id", "")),
                    "split": split_name,
                })
                count += 1
                if count % 10000 == 0:
                    print(f"      ... {count:,}/{expected_count:,}")

            print(f"    {split_name}: {count:,} rows streamed")
    finally:
        # Restore original decoder
        audio_mod.Audio.decode_example = _orig_decode

    if not all_rows:
        return False

    # Save to CSV cache
    with open(LIBRISPEECH_CACHE, 'w', newline='', encoding='utf-8') as f:
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"    Cached {len(all_rows):,} rows → {LIBRISPEECH_CACHE}")
    return True


# Speaker gender cache
_LIBRISPEECH_GENDER_CACHE = {}

def _load_speaker_gender_cache():
    """Load speaker gender mapping from SPEAKERS.TXT or local cache."""
    global _LIBRISPEECH_GENDER_CACHE
    cache_file = META_DIR / "librispeech_speakers.json"

    # Try cache first
    if cache_file.exists():
        with open(cache_file) as f:
            _LIBRISPEECH_GENDER_CACHE = json.load(f)
        return

    # Parse SPEAKERS.TXT if available
    speakers_txt = AUDIO_DIR / "librispeech" / "SPEAKERS.TXT"
    if speakers_txt.exists():
        with open(speakers_txt) as f:
            for line in f:
                if line.startswith(";") or not line.strip():
                    continue
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    sid = parts[0].strip()
                    gender = parts[1].strip().upper()
                    _LIBRISPEECH_GENDER_CACHE[sid] = "male" if gender == "M" else "female"
        with open(cache_file, "w") as f:
            json.dump(_LIBRISPEECH_GENDER_CACHE, f)
        print(f"  [LibriSpeech] Loaded {len(_LIBRISPEECH_GENDER_CACHE)} speaker genders from SPEAKERS.TXT")
        return

    print("  [LibriSpeech] No SPEAKERS.TXT found — using speaker_id parity for gender.")


def _get_speaker_gender(speaker_id):
    """Get gender for a speaker, using cache or parity fallback."""
    sid = str(speaker_id)
    if sid in _LIBRISPEECH_GENDER_CACHE:
        return _LIBRISPEECH_GENDER_CACHE[sid]
    return "male" if int(speaker_id) % 2 == 0 else "female"


def _process_librispeech_from_rows(rows):
    """Process LibriSpeech rows (from cache or local) into training examples."""
    _load_speaker_gender_cache()

    train_examples = []
    val_examples = []
    total_trans = 0
    total_augmented = 0
    total_dropped = 0
    split_counts = Counter()

    for row in rows:
        text = (row.get("text") or "").strip()
        speaker_id = row.get("speaker_id", "")
        chapter_id = row.get("chapter_id", "")
        utt_id = row.get("id", "")
        split_name = row.get("split", "train-clean-100")

        # Drop transcripts < 3 words
        if word_count(text) < 3:
            total_dropped += 1
            continue

        # Map to local audio path
        # HF id format: "374-180298-0000" → speaker/chapter/id.flac
        parts = utt_id.split("-") if utt_id else []
        if len(parts) >= 3:
            spk, chap = parts[0], parts[1]
            # Convert split name: "train.clean.100" → "train-clean-100"
            subset = split_name.replace(".", "-")
            audio_path = AUDIO_DIR / "librispeech" / subset / spk / chap / f"{utt_id}.flac"
        else:
            audio_path = AUDIO_DIR / "librispeech" / f"{utt_id}.flac"

        # Transcription example
        train_examples.append(make_example(
            audio_path, random.choice(TRANSCRIPTION_QUESTIONS),
            text, "transcription", "speech",
            f"librispeech_{split_name.replace('.', '-')}"
        ))
        total_trans += 1
        split_counts[split_name] += 1

        # 20% augmentation: speech-description (clean-100 only)
        if "100" in split_name and random.random() < 0.2:
            gender = _get_speaker_gender(speaker_id)
            description = generate_speech_description(gender)
            train_examples.append(make_example(
                audio_path, "Describe the speech in this audio.",
                description, "speech_description", "speech",
                f"librispeech_{split_name.replace('.', '-')}_augmented"
            ))
            total_augmented += 1

    # Use 5% of clean-100 transcription for validation
    if train_examples:
        clean100_examples = [e for e in train_examples
                            if "clean-100" in e["source"] and "augmented" not in e["source"]]
        n_val = int(len(clean100_examples) * 0.05)
        if n_val > 0:
            random.shuffle(clean100_examples)
            val_examples = clean100_examples[:n_val]
            val_paths = {e["audio_path"] for e in val_examples}
            train_examples = [e for e in train_examples if e["audio_path"] not in val_paths]

    for split_name, count in sorted(split_counts.items()):
        print(f"    {split_name}: {count:,} transcription examples")
    print(f"  [LibriSpeech] Transcription: {total_trans:,}, Augmented: {total_augmented:,}, "
          f"Dropped: {total_dropped:,}")
    print(f"  [LibriSpeech] Train: {len(train_examples):,}, Val: {len(val_examples):,}")
    return train_examples, val_examples


def process_librispeech():
    """
    LibriSpeech clean-100 + clean-360 — text metadata.

    Two modes:
      A) HuggingFace streaming (preferred) — monkey-patches Audio decoder
         to skip torchcodec. Downloads text-only metadata, caches to CSV.
      B) Local .trans.txt files (fallback) — requires manual download from
         openslr.org.
    """
    # ── Mode A: HuggingFace streaming with cache ──
    if LIBRISPEECH_CACHE.exists():
        print(f"  [LibriSpeech] Reading from cache: {LIBRISPEECH_CACHE}")
        rows = []
        with open(LIBRISPEECH_CACHE, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        print(f"    Loaded {len(rows):,} rows from cache")
        return _process_librispeech_from_rows(rows)

    # ── Mode B: Try local .trans.txt files ──
    libri_dir = AUDIO_DIR / "librispeech"
    has_local = any(
        (libri_dir / subset).exists()
        for subset in ["train-clean-100", "train-clean-360"]
    )

    if has_local:
        print("  [LibriSpeech] Found local files — parsing .trans.txt...")
        rows = []
        for subset in ["train-clean-100", "train-clean-360"]:
            subset_dir = libri_dir / subset
            if not subset_dir.exists():
                print(f"    {subset}/ not found — skipping")
                continue
            trans_files = sorted(subset_dir.rglob("*.trans.txt"))
            for trans_file in trans_files:
                with open(trans_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(maxsplit=1)
                        if len(parts) < 2:
                            continue
                        utt_id, text = parts
                        spk = utt_id.split("-")[0] if "-" in utt_id else ""
                        chap = utt_id.split("-")[1] if utt_id.count("-") >= 1 else ""
                        rows.append({
                            "id": utt_id,
                            "text": text,
                            "speaker_id": spk,
                            "chapter_id": chap,
                            "split": subset,
                        })
            print(f"    {subset}: {sum(1 for r in rows if r['split'] == subset):,} rows from {len(trans_files)} .trans.txt files")
        return _process_librispeech_from_rows(rows)

    # ── Mode C: Download via HuggingFace streaming ──
    print("  [LibriSpeech] No local files found. Downloading metadata via HuggingFace streaming...")
    print("    (This monkey-patches Audio.decode_example to skip torchcodec)")
    try:
        success = _download_librispeech_metadata()
        if not success:
            print("  [LibriSpeech] Download failed — skipping.")
            return [], []
    except Exception as e:
        print(f"  [LibriSpeech] Error: {e}")
        print(f"  ↳ Manual download: wget https://www.openslr.org/resources/12/train-clean-100.tar.gz")
        return [], []

    # Read from freshly-created cache
    rows = []
    with open(LIBRISPEECH_CACHE, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return _process_librispeech_from_rows(rows)


# ═══════════════════════════════════════════════════════════════════════
# 8. Unanswerable Injection (3-type stratified, 6% of total)
# ═══════════════════════════════════════════════════════════════════════

def inject_unanswerable(examples, rate=0.06):
    """
    Inject unanswerable queries at `rate` of total examples.
    Uses audio paths from existing examples (real audio, wrong question).

    3-type stratified taxonomy (equal distribution):
      Type 1: Existence contradictions (wrong-domain questions)
      Type 2: Attribute ambiguity (over-specific questions)
      Type 3: Logical traps (false premise questions)
    """
    n_inject = int(len(examples) * rate)
    n_per_type = n_inject // 3

    # Separate audio paths by encoder type for Type 1
    sound_paths = [e['audio_path'] for e in examples if e['encoder_type'] == 'audio']
    speech_paths = [e['audio_path'] for e in examples if e['encoder_type'] == 'speech']
    all_paths = [e['audio_path'] for e in examples]

    if not all_paths:
        return examples

    injected = []

    # Type 1: Existence contradictions
    for _ in range(n_per_type):
        if sound_paths and random.random() < 0.6:
            # Sound audio + speech question
            audio_path = random.choice(sound_paths)
            # Distinguish music from general sound
            music_sources = ("musiccaps", "jamendo")
            is_music = any(s in audio_path for s in music_sources)
            if is_music:
                q, a = random.choice(UNANSWERABLE_TYPE1_MUSIC)
            else:
                q, a = random.choice(UNANSWERABLE_TYPE1_SOUND)
        elif speech_paths:
            # Speech audio + sound question
            audio_path = random.choice(speech_paths)
            q, a = random.choice(UNANSWERABLE_TYPE1_SPEECH)
        else:
            audio_path = random.choice(all_paths)
            q, a = random.choice(UNANSWERABLE_TYPE1_SOUND)

        injected.append(make_example(
            audio_path, q, a, "unanswerable", "both", "unanswerable_type1"
        ))

    # Type 2: Attribute ambiguity
    for _ in range(n_per_type):
        audio_path = random.choice(all_paths)
        q, a = random.choice(UNANSWERABLE_TYPE2)
        injected.append(make_example(
            audio_path, q, a, "unanswerable", "both", "unanswerable_type2"
        ))

    # Type 3: Logical traps
    remaining = n_inject - 2 * n_per_type  # handle rounding
    for _ in range(remaining):
        audio_path = random.choice(all_paths)
        q, a = random.choice(UNANSWERABLE_TYPE3)
        injected.append(make_example(
            audio_path, q, a, "unanswerable", "both", "unanswerable_type3"
        ))

    print(f"  [Unanswerable] Injected {len(injected):,} examples ({rate*100:.0f}%)")
    print(f"    Type 1 (existence):  {n_per_type:,}")
    print(f"    Type 2 (attribute):  {n_per_type:,}")
    print(f"    Type 3 (logical):    {remaining:,}")

    return examples + injected


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 65)
    print("  Stage 1 Data Preparation (Training Plan v6)")
    print("=" * 65)

    train_examples = []
    val_examples   = []

    # ── 1. AudioCaps ─────────────────────────────────────────────
    print("\n1. Processing AudioCaps...")
    train_examples += process_audiocaps(str(META_DIR / "audiocaps_train.csv"), "train")
    val_examples   += process_audiocaps(str(META_DIR / "audiocaps_val.csv"), "val")

    # ── 2. Clotho ────────────────────────────────────────────────
    print("\n2. Processing Clotho...")
    train_examples += process_clotho(str(META_DIR / "clotho_captions_development.csv"), "development")
    val_examples   += process_clotho(str(META_DIR / "clotho_captions_validation.csv"), "validation")

    # ── 3. FSD50K ────────────────────────────────────────────────
    print("\n3. Processing FSD50K...")
    train_examples += process_fsd50k(str(META_DIR / "fsd50k_ground_truth_dev.csv"), "dev")
    val_examples   += process_fsd50k(str(META_DIR / "fsd50k_ground_truth_eval.csv"), "eval")

    # ── 4. ClothoAQA ─────────────────────────────────────────────
    print("\n4. Processing ClothoAQA (unanimous only)...")
    train_examples += process_clotho_aqa(str(META_DIR / "clotho_aqa_train.csv"), "development")
    val_examples   += process_clotho_aqa(str(META_DIR / "clotho_aqa_val.csv"), "validation")

    # ── 5. MusicCaps ─────────────────────────────────────────────
    print("\n5. Processing MusicCaps...")
    mc_train, mc_val = process_musiccaps()
    train_examples += mc_train
    val_examples   += mc_val

    # Collect MusicCaps captions for dedup against JamendoMaxCaps
    musiccaps_captions = [e["answer"] for e in mc_train + mc_val]

    # ── 6. JamendoMaxCaps ────────────────────────────────────────
    print("\n6. Processing JamendoMaxCaps (362K → 5K subset)...")
    jamendo_examples = process_jamendo(musiccaps_captions)
    train_examples += jamendo_examples  # All to train (small enough)

    # ── 7. LibriSpeech ───────────────────────────────────────────
    print("\n7. Processing LibriSpeech...")
    libri_train, libri_val = process_librispeech()
    train_examples += libri_train
    val_examples   += libri_val

    # ── 8. Unanswerable injection ────────────────────────────────
    print("\n8. Injecting unanswerable queries (6%)...")
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
    print(f"\n{'=' * 65}")
    print(f"  Stage 1 Data Summary")
    print(f"{'=' * 65}")
    print(f"  Train: {len(train_examples):>8,} examples → {train_path}")
    print(f"  Val:   {len(val_examples):>8,} examples → {val_path}")
    print()

    # Task type breakdown
    for split_name, exs in [("Train", train_examples), ("Val", val_examples)]:
        counts = Counter(e['task_type'] for e in exs)
        sources = Counter(e['source'] for e in exs)
        enc_types = Counter(e['encoder_type'] for e in exs)
        print(f"  {split_name} by task:")
        for task, count in sorted(counts.items()):
            print(f"    {task:<25} {count:>8,}")
        print(f"\n  {split_name} by encoder_type:")
        for enc, count in sorted(enc_types.items()):
            print(f"    {enc:<25} {count:>8,}")
        print(f"\n  {split_name} by source:")
        for src, count in sorted(sources.items(), key=lambda x: -x[1]):
            print(f"    {src:<35} {count:>8,}")
        print()

    # ── Dataset availability status ──────────────────────────────
    print(f"  Dataset Status:")
    datasets_status = [
        ("AudioCaps metadata",       META_DIR / "audiocaps_train.csv"),
        ("Clotho metadata",          META_DIR / "clotho_captions_development.csv"),
        ("FSD50K metadata",          META_DIR / "fsd50k_ground_truth_dev.csv"),
        ("ClothoAQA metadata",       META_DIR / "clotho_aqa_train.csv"),
        ("MusicCaps",                "via HuggingFace datasets"),
        ("JamendoMaxCaps",           "via HuggingFace datasets"),
        ("LibriSpeech transcripts",  AUDIO_DIR / "librispeech" / "train-clean-100"),
    ]
    for name, path in datasets_status:
        if isinstance(path, str):
            status = "✓ auto-download"
        elif Path(path).exists():
            status = "✓ found"
        else:
            status = "✗ not found"
        print(f"    {status:<20} {name}")

    print(f"\n  Next steps:")
    print(f"    1. Download audio files for each dataset (separate step)")
    print(f"    2. Proceed to train_mappers.py (Stage 1 training)")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()