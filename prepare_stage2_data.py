#!/usr/bin/env python3
"""
Stage 2 Data Preparation — Phase A (Grounding) + Phase B (CoT)

METADATA-ONLY: Downloads small JSON files, creates JSONL metadata.
Audio download is handled separately on the GPU machine.

Outputs:
  data/stage2/phase_a_train.jsonl
  data/stage2/phase_a_val.jsonl
  data/stage2/phase_b_train.jsonl
"""

import json
import os
import re
import random
import tarfile
import urllib.request
import urllib.error
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

SEED = 42
random.seed(SEED)

DATA_DIR = Path("data")
AUDIO_DIR = DATA_DIR / "audio"
META_DIR = DATA_DIR / "metadata"
STAGE1_DIR = DATA_DIR / "stage1"
STAGE2_DIR = DATA_DIR / "stage2"
CACHE_DIR = META_DIR / "stage2_cache"

STAGE2_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STAGE1_TRAIN = STAGE1_DIR / "stage1_train.jsonl"
STAGE1_VAL = STAGE1_DIR / "stage1_val.jsonl"

# ── HuggingFace URLs ──
AUDIOSKILLS_BASE = "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/audioskills_xl"
AFTHINK_BASE = "https://huggingface.co/datasets/nvidia/AF-Think/resolve/main/afthink"
AFCOT_BASE = "https://huggingface.co/datasets/nvidia/AF-Think/resolve/main/afcot"
WAVCAPS_JSON_BASE = "https://huggingface.co/datasets/cvssp/WavCaps/resolve/main/json_files"

# ── AudioSkills splits (exact case-sensitive filenames) ──
# Matches audio datasets we have OR plan to download
AUDIOSKILLS_SPLITS = {
    "FSD50k.json": "fsd50k",
    "Clotho-v2.json": "clotho",
    "MusicCaps.json": "musiccaps",
    "CountingQA.json": "fsd50k",
    "ESC-50.json": "esc50",
    "UrbanSound8K.json": "urbansound8k",
    "WavCaps.json": "wavcaps_audioset_sl",    # MCQ for WavCaps audio (32 MB)
    "AudioSet_SL.json": "wavcaps_audioset_sl", # MCQ for AudioSet-SL audio (9.3 MB)
    "SoundBible.json": "wavcaps_soundbible",   # MCQ for SoundBible audio (24 KB)
}


# Subsample caps for large AudioSkills splits
AUDIOSKILLS_SUBSAMPLE = {
    "CountingQA.json": 20000,
    "WavCaps.json": 40000,
}

# ── AF-Think afthink/ splits (MCQ format, no CoT tags) ──
AFTHINK_SPLITS = {
    "FSD50k.json": "fsd50k",
    "Clotho-v2.json": "clotho",
    "MusicCaps.json": "musiccaps",
    "ESC-50.json": "esc50",            # hyphen! (was ESC50.json → 404)
    "UrbanSound8K.json": "urbansound8k",
    "AudioSet_SL.json": "wavcaps_audioset_sl",
}


# ── AF-Think afcot/ splits (structured CoT, for extractive compression) ──
AFCOT_SPLITS = {
    "FSD50K.json": "fsd50k",         # capital K
    "Clotho-v2.json": "clotho",
    "MusicCaps.json": "musiccaps",
    "Clotho-AQA.json": "clotho",
    "ESC50.json": "esc50",
    "UrbanSound8K.json": "urbansound8k",
    "AudioSet_SL.json": "wavcaps_audioset_sl",
    "SoundBible.json": "wavcaps_soundbible",
    "WavCaps.json": "wavcaps_audioset_sl",   # CoT for WavCaps audio (53 MB)
}


# ── WavCaps JSON caption files (metadata only, audio downloaded separately) ──
WAVCAPS_SUBSETS = {
    "SoundBible": {
        "json_path": "SoundBible/sb_final.json",
        "audio_dir": "wavcaps_soundbible",  # expected: data/audio/wavcaps_soundbible/
        "description": "Diverse sound effects (bells, thunder, animals, etc.)",
    },
    "AudioSet_SL": {
        "json_path": "AudioSet_SL/as_final.json",
        "audio_dir": "wavcaps_audioset_sl",  # expected: data/audio/wavcaps_audioset_sl/
        "description": "Strongly-labelled AudioSet clips with ChatGPT-cleaned captions",
    },
}

# ── NSynth: metadata extracted by streaming tar.gz (only examples.json, no audio) ──
NSYNTH_TAR_URLS = {
    "valid": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz",
    "test": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz",
}
NSYNTH_SUBSAMPLE = 8000
NSYNTH_AUDIO_DIR = "nsynth"  # data/audio/nsynth/

NSYNTH_Q_TEMPLATES = [
    "What instrument family is playing in this audio?",
    "Identify the type of instrument in this recording.",
    "What kind of musical instrument do you hear?",
    "Classify the instrument being played.",
]
NSYNTH_SOURCE_Q_TEMPLATES = [
    "Is this instrument acoustic, electronic, or synthetic?",
    "What is the source type of this instrument?",
]

# ── Unanswerable templates (Stage 2 — different from Stage 1) ──
UNANSWERABLE_T1_Q = [
    "What language is the person speaking?",
    "Transcribe the conversation in this audio.",
    "How many people are talking in this audio?",
    "What words are being whispered?",
    "What is the person saying at the beginning?",
    "Translate the speech in this recording.",
]
UNANSWERABLE_T1_A = [
    "This audio does not contain any discernible speech or conversation.",
    "No speech is present in this recording. The audio contains only non-speech sounds.",
    "This audio clip contains no spoken language that can be transcribed.",
]
UNANSWERABLE_T2_Q = [
    "What brand of instrument is being played?",
    "What is the exact distance to the sound source?",
    "What color is the object making this sound?",
    "How old is the singer in this recording?",
    "What is the room temperature where this was recorded?",
    "What model of car is producing this engine noise?",
]
UNANSWERABLE_T2_A = [
    "The audio does not contain sufficient acoustic information to determine this specific detail.",
    "This level of specificity cannot be reliably determined from audio alone.",
    "The recording provides insufficient detail to identify this attribute.",
]
UNANSWERABLE_T3_Q = [
    "After the guitar solo, what does the drummer play?",
    "The second speaker sounds angry, right?",
    "Which of the three birds is singing the loudest?",
    "The explosion at the end was clearly fireworks, correct?",
    "The dog in this audio sounds aggressive, right?",
    "The singer switches from soprano to tenor at the end, correct?",
]
UNANSWERABLE_T3_A = [
    "The premise of this question does not match the audio content.",
    "This question contains assumptions that are not supported by the actual audio.",
    "The audio does not contain the elements described in this question.",
]

LIBRISPEECH_QA_TEMPLATES = [
    ("What is the main topic being discussed?", "topic"),
    ("Summarize what the speaker is saying.", "summary"),
    ("What is the speaker describing?", "description"),
    ("What key information is conveyed in this speech?", "key_info"),
    ("Provide a brief summary of the spoken content.", "summary"),
]

WAVCAPS_CAPTIONING_PROMPTS = [
    "Describe the sounds in this audio.",
    "What sounds can you hear in this recording?",
    "Provide a caption for this audio clip.",
    "What is happening in this audio?",
]


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

def download_json_cached(url: str, cache_name: str) -> Optional[list]:
    """Download a JSON file, caching locally. Returns parsed JSON or None."""
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists():
        print(f"    [cache hit] {cache_name}")
        with open(cache_path) as f:
            return json.load(f)

    print(f"    Downloading {cache_name}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=300) as r:
            raw = r.read()
        with open(cache_path, 'wb') as f:
            f.write(raw)
        data = json.loads(raw)
        entries = len(data) if isinstance(data, list) else len(data.get('data', data))
        print(f"    Downloaded {len(raw)/1024:.0f} KB → {entries} entries")
        return data
    except Exception as e:
        print(f"    ⚠️  Failed: {e}")
        return None


def load_jsonl(filepath: Path) -> List[dict]:
    """Load JSONL file."""
    data = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def make_example(audio_path, question, answer, task_type, encoder_type, source):
    """Create a training example dict."""
    return {
        "audio_path": str(audio_path),
        "question": question,
        "answer": answer,
        "task_type": task_type,
        "encoder_type": encoder_type,
        "source": source,
    }


def expected_audio_path(audio_prefix: str, sound_field: str) -> str:
    """Create expected audio path. No existence check — audio downloaded separately.

    Args:
        audio_prefix: Subdirectory under data/audio/ (e.g. "fsd50k", "esc50")
        sound_field: The 'sound' value from the dataset JSON entry
    Returns:
        Expected path like "data/audio/fsd50k/12345.wav"
    """
    basename = os.path.basename(sound_field)
    return str(AUDIO_DIR / audio_prefix / basename)


def write_jsonl(data: List[dict], filepath: Path):
    """Write list of dicts to JSONL file."""
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  → Wrote {len(data):,} examples to {filepath}")


# ═══════════════════════════════════════════════════════════════════════
# Phase A: Audio Grounding Data
# ═══════════════════════════════════════════════════════════════════════

def phase_a_subsample_stage1(stage1_train: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Subsample Stage 1 data for Phase A."""
    print("\n1. Subsampling Stage 1 data for Phase A...")

    by_source = defaultdict(list)
    for item in stage1_train:
        by_source[item["source"]].append(item)

    train = []

    # AudioCaps + Clotho captions → 60K
    caps = [e for e in (by_source.get("audiocaps_train", []) +
                        by_source.get("clotho_development", []))
            if e["task_type"] == "captioning"]
    random.shuffle(caps)
    cap_60k = caps[:60000]
    train.extend(cap_60k)
    print(f"  AudioCaps+Clotho captions: {len(cap_60k):,} (from {len(caps):,})")

    # FSD50K → all
    fsd = by_source.get("fsd50k_dev", [])
    train.extend(fsd)
    print(f"  FSD50K classification: {len(fsd):,}")

    # MusicCaps + Jamendo → all
    music = by_source.get("musiccaps", []) + by_source.get("jamendo", [])
    train.extend(music)
    print(f"  MusicCaps+Jamendo: {len(music):,}")

    # ClothoAQA → all
    aqa = by_source.get("clotho_aqa_development", [])
    train.extend(aqa)
    print(f"  ClothoAQA: {len(aqa):,}")

    # Val (excluding old unanswerable)
    val = [e for e in load_jsonl(STAGE1_VAL) if e["task_type"] != "unanswerable"]
    print(f"  Val (non-unanswerable): {len(val):,}")

    return train, val


def phase_a_librispeech_qa(stage1_train: List[dict]) -> List[dict]:
    """Reformat LibriSpeech as QA pairs."""
    print("\n2. Reformatting LibriSpeech as QA pairs...")

    libri = [e for e in stage1_train
             if "librispeech" in e.get("source", "")
             and "augmented" not in e.get("source", "")]
    random.shuffle(libri)
    libri = libri[:50000]

    examples = []
    for item in libri:
        words = item["answer"].split()
        if len(words) < 5:
            continue

        tmpl, qtype = random.choice(LIBRISPEECH_QA_TEMPLATES)
        if qtype == "topic":
            ans = f"The speaker is discussing: {' '.join(words[:20]).lower().rstrip('.')}."
        elif qtype == "summary":
            ans = ' '.join(words[:15]) + ("..." if len(words) > 15 else "")
        elif qtype == "description":
            ans = f"The speaker describes {' '.join(words[:15]).lower().rstrip('.')}."
        else:
            ans = ' '.join(words[:20])

        examples.append(make_example(
            item["audio_path"], tmpl, ans,
            "speech_qa", "speech", "librispeech_qa"
        ))

    print(f"  LibriSpeech QA: {len(examples):,}")
    return examples


def phase_a_audioskills() -> List[dict]:
    """Download AudioSkills MCQ splits. No audio filtering — all entries included."""
    print("\n3. Processing AudioSkills MCQ (all matching splits)...")

    all_examples = []
    for json_file, audio_prefix in AUDIOSKILLS_SPLITS.items():
        url = f"{AUDIOSKILLS_BASE}/{json_file}"
        cache_name = f"audioskills_{json_file}"
        data = download_json_cached(url, cache_name)
        if data is None:
            continue

        split_examples = []
        for entry in data:
            sound = entry.get("sound", "")
            if not sound:
                continue

            convs = entry.get("conversations", [])
            if len(convs) < 2:
                continue

            question = convs[0].get("value", "").replace("<sound>", "").replace("\n", " ").strip()
            answer = convs[1].get("value", "").strip()
            if not question or not answer:
                continue

            audio_path = expected_audio_path(audio_prefix, sound)
            split_examples.append(make_example(
                audio_path, question, answer,
                "mcq", "audio", f"audioskills_{json_file.replace('.json', '')}"
            ))

        # Subsample if capped
        cap = AUDIOSKILLS_SUBSAMPLE.get(json_file)
        if cap and len(split_examples) > cap:
            random.shuffle(split_examples)
            split_examples = split_examples[:cap]
            print(f"    {json_file}: {cap:,} entries (subsampled from {len(data):,})")
        else:
            print(f"    {json_file}: {len(split_examples):,} entries")

        all_examples.extend(split_examples)

    print(f"  AudioSkills total: {len(all_examples):,}")
    return all_examples


def phase_a_afthink_mcq() -> List[dict]:
    """Download AF-Think MCQ (afthink/) splits. No audio filtering."""
    print("\n4. Processing AF-Think MCQ (afthink/)...")

    all_examples = []
    for json_file, audio_prefix in AFTHINK_SPLITS.items():
        url = f"{AFTHINK_BASE}/{json_file}"
        cache_name = f"afthink_{json_file}"
        data = download_json_cached(url, cache_name)
        if data is None:
            continue

        count = 0
        for entry in data:
            sound = entry.get("sound", "")
            if not sound:
                continue

            convs = entry.get("conversations", [])
            if len(convs) < 2:
                continue

            question = convs[0].get("value", "").replace("<sound>", "").replace("\n", " ").strip()
            question = question.replace(
                "Please think and reason about the input audio before you respond.", ""
            ).strip()
            answer = convs[1].get("value", "").strip()
            if not question or not answer:
                continue

            audio_path = expected_audio_path(audio_prefix, sound)
            all_examples.append(make_example(
                audio_path, question, answer,
                "mcq_reasoning", "audio", f"afthink_{json_file.replace('.json', '')}"
            ))
            count += 1

        print(f"    {json_file}: {count:,} entries")

    print(f"  AF-Think MCQ total: {len(all_examples):,}")
    return all_examples


def phase_a_wavcaps() -> List[dict]:
    """Download WavCaps JSON captions for SoundBible + AudioSet_SL.

    Metadata only — audio downloaded separately on GPU machine.
    SoundBible: 1,232 diverse sound effects with ChatGPT-cleaned captions.
    AudioSet_SL: ~49K strongly-labelled AudioSet clips with captions.
    """
    print("\n5. Processing WavCaps (SoundBible + AudioSet_SL)...")

    # Subsample AudioSet_SL to 50K
    WAVCAPS_SUBSAMPLE = {"AudioSet_SL": 50000}

    all_examples = []
    for subset_name, cfg in WAVCAPS_SUBSETS.items():
        json_url = f"{WAVCAPS_JSON_BASE}/{cfg['json_path']}"
        cache_name = f"wavcaps_{subset_name}.json"
        data = download_json_cached(json_url, cache_name)
        if data is None:
            continue

        entries = data.get("data", data) if isinstance(data, dict) else data
        audio_dir = cfg["audio_dir"]
        subset_examples = []

        for entry in entries:
            caption = entry.get("caption", "").strip()
            file_id = str(entry.get("id", ""))
            duration = entry.get("duration", 0)

            if not caption or not file_id:
                continue
            if duration and duration < 0.5:
                continue

            audio_path = str(AUDIO_DIR / audio_dir / f"{file_id}.wav")

            question = random.choice(WAVCAPS_CAPTIONING_PROMPTS)
            subset_examples.append(make_example(
                audio_path, question, caption,
                "captioning", "audio", f"wavcaps_{subset_name.lower()}"
            ))

        # Subsample if capped
        cap = WAVCAPS_SUBSAMPLE.get(subset_name)
        if cap and len(subset_examples) > cap:
            random.shuffle(subset_examples)
            subset_examples = subset_examples[:cap]
            print(f"    {subset_name}: {cap:,} entries (subsampled from {len(entries):,}) ({cfg['description']})")
        else:
            print(f"    {subset_name}: {len(subset_examples):,} entries ({cfg['description']})")

        all_examples.extend(subset_examples)

    print(f"  WavCaps total: {len(all_examples):,}")
    return all_examples


def phase_a_nsynth() -> List[dict]:
    """Extract NSynth metadata by streaming tar.gz from Google Magenta.

    Streams through the tar.gz and extracts ONLY examples.json (~2-5 MB).
    No audio files are saved. Audio download handled on GPU machine.

    NSynth: 300K+ 4-second musical note samples from 1000+ instruments.
    10 families: bass, brass, flute, guitar, keyboard, mallet, organ, reed, string, vocal.
    3 sources: acoustic, electronic, synthetic.
    """
    print("\n6. Processing NSynth (instrument classification)...")

    all_examples = []

    for split_name, url in NSYNTH_TAR_URLS.items():
        cache_path = CACHE_DIR / f"nsynth_{split_name}_meta.json"

        if cache_path.exists():
            print(f"    [cache hit] nsynth_{split_name}_meta.json")
            with open(cache_path) as f:
                metadata = json.load(f)
        else:
            print(f"    Streaming nsynth-{split_name} tar.gz to extract metadata...")
            print(f"    (extracts only examples.json — no audio saved)")
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                resp = urllib.request.urlopen(req, timeout=600)
                metadata = None
                with tarfile.open(fileobj=resp, mode='r|gz') as tar:
                    for member in tar:
                        if member.name.endswith('examples.json'):
                            f = tar.extractfile(member)
                            if f is not None:
                                metadata = json.load(f)
                            break
                resp.close()

                if metadata is None:
                    print(f"    ⚠️  examples.json not found in tar")
                    continue

                # Cache the extracted metadata
                with open(cache_path, 'w') as f:
                    json.dump(metadata, f)
                print(f"    Extracted {len(metadata):,} entries → cached")

            except Exception as e:
                print(f"    ⚠️  Failed: {e}")
                continue

        # Create classification examples
        for note_str, info in metadata.items():
            family = info.get("instrument_family_str", "")
            source_type = info.get("instrument_source_str", "")

            if not family:
                continue

            audio_path = str(AUDIO_DIR / NSYNTH_AUDIO_DIR / f"{note_str}.wav")

            # Vary question types
            if random.random() < 0.6:
                question = random.choice(NSYNTH_Q_TEMPLATES)
                answer = f"The instrument playing is a {source_type} {family}."
            else:
                question = random.choice(NSYNTH_SOURCE_Q_TEMPLATES)
                answer = f"This instrument is {source_type}. It belongs to the {family} family."

            all_examples.append(make_example(
                audio_path, question, answer,
                "classification", "audio", f"nsynth_{split_name}"
            ))

        print(f"    {split_name}: {len(metadata):,} entries")

    # Subsample
    if len(all_examples) > NSYNTH_SUBSAMPLE:
        random.shuffle(all_examples)
        all_examples = all_examples[:NSYNTH_SUBSAMPLE]

    print(f"  NSynth total: {len(all_examples):,} (subsampled to {NSYNTH_SUBSAMPLE:,})")
    return all_examples


def phase_a_unanswerable(train_examples: List[dict]) -> List[dict]:
    """Generate 12% unanswerable injection with NEW templates."""
    print("\n7. Generating Phase A unanswerable (12%)...")

    non_unans = len(train_examples)
    target = int(non_unans * 0.12 / 0.88)
    per_type = target // 3

    audio_paths = [e["audio_path"] for e in train_examples if e["encoder_type"] == "audio"]
    speech_paths = [e["audio_path"] for e in train_examples if e["encoder_type"] == "speech"]
    all_paths = audio_paths + speech_paths

    unanswerable = []

    # Type 1: Ask for speech in non-speech audio
    random.shuffle(audio_paths)
    for i in range(per_type):
        unanswerable.append(make_example(
            audio_paths[i % len(audio_paths)],
            random.choice(UNANSWERABLE_T1_Q), random.choice(UNANSWERABLE_T1_A),
            "unanswerable", "both", "unanswerable_s2_type1"
        ))

    # Type 2: Ask for specifics audio can't provide
    random.shuffle(all_paths)
    for i in range(per_type):
        unanswerable.append(make_example(
            all_paths[i % len(all_paths)],
            random.choice(UNANSWERABLE_T2_Q), random.choice(UNANSWERABLE_T2_A),
            "unanswerable", "both", "unanswerable_s2_type2"
        ))

    # Type 3: Logical traps / false premises
    random.shuffle(all_paths)
    for i in range(per_type):
        unanswerable.append(make_example(
            all_paths[i % len(all_paths)],
            random.choice(UNANSWERABLE_T3_Q), random.choice(UNANSWERABLE_T3_A),
            "unanswerable", "both", "unanswerable_s2_type3"
        ))

    rate = len(unanswerable) / (non_unans + len(unanswerable)) * 100
    print(f"  Unanswerable: {len(unanswerable):,} ({rate:.1f}% of total)")
    return unanswerable


# ═══════════════════════════════════════════════════════════════════════
# Phase B: CoT + Format Discrimination
# ═══════════════════════════════════════════════════════════════════════

def phase_b_extractive_cot() -> List[dict]:
    """Extractive CoT from AF-Think afcot/ splits. No audio filtering.

    CAPTION → <think> (avg 24 words, grounded audio description)
    CONCLUSION → answer (avg 33 words, factual response)
    Hard cap: skip if CAPTION + CONCLUSION > 80 words.
    """
    print("\n8. Processing AF-Think CoT (extractive compression)...")

    all_examples = []
    stats = Counter()

    for json_file, audio_prefix in AFCOT_SPLITS.items():
        url = f"{AFCOT_BASE}/{json_file}"
        cache_name = f"afcot_{json_file}"
        data = download_json_cached(url, cache_name)
        if data is None:
            continue

        count = 0
        for entry in data:
            sound = entry.get("sound", "")
            if not sound:
                stats["no_sound"] += 1
                continue

            convs = entry.get("conversations", [])
            if len(convs) < 2:
                stats["no_convs"] += 1
                continue

            gpt_msg = convs[1].get("value", "")

            caption_m = re.search(r'<CAPTION>(.*?)</CAPTION>', gpt_msg, re.DOTALL)
            conclusion_m = re.search(r'<CONCLUSION>(.*?)</CONCLUSION>', gpt_msg, re.DOTALL)

            if not caption_m or not conclusion_m:
                stats["no_tags"] += 1
                continue

            caption = caption_m.group(1).strip()
            conclusion = conclusion_m.group(1).strip()

            # Hard cap: 80 words combined
            if len(caption.split()) + len(conclusion.split()) > 80:
                stats["too_long"] += 1
                continue

            think_answer = f"<think>{caption}</think>\n{conclusion}"

            question = convs[0].get("value", "").replace("<sound>", "").replace("\n", " ").strip()
            question = re.sub(r'\s*Output the answer with.*?tags\.?\s*$', '', question).strip()
            if not question:
                stats["empty_q"] += 1
                continue

            audio_path = expected_audio_path(audio_prefix, sound)
            all_examples.append(make_example(
                audio_path, question, think_answer,
                "cot_reasoning", "audio", f"afcot_{json_file.replace('.json', '')}"
            ))
            count += 1
            stats["matched"] += 1

        print(f"    {json_file}: {count:,} extracted")

    print(f"  AF-Think CoT total (before cap): {len(all_examples):,}")
    skip = {k: v for k, v in stats.items() if k != "matched"}
    if skip:
        print(f"  Skipped: {dict(skip)}")

    # Stratified subsample: keep all small sources, only cap WavCaps + FSD50K
    AFCOT_TOTAL_CAP = 25000
    if len(all_examples) > AFCOT_TOTAL_CAP:
        big_sources = {"afcot_WavCaps", "afcot_FSD50K"}
        small = [e for e in all_examples if e["source"] not in big_sources]
        wavcaps = [e for e in all_examples if e["source"] == "afcot_WavCaps"]
        fsd50k = [e for e in all_examples if e["source"] == "afcot_FSD50K"]

        remaining = AFCOT_TOTAL_CAP - len(small)
        # FSD50K gets 57%, WavCaps gets 43% (FSD50K > WavCaps as requested)
        fsd50k_cap = int(remaining * 0.57)
        wavcaps_cap = remaining - fsd50k_cap

        random.shuffle(fsd50k)
        random.shuffle(wavcaps)
        all_examples = small + fsd50k[:fsd50k_cap] + wavcaps[:wavcaps_cap]

        print(f"  Subsampled to {len(all_examples):,} (stratified):")
        print(f"    Small sources (kept fully): {len(small):,}")
        print(f"    afcot_FSD50K: {min(len(fsd50k), fsd50k_cap):,} (from {len(fsd50k):,})")
        print(f"    afcot_WavCaps: {min(len(wavcaps), wavcaps_cap):,} (from {len(wavcaps):,})")

    # Show final source distribution
    src_dist = Counter(e["source"] for e in all_examples)
    for src, cnt in sorted(src_dist.items(), key=lambda x: -x[1]):
        print(f"    {src:<35s} {cnt:>6,}")

    return all_examples


def phase_b_format_discrimination(phase_a_train: List[dict]) -> List[dict]:
    """Subsample simple Phase A examples WITHOUT <think> tags.

    Teaches model WHEN to reason vs when to just answer directly.
    """
    print("\n9. Adding format discrimination (no <think> examples)...")

    simple = [e for e in phase_a_train
              if e["task_type"] in ("captioning", "classification", "binary_qa")]
    n = min(len(simple), 5000)
    random.shuffle(simple)
    disc = simple[:n]
    print(f"  Format discrimination: {len(disc):,} simple examples (no <think>)")
    return disc


# ═══════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  Stage 2 Data Preparation (Metadata-Only)")
    print("  Audio download handled separately on GPU machine")
    print("=" * 70)

    if not STAGE1_TRAIN.exists():
        print(f"\n❌ {STAGE1_TRAIN} not found. Run prepare_stage1_data.py first.")
        return

    print(f"\nLoading Stage 1 training data...")
    stage1_train = load_jsonl(STAGE1_TRAIN)
    print(f"  Loaded {len(stage1_train):,} Stage 1 examples")

    # ═══════════════════════════════════════════════════════════════
    # PHASE A
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PHASE A — Audio Grounding")
    print("=" * 70)

    phase_a_train, phase_a_val = phase_a_subsample_stage1(stage1_train)
    phase_a_train.extend(phase_a_librispeech_qa(stage1_train))
    phase_a_train.extend(phase_a_audioskills())
    phase_a_train.extend(phase_a_afthink_mcq())
    phase_a_train.extend(phase_a_wavcaps())
    phase_a_train.extend(phase_a_nsynth())
    phase_a_train.extend(phase_a_unanswerable(phase_a_train))

    # Proportional val unanswerable
    val_n = int(len(phase_a_val) * 0.12)
    val_paths = [e["audio_path"] for e in phase_a_val]
    for i in range(val_n):
        templates = [UNANSWERABLE_T1_Q, UNANSWERABLE_T2_Q, UNANSWERABLE_T3_Q]
        answers = [UNANSWERABLE_T1_A, UNANSWERABLE_T2_A, UNANSWERABLE_T3_A]
        t = i % 3
        phase_a_val.append(make_example(
            val_paths[i % len(val_paths)],
            random.choice(templates[t]), random.choice(answers[t]),
            "unanswerable", "both", f"unanswerable_s2_type{t+1}"
        ))

    random.shuffle(phase_a_train)
    random.shuffle(phase_a_val)

    # ═══════════════════════════════════════════════════════════════
    # PHASE B
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PHASE B — CoT + Format Discrimination")
    print("=" * 70)

    phase_b_train = phase_b_extractive_cot()
    phase_b_train.extend(phase_b_format_discrimination(phase_a_train))
    random.shuffle(phase_b_train)

    # ═══════════════════════════════════════════════════════════════
    # Write Output
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Writing Output Files")
    print("=" * 70)

    write_jsonl(phase_a_train, STAGE2_DIR / "phase_a_train.jsonl")
    write_jsonl(phase_a_val, STAGE2_DIR / "phase_a_val.jsonl")
    write_jsonl(phase_b_train, STAGE2_DIR / "phase_b_train.jsonl")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    for label, dataset in [("Phase A Train", phase_a_train),
                           ("Phase A Val", phase_a_val),
                           ("Phase B Train", phase_b_train)]:
        print(f"\n  {label}: {len(dataset):,}")
        sources = Counter(e["source"] for e in dataset)
        for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
            pct = cnt / len(dataset) * 100
            print(f"    {src:<45s} {cnt:>8,}  ({pct:5.1f}%)")

    # Task distribution
    print(f"\n  Phase A by task:")
    for task, cnt in sorted(Counter(e["task_type"] for e in phase_a_train).items(),
                            key=lambda x: -x[1]):
        pct = cnt / len(phase_a_train) * 100
        print(f"    {task:<25s} {cnt:>8,}  ({pct:5.1f}%)")

    # Encoder distribution
    print(f"\n  Phase A by encoder:")
    for enc, cnt in sorted(Counter(e["encoder_type"] for e in phase_a_train).items(),
                           key=lambda x: -x[1]):
        pct = cnt / len(phase_a_train) * 100
        print(f"    {enc:<15s} {cnt:>8,}  ({pct:5.1f}%)")

    # Phase B stats
    think_ct = sum(1 for e in phase_b_train if "<think>" in e.get("answer", ""))
    print(f"\n  Phase B format:")
    print(f"    With <think>:    {think_ct:,}")
    print(f"    Without <think>: {len(phase_b_train) - think_ct:,} (format discrimination)")

    # Unanswerable rate
    unans = sum(1 for e in phase_a_train if e["task_type"] == "unanswerable")
    print(f"\n  Unanswerable rate: {unans / len(phase_a_train) * 100:.1f}% (target: 12%)")

    # ── Audio download manifest ──
    print(f"\n" + "=" * 70)
    print("  AUDIO DOWNLOAD MANIFEST (for GPU machine)")
    print("=" * 70)

    # Collect unique audio directories referenced
    audio_dirs = Counter()
    for dataset in [phase_a_train, phase_a_val, phase_b_train]:
        for e in dataset:
            p = e["audio_path"]
            if p.startswith("data/audio/"):
                parts = p.replace("data/audio/", "").split("/")
                if parts:
                    audio_dirs[parts[0]] += 1

    print(f"\n  Audio directories referenced:")
    for d, cnt in sorted(audio_dirs.items(), key=lambda x: -x[1]):
        print(f"    data/audio/{d:<30s} {cnt:>8,} files")

    print(f"\n  Download sources:")
    print(f"    fsd50k          → HuggingFace (huggingface.co/datasets/google/FSD50K)")
    print(f"    clotho           → Zenodo (zenodo.org/record/3490684)")
    print(f"    audiocaps/train  → YouTube via yt-dlp")
    print(f"    musiccaps        → YouTube via yt-dlp")
    print(f"    jamendo          → Jamendo API / HuggingFace")
    print(f"    esc50            → GitHub (github.com/karolpiczak/ESC-50)")
    print(f"    urbansound8k     → urbansounddataset.weebly.com (registration required)")
    print(f"    wavcaps_soundbible    → HuggingFace (cvssp/WavCaps, Zip_files/SoundBible/)")
    print(f"    wavcaps_audioset_sl   → HuggingFace (cvssp/WavCaps, Zip_files/AudioSet_SL/)")
    print(f"    nsynth           → Google Magenta (nsynth-valid + nsynth-test tar.gz)")
    print(f"    librispeech      → HuggingFace (openslr/librispeech_asr)")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
