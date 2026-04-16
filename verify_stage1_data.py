#!/usr/bin/env python3
"""
Stage 1 Data Verification — checks distribution, quality, and filtering
against the training plan targets.

Run: python verify_stage1_data.py
"""
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import random
import statistics

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

TRAIN_FILE = Path("data/stage1/stage1_train.jsonl")
VAL_FILE   = Path("data/stage1/stage1_val.jsonl")

# Training plan v6 targets (updated with actual counts)
PLAN_TARGETS = {
    # Sound data (AudioMapper)
    "audiocaps_train":          {"expected": 80000, "tolerance": 0.05, "encoder": "audio", "task": "captioning"},
    "clotho_development":       {"expected": 19000, "tolerance": 0.05, "encoder": "audio", "task": "captioning"},
    "fsd50k_dev":               {"expected": 41000, "tolerance": 0.05, "encoder": "audio", "task": "classification"},
    "clotho_aqa_development":   {"expected": 15000, "tolerance": 0.05, "encoder": "audio", "task": "binary_qa"},
    # Music data (AudioMapper)
    "musiccaps":                {"expected": 5000,  "tolerance": 0.10, "encoder": "audio", "task": "music_captioning"},
    "jamendo":                  {"expected": 5000,  "tolerance": 0.10, "encoder": "audio", "task": "music_captioning"},
    # Speech data (SpeechMapper)
    "librispeech_train-clean-100":       {"expected": 28500,  "tolerance": 0.05, "encoder": "speech", "task": "transcription"},
    "librispeech_train-clean-360":       {"expected": 104000, "tolerance": 0.05, "encoder": "speech", "task": "transcription"},
    "librispeech_train-clean-100_augmented": {"expected": 5700, "tolerance": 0.15, "encoder": "speech", "task": "speech_description"},
    # Unanswerable (hallucination control)
    "unanswerable_type1":       {"expected": None, "tolerance": 0.20, "encoder": "mixed", "task": "unanswerable"},
    "unanswerable_type2":       {"expected": None, "tolerance": 0.20, "encoder": "mixed", "task": "unanswerable"},
    "unanswerable_type3":       {"expected": None, "tolerance": 0.20, "encoder": "mixed", "task": "unanswerable"},
}


def load_jsonl(filepath):
    """Load JSONL file, return list of dicts."""
    data = []
    errors = 0
    with open(filepath) as f:
        for i, line in enumerate(f, 1):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                errors += 1
                if errors <= 5:
                    print(f"  ⚠️  JSON parse error at line {i}")
    if errors:
        print(f"  ⚠️  Total JSON errors: {errors}")
    return data


def print_header(title):
    width = 70
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def print_section(title):
    print(f"\n  ── {title} {'─' * max(0, 55 - len(title))}")


def check_distribution(data, label="train"):
    """Check distribution by source, task, and encoder type."""
    print_header(f"Distribution Analysis ({label}: {len(data):,} examples)")

    # ── By Source ──
    print_section("By Source")
    by_source = Counter(d.get("source", "MISSING") for d in data)
    max_count = max(by_source.values()) if by_source else 1

    for source, count in sorted(by_source.items(), key=lambda x: -x[1]):
        pct = count / len(data) * 100
        bar = "█" * int(count / max_count * 30)
        print(f"    {source:<45s} {count:>8,}  ({pct:5.1f}%)  {bar}")

    # ── By Task Type ──
    print_section("By Task Type")
    by_task = Counter(d.get("task_type", "MISSING") for d in data)
    for task, count in sorted(by_task.items(), key=lambda x: -x[1]):
        pct = count / len(data) * 100
        print(f"    {task:<30s} {count:>8,}  ({pct:5.1f}%)")

    # ── By Encoder Type ──
    print_section("By Encoder Type")
    by_encoder = Counter(d.get("encoder_type", "MISSING") for d in data)
    for enc, count in sorted(by_encoder.items(), key=lambda x: -x[1]):
        pct = count / len(data) * 100
        bar = "█" * int(count / len(data) * 40)
        print(f"    {enc:<15s} {count:>8,}  ({pct:5.1f}%)  {bar}")

    # ── AudioMapper vs SpeechMapper split ──
    audio_count = sum(c for e, c in by_encoder.items() if e == "audio")
    speech_count = sum(c for e, c in by_encoder.items() if e == "speech")
    total = audio_count + speech_count
    if total > 0:
        print(f"\n    AudioMapper data:  {audio_count:>8,}  ({audio_count/total*100:.1f}%)")
        print(f"    SpeechMapper data: {speech_count:>8,}  ({speech_count/total*100:.1f}%)")
        ratio = audio_count / speech_count if speech_count else float('inf')
        print(f"    Audio:Speech ratio: {ratio:.2f}:1")

    return by_source, by_task, by_encoder


def check_against_plan(by_source, total_count):
    """Compare actual counts against training plan targets."""
    print_header("Training Plan Compliance Check")

    issues = []
    for source, target_info in PLAN_TARGETS.items():
        actual = by_source.get(source, 0)
        expected = target_info["expected"]

        if expected is None:
            # For unanswerable, just check existence
            status = "✅" if actual > 0 else "❌ MISSING"
            print(f"    {status}  {source:<45s} {actual:>8,}  (no target)")
            if actual == 0:
                issues.append(f"{source}: expected some entries, got 0")
            continue

        diff_pct = abs(actual - expected) / expected * 100
        tolerance_pct = target_info["tolerance"] * 100

        if diff_pct <= tolerance_pct:
            status = "✅"
        elif actual == 0:
            status = "❌"
            issues.append(f"{source}: expected ~{expected:,}, got 0")
        else:
            status = "⚠️ "
            issues.append(f"{source}: expected ~{expected:,}, got {actual:,} (diff: {diff_pct:.1f}%)")

        print(f"    {status}  {source:<45s} {actual:>8,} / ~{expected:,}  (diff: {diff_pct:.1f}%)")

    # Check unanswerable rate
    unanswerable_total = sum(by_source.get(f"unanswerable_type{i}", 0) for i in range(1, 4))
    non_unanswerable = total_count - unanswerable_total
    actual_rate = unanswerable_total / non_unanswerable * 100 if non_unanswerable > 0 else 0
    target_rate = 6.0

    print(f"\n    Unanswerable injection rate: {actual_rate:.1f}% (target: {target_rate}%)")
    if abs(actual_rate - target_rate) > 1.5:
        issues.append(f"Unanswerable rate: {actual_rate:.1f}% vs target {target_rate}%")
        print(f"    ⚠️  Rate deviates from target by {abs(actual_rate - target_rate):.1f}pp")
    else:
        print(f"    ✅  Within acceptable range")

    # Check unanswerable type balance
    type_counts = [by_source.get(f"unanswerable_type{i}", 0) for i in range(1, 4)]
    if all(t > 0 for t in type_counts):
        max_diff = max(type_counts) - min(type_counts)
        avg = sum(type_counts) / 3
        skew = max_diff / avg * 100 if avg > 0 else 0
        if skew < 5:
            print(f"    ✅  Unanswerable types balanced (skew: {skew:.1f}%)")
        else:
            print(f"    ⚠️  Unanswerable type imbalance: {type_counts} (skew: {skew:.1f}%)")

    return issues


def check_data_quality(data, label="train"):
    """Check for common data quality issues."""
    print_header(f"Data Quality Checks ({label})")

    issues = []

    # ── Required fields ──
    print_section("Field Completeness")
    required_fields = ["audio_path", "question", "answer", "task_type", "encoder_type", "source"]
    for field in required_fields:
        missing = sum(1 for d in data if not d.get(field))
        if missing > 0:
            print(f"    ❌  '{field}' missing/empty in {missing:,} entries")
            issues.append(f"'{field}' missing in {missing:,} entries")
        else:
            print(f"    ✅  '{field}' present in all entries")

    # ── Answer length distribution ──
    print_section("Answer Length Distribution (words)")
    answer_lengths = [len(str(d.get("answer", "")).split()) for d in data]
    if answer_lengths:
        print(f"    min:    {min(answer_lengths)}")
        print(f"    max:    {max(answer_lengths)}")
        print(f"    mean:   {statistics.mean(answer_lengths):.1f}")
        print(f"    median: {statistics.median(answer_lengths):.1f}")

        # Flag very short answers (excluding binary QA)
        non_binary = [(d, len(str(d.get("answer", "")).split()))
                      for d in data if d.get("task_type") not in ("binary_qa", "unanswerable")]
        short_non_binary = sum(1 for _, wc in non_binary if wc < 3)
        if short_non_binary > 0:
            print(f"    ⚠️  {short_non_binary:,} non-binary/non-unanswerable answers < 3 words")

    # ── Empty/very short answers by task ──
    print_section("Answer Length by Task")
    by_task = defaultdict(list)
    for d in data:
        wc = len(str(d.get("answer", "")).split())
        by_task[d.get("task_type", "unknown")].append(wc)

    for task, lengths in sorted(by_task.items()):
        avg = statistics.mean(lengths) if lengths else 0
        med = statistics.median(lengths) if lengths else 0
        print(f"    {task:<25s}  n={len(lengths):>8,}  avg_words={avg:5.1f}  median={med:4.1f}  min={min(lengths)}  max={max(lengths)}")

    # ── Duplicate detection ──
    print_section("Duplicate Detection")

    # Exact audio_path + question duplicates
    seen_pairs = Counter()
    for d in data:
        key = (d.get("audio_path", ""), d.get("question", ""))
        seen_pairs[key] += 1
    exact_dups = sum(c - 1 for c in seen_pairs.values() if c > 1)
    n_dup_pairs = sum(1 for c in seen_pairs.values() if c > 1)
    if exact_dups > 0:
        print(f"    ⚠️  {exact_dups:,} exact duplicates (audio_path + question) in {n_dup_pairs:,} pairs")
        # Show top duplicated pairs
        for (path, q), count in seen_pairs.most_common(3):
            if count > 1:
                base = os.path.basename(path)
                print(f"       {base} × {count}: \"{q[:50]}...\"")
    else:
        print(f"    ✅  No exact (audio_path + question) duplicates")

    # Exact answer duplicates (checking for over-representation)
    answer_counts = Counter(d.get("answer", "") for d in data)
    top_answers = answer_counts.most_common(10)
    print(f"\n    Top 10 most common answers:")
    for ans, count in top_answers:
        display = ans[:60] + "..." if len(ans) > 60 else ans
        pct = count / len(data) * 100
        print(f"      {count:>6,} ({pct:4.1f}%)  \"{display}\"")

    # ── Audio path validation ──
    print_section("Audio Path Validation")
    path_prefixes = Counter()
    invalid_extensions = Counter()

    for d in data:
        path = d.get("audio_path", "")
        # Check prefix
        parts = Path(path).parts
        if len(parts) >= 3:
            prefix = "/".join(parts[:3])
            path_prefixes[prefix] += 1
        # Check extension
        ext = Path(path).suffix.lower()
        if ext not in (".wav", ".flac", ".mp3", ".ogg"):
            invalid_extensions[ext] += 1

    print(f"    Audio path prefixes:")
    for prefix, count in sorted(path_prefixes.items(), key=lambda x: -x[1]):
        print(f"      {prefix:<40s} {count:>8,}")

    if invalid_extensions:
        print(f"    ⚠️  Unexpected extensions: {dict(invalid_extensions)}")
    else:
        print(f"    ✅  All audio paths have valid extensions")

    # ── Encoder routing validation ──
    print_section("Encoder Routing Validation")
    routing_issues = 0
    for d in data:
        source = d.get("source", "")
        encoder = d.get("encoder_type", "")

        # Check that speech sources use speech encoder
        if "librispeech" in source and encoder != "speech":
            routing_issues += 1
        # Check that sound/music sources use audio encoder
        if any(s in source for s in ["audiocaps", "clotho", "fsd50k", "musiccaps", "jamendo"]):
            if encoder != "audio":
                routing_issues += 1

    if routing_issues:
        print(f"    ❌  {routing_issues:,} entries with wrong encoder routing")
    else:
        print(f"    ✅  All encoder routing correct (speech→speech, audio/music→audio)")

    return issues


def check_question_diversity(data):
    """Check that question templates are diverse per task."""
    print_header("Question Template Diversity")

    by_task = defaultdict(list)
    for d in data:
        by_task[d.get("task_type", "unknown")].append(d.get("question", ""))

    for task, questions in sorted(by_task.items()):
        unique_q = len(set(questions))
        total_q = len(questions)
        print(f"    {task:<25s}  {unique_q:>4} unique templates / {total_q:>8,} total")

        # Show sample questions
        sample_qs = random.sample(list(set(questions)), min(3, unique_q))
        for q in sample_qs:
            print(f"      └─ \"{q[:70]}{'...' if len(q) > 70 else ''}\"")


def show_random_samples(data, n=2):
    """Show random sample entries for manual inspection."""
    print_header("Random Samples (for manual review)")

    # Group by source, show samples from each
    by_source = defaultdict(list)
    for d in data:
        by_source[d.get("source", "unknown")].append(d)

    for source in sorted(by_source.keys()):
        examples = by_source[source]
        samples = random.sample(examples, min(n, len(examples)))
        print(f"\n  ── {source} ({len(examples):,} total) ──")
        for s in samples:
            q = s.get("question", "")[:80]
            a = s.get("answer", "")[:120]
            p = os.path.basename(s.get("audio_path", ""))
            print(f"    📁 {p}")
            print(f"    ❓ {q}")
            print(f"    💬 {a}")
            print()


def check_train_val_leak(train_data, val_data):
    """Check for data leakage between train and val."""
    print_header("Train/Val Leakage Check")

    train_paths = set(d.get("audio_path", "") for d in train_data)
    val_paths = set(d.get("audio_path", "") for d in val_data)

    overlap = train_paths & val_paths
    if overlap:
        print(f"    ⚠️  {len(overlap):,} audio paths appear in BOTH train and val")
        for p in list(overlap)[:5]:
            print(f"       {p}")
    else:
        print(f"    ✅  No audio path overlap between train and val")

    # Check answer leakage (same audio+question in both)
    train_keys = set((d.get("audio_path", ""), d.get("question", "")) for d in train_data)
    val_keys = set((d.get("audio_path", ""), d.get("question", "")) for d in val_data)
    key_overlap = train_keys & val_keys
    if key_overlap:
        print(f"    ❌  {len(key_overlap):,} exact (audio_path + question) pairs in BOTH train and val")
    else:
        print(f"    ✅  No exact (audio_path + question) leakage")

    # Val distribution
    print(f"\n    Val size: {len(val_data):,} ({len(val_data)/len(train_data)*100:.1f}% of train)")
    val_sources = Counter(d.get("source", "") for d in val_data)
    print(f"    Val sources:")
    for src, cnt in sorted(val_sources.items(), key=lambda x: -x[1]):
        print(f"      {src:<40s} {cnt:>6,}")


def print_final_summary(train_data, val_data, plan_issues, quality_issues):
    """Print final pass/fail summary."""
    print_header("FINAL SUMMARY")

    total = len(train_data) + len(val_data)
    print(f"    Total examples: {total:,} (train: {len(train_data):,}, val: {len(val_data):,})")

    all_issues = plan_issues + quality_issues
    if not all_issues:
        print(f"\n    ✅  ALL CHECKS PASSED — data looks good for Stage 1 training!")
    else:
        print(f"\n    ⚠️  {len(all_issues)} issue(s) found:")
        for issue in all_issues:
            print(f"       • {issue}")

    print(f"\n{'═' * 70}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  Stage 1 Data Verification Report")
    print("=" * 70)

    if not TRAIN_FILE.exists():
        print(f"\n  ❌  {TRAIN_FILE} not found. Run prepare_stage1_data.py first.")
        return

    # Load data
    print(f"\n  Loading {TRAIN_FILE}...")
    train_data = load_jsonl(TRAIN_FILE)
    print(f"  Loaded {len(train_data):,} training examples")

    val_data = []
    if VAL_FILE.exists():
        print(f"  Loading {VAL_FILE}...")
        val_data = load_jsonl(VAL_FILE)
        print(f"  Loaded {len(val_data):,} validation examples")

    random.seed(42)  # Reproducible samples

    # Run checks
    by_source, by_task, by_encoder = check_distribution(train_data, "train")
    plan_issues = check_against_plan(by_source, len(train_data))
    quality_issues = check_data_quality(train_data, "train")
    check_question_diversity(train_data)

    if val_data:
        check_train_val_leak(train_data, val_data)
        check_distribution(val_data, "val")

    show_random_samples(train_data, n=1)

    # Final summary
    print_final_summary(train_data, val_data, plan_issues, quality_issues)


if __name__ == "__main__":
    main()
