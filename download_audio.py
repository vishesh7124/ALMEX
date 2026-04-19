"""
download_audio.py — Comprehensive Audio Downloader for ALM Training

Downloads audio files for all 11 data sources used in Stage 1 and Stage 2.
Each source has a dedicated downloader function. All audio is saved as
16kHz mono WAV in data/audio/<source>/<split>/<filename>.wav

Usage:
    python download_audio.py --source <name>        # download one source
    python download_audio.py --source all            # download everything
    python download_audio.py --source <name> --limit 100  # test with 100 files
    python download_audio.py --status                # show download progress

Sources:
    librispeech         HuggingFace (openslr/librispeech_asr)      ~12 GB
    audiocaps           YouTube via yt-dlp                          ~15 GB
    fsd50k              HuggingFace (google/FSD50K) or Zenodo       ~27 GB
    wavcaps_audioset_sl HuggingFace (cvssp/WavCaps) zip             ~33 GB
    wavcaps_soundbible  HuggingFace (cvssp/WavCaps) zip             ~553 MB
    clotho              Zenodo (zenodo.org/record/3490684)           ~8 GB
    musiccaps           YouTube via yt-dlp                           ~3 GB
    nsynth              Google Magenta (tar.gz)                      ~1.3 GB
    urbansound8k        Manual download (registration required)      ~6 GB
    jamendo             HuggingFace (amaai-lab/JamendoMaxCaps)       ~2 GB
    esc50               GitHub (karolpiczak/ESC-50)                  ~600 MB

Features:
    - Crash-safe resume via progress files (data/download_progress_<source>.json)
    - Ctrl+C pauses cleanly, re-run to resume
    - Parallel downloads where possible
    - Only downloads files referenced in training JSONLs

Requirements:
    pip install yt-dlp tqdm datasets huggingface_hub
    sudo apt install ffmpeg
"""

import json
import os
import csv
import glob
import signal
import subprocess
import sys
import time
import shutil
import zipfile
import tarfile
from pathlib import Path
from collections import Counter

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

DATA_DIR = Path("data")
AUDIO_DIR = DATA_DIR / "audio"

SAMPLE_RATE = 16000

# ─────────────────────────────────────────────────────────────────────
# Pause handler
# ─────────────────────────────────────────────────────────────────────

_pause_requested = False

def _signal_handler(sig, frame):
    global _pause_requested
    if _pause_requested:
        print("\n\n  Force quit. Progress saved.")
        sys.exit(1)
    _pause_requested = True
    print("\n\n  ⏸  Pause requested — finishing current file, then saving...\n")


# ─────────────────────────────────────────────────────────────────────
# Progress tracker
# ─────────────────────────────────────────────────────────────────────

def progress_path(source: str) -> Path:
    return DATA_DIR / f"download_progress_{source}.json"

def load_progress(source: str) -> dict:
    p = progress_path(source)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"completed": [], "failed": [], "success": 0, "fail": 0, "skip": 0}

def save_progress(source: str, prog: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(progress_path(source), 'w') as f:
        json.dump(prog, f)


# ─────────────────────────────────────────────────────────────────────
# Collect needed audio paths from JSONLs
# ─────────────────────────────────────────────────────────────────────

def get_needed_paths(source_prefix: str) -> set:
    """Scan all training JSONLs and collect unique audio paths for a source."""
    needed = set()
    jsonl_files = [
        "data/stage1/stage1_train.jsonl",
        "data/stage1/stage1_val.jsonl",
        "data/stage2/phase_a_train.jsonl",
        "data/stage2/phase_a_val.jsonl",
        "data/stage2/phase_b_train.jsonl",
    ]
    for jf in jsonl_files:
        if not os.path.exists(jf):
            continue
        with open(jf) as f:
            for line in f:
                d = json.loads(line)
                p = d["audio_path"]
                if p.startswith(f"data/audio/{source_prefix}/"):
                    needed.add(p)
    return needed


# ─────────────────────────────────────────────────────────────────────
# Audio conversion utility
# ─────────────────────────────────────────────────────────────────────

def convert_to_wav(input_path: str, output_path: str,
                   sr: int = 16000, duration: float = None,
                   start: float = None) -> bool:
    """Convert any audio format to 16kHz mono WAV using ffmpeg."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", input_path]
    if start is not None:
        cmd += ["-ss", str(start)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-ar", str(sr), "-ac", "1", "-loglevel", "error", output_path]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        return r.returncode == 0 and os.path.exists(output_path)
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════
# Source-specific downloaders
# ═══════════════════════════════════════════════════════════════════════

def download_librispeech(limit=None):
    """
    LibriSpeech via HuggingFace datasets library.
    Downloads train-clean-100 and train-clean-360 splits.
    """
    from datasets import load_dataset
    import soundfile as sf

    source = "librispeech"
    needed = get_needed_paths(source)
    if not needed:
        print(f"  No {source} paths needed in JSONLs. Skipping.")
        return

    prog = load_progress(source)
    done_set = set(prog["completed"])

    todo = [p for p in needed if p not in done_set and not os.path.exists(p)]
    print(f"  {source}: {len(needed):,} needed, {len(todo):,} to download")

    if not todo:
        print(f"  ✓ All {source} files already present.")
        return

    # Parse splits from paths: data/audio/librispeech/train-clean-360/...
    splits_needed = set()
    for p in todo:
        parts = p.split("/")
        if len(parts) >= 4:
            splits_needed.add(parts[3])  # e.g., "train-clean-360"

    for split in sorted(splits_needed):
        if _pause_requested:
            break
        hf_split = split.replace("-", ".")  # "train-clean-360" → "train.clean.360"
        print(f"\n  Loading HuggingFace split: {split} ({hf_split})...")

        try:
            ds = load_dataset(
                "openslr/librispeech_asr", split=hf_split,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"  ⚠ Failed to load {split}: {e}")
            continue

        split_paths = [p for p in todo if f"/{split}/" in p]

        # Build lookup: speaker_id/chapter_id/filename → dataset index
        # Path format: data/audio/librispeech/train-clean-360/3686/171133/3686-171133-0001.flac
        needed_ids = set()
        path_lookup = {}
        for p in split_paths:
            fname = os.path.basename(p).replace(".flac", "")
            needed_ids.add(fname)
            path_lookup[fname] = p

        saved = 0
        for item in tqdm(ds, desc=f"  {split}", ncols=80):
            if _pause_requested:
                break
            if limit and saved >= limit:
                break

            file_id = item.get("id", "")
            if file_id not in needed_ids:
                continue

            out_path = path_lookup[file_id]
            if os.path.exists(out_path):
                prog["completed"].append(out_path)
                prog["skip"] += 1
                continue

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            audio = item["audio"]
            waveform = audio["array"]
            sr = audio["sampling_rate"]

            try:
                import numpy as np
                # Resample if needed
                if sr != SAMPLE_RATE:
                    import torchaudio
                    import torch
                    wav_t = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
                    wav_t = torchaudio.functional.resample(wav_t, sr, SAMPLE_RATE)
                    waveform = wav_t.squeeze(0).numpy()
                    sr = SAMPLE_RATE

                sf.write(out_path, waveform, sr)
                prog["completed"].append(out_path)
                prog["success"] += 1
                saved += 1
            except Exception as e:
                prog["failed"].append(out_path)
                prog["fail"] += 1

        save_progress(source, prog)
        print(f"    {split}: saved {saved}, skipped {prog['skip']}")


def download_audiocaps(limit=None):
    """
    AudioCaps via YouTube (yt-dlp).
    Requires: yt-dlp, ffmpeg
    """
    source = "audiocaps"
    needed = get_needed_paths(source)
    if not needed:
        print(f"  No {source} paths needed. Skipping.")
        return

    prog = load_progress(source)
    done_set = set(prog["completed"])
    todo = [p for p in needed if p not in done_set and not os.path.exists(p)]
    print(f"  {source}: {len(needed):,} needed, {len(todo):,} to download")

    if not todo:
        print(f"  ✓ All {source} files already present.")
        return

    # Need metadata CSV for start times
    meta_files = list(Path("data/metadata").glob("audiocaps*.csv"))
    if not meta_files:
        print(f"  ⚠ No audiocaps CSV found in data/metadata/. Cannot download.")
        print(f"    Download from: https://github.com/cdjkim/audiocaps")
        return

    # Build lookup: youtube_id → start_time
    yt_lookup = {}
    for mf in meta_files:
        with open(mf) as f:
            for row in csv.DictReader(f):
                yt_id = row.get('youtube_id', '')
                raw_start = row.get('start_time')
                start = int(raw_start) if raw_start else 0
                if yt_id:
                    yt_lookup[yt_id] = start

    for out_path in tqdm(todo, desc=f"  {source}", ncols=80):
        if _pause_requested:
            break
        if limit and prog["success"] >= limit:
            break

        # Extract youtube_id from filename: Y<youtube_id>.wav
        fname = os.path.basename(out_path)
        yt_id = fname.replace("Y", "").replace(".wav", "")
        start = yt_lookup.get(yt_id, 0)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Download with yt-dlp
        tmp_base = out_path.replace('.wav', '_tmp')
        tmp_template = tmp_base + ".%(ext)s"
        url = f"https://www.youtube.com/watch?v={yt_id}"

        try:
            r = subprocess.run([
                "yt-dlp", "--quiet", "--no-warnings", "-x",
                "--audio-quality", "0", "-o", tmp_template, url
            ], capture_output=True, timeout=60)

            if r.returncode != 0:
                prog["failed"].append(out_path)
                prog["fail"] += 1
                continue

            tmp_files = glob.glob(tmp_base + ".*")
            if not tmp_files:
                prog["failed"].append(out_path)
                prog["fail"] += 1
                continue

            if convert_to_wav(tmp_files[0], out_path, duration=10, start=start):
                prog["completed"].append(out_path)
                prog["success"] += 1
            else:
                prog["failed"].append(out_path)
                prog["fail"] += 1

            # Cleanup temp
            for tf in glob.glob(tmp_base + ".*"):
                if tf != out_path:
                    os.remove(tf)

        except Exception:
            prog["failed"].append(out_path)
            prog["fail"] += 1

    save_progress(source, prog)


def download_musiccaps(limit=None):
    """
    MusicCaps via YouTube (yt-dlp). Same method as AudioCaps.
    MusicCaps metadata has youtube_id + start_s + end_s.
    """
    source = "musiccaps"
    needed = get_needed_paths(source)
    if not needed:
        print(f"  No {source} paths needed. Skipping.")
        return

    prog = load_progress(source)
    done_set = set(prog["completed"])
    todo = [p for p in needed if p not in done_set and not os.path.exists(p)]
    print(f"  {source}: {len(needed):,} needed, {len(todo):,} to download")

    if not todo:
        print(f"  ✓ All {source} files already present.")
        return

    # Load MusicCaps metadata for timestamps
    meta_files = list(Path("data/metadata").glob("musiccaps*.csv"))
    yt_lookup = {}
    for mf in meta_files:
        with open(mf) as f:
            for row in csv.DictReader(f):
                yt_id = row.get('ytid', '')
                raw_start = row.get('start_s')
                raw_end = row.get('end_s')
                start = float(raw_start) if raw_start else 0
                end = float(raw_end) if raw_end else 10
                if yt_id:
                    yt_lookup[yt_id] = (start, end - start)

    for out_path in tqdm(todo, desc=f"  {source}", ncols=80):
        if _pause_requested:
            break
        if limit and prog["success"] >= limit:
            break

        fname = os.path.basename(out_path).replace(".wav", "")
        start, duration = yt_lookup.get(fname, (0, 10))

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp_base = out_path.replace('.wav', '_tmp')
        tmp_template = tmp_base + ".%(ext)s"
        url = f"https://www.youtube.com/watch?v={fname}"

        try:
            r = subprocess.run([
                "yt-dlp", "--quiet", "--no-warnings", "-x",
                "--audio-quality", "0", "-o", tmp_template, url
            ], capture_output=True, timeout=60)

            if r.returncode == 0:
                tmp_files = glob.glob(tmp_base + ".*")
                if tmp_files and convert_to_wav(tmp_files[0], out_path,
                                                duration=duration, start=start):
                    prog["completed"].append(out_path)
                    prog["success"] += 1
                else:
                    prog["failed"].append(out_path)
                    prog["fail"] += 1
            else:
                prog["failed"].append(out_path)
                prog["fail"] += 1

            for tf in glob.glob(tmp_base + ".*"):
                if tf != out_path:
                    os.remove(tf)
        except Exception:
            prog["failed"].append(out_path)
            prog["fail"] += 1

    save_progress(source, prog)


def download_fsd50k(limit=None):
    """
    FSD50K from HuggingFace.
    Repository: google/FSD50K (or manual from Zenodo)
    """
    from datasets import load_dataset
    import soundfile as sf
    import numpy as np

    source = "fsd50k"
    needed = get_needed_paths(source)
    if not needed:
        print(f"  No {source} paths needed. Skipping.")
        return

    prog = load_progress(source)
    done_set = set(prog["completed"])
    todo = [p for p in needed if p not in done_set and not os.path.exists(p)]
    print(f"  {source}: {len(needed):,} needed, {len(todo):,} to download")

    if not todo:
        print(f"  ✓ All {source} files already present.")
        return

    # Build set of needed file IDs
    needed_ids = {}
    for p in todo:
        fname = os.path.basename(p).replace(".wav", "")
        needed_ids[fname] = p

    # Determine which splits to load
    splits_to_load = set()
    for p in todo:
        if "/dev/" in p:
            splits_to_load.add("dev")
        elif "/eval/" in p:
            splits_to_load.add("eval")

    for split in sorted(splits_to_load):
        if _pause_requested:
            break
        hf_split = "test" if split == "eval" else "train"
        print(f"\n  Loading FSD50K {split} (HuggingFace split: {hf_split})...")

        try:
            ds = load_dataset(
                "google/FSD50K", split=hf_split,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"  ⚠ Failed to load FSD50K: {e}")
            print(f"    Alternative: download from https://zenodo.org/records/4060432")
            continue

        saved = 0
        for item in tqdm(ds, desc=f"  FSD50K {split}", ncols=80):
            if _pause_requested:
                break
            if limit and saved >= limit:
                break

            file_id = str(item.get("fname", item.get("id", "")))
            if file_id not in needed_ids:
                continue

            out_path = needed_ids[file_id]
            if os.path.exists(out_path):
                continue

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            audio = item["audio"]
            waveform = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]

            try:
                if sr != SAMPLE_RATE:
                    import torchaudio, torch
                    wav_t = torch.tensor(waveform).unsqueeze(0)
                    wav_t = torchaudio.functional.resample(wav_t, sr, SAMPLE_RATE)
                    waveform = wav_t.squeeze(0).numpy()
                    sr = SAMPLE_RATE
                sf.write(out_path, waveform, sr)
                prog["completed"].append(out_path)
                prog["success"] += 1
                saved += 1
            except Exception:
                prog["failed"].append(out_path)
                prog["fail"] += 1

        save_progress(source, prog)


def download_clotho(limit=None):
    """
    Clotho from Zenodo.
    Downloads development, validation splits as zip.
    """
    source = "clotho"
    needed = get_needed_paths(source)
    if not needed:
        print(f"  No {source} paths needed. Skipping.")
        return

    prog = load_progress(source)
    done_set = set(prog["completed"])
    todo = [p for p in needed if p not in done_set and not os.path.exists(p)]
    print(f"  {source}: {len(needed):,} needed, {len(todo):,} to download")

    if not todo:
        print(f"  ✓ All {source} files already present.")
        return

    # Clotho download URLs (Zenodo)
    CLOTHO_URLS = {
        "development": "https://zenodo.org/records/4783391/files/clotho_audio_development.7z",
        "validation": "https://zenodo.org/records/4783391/files/clotho_audio_validation.7z",
        "evaluation": "https://zenodo.org/records/4783391/files/clotho_audio_evaluation.7z",
    }

    print(f"\n  ⚠ Clotho requires manual download from Zenodo:")
    for split, url in CLOTHO_URLS.items():
        split_dir = AUDIO_DIR / source / split
        needed_split = [p for p in todo if f"/{split}/" in p]
        if needed_split:
            print(f"    {split} ({len(needed_split):,} files):")
            print(f"      1. Download from: {url}")
            print(f"      2. Extract to: {split_dir}/")
    print(f"    3. Re-run this script to verify.")

    # Check if any files appeared
    found = 0
    for p in todo:
        if os.path.exists(p):
            prog["completed"].append(p)
            prog["skip"] += 1
            found += 1

    if found:
        print(f"\n  Found {found:,} files already on disk.")
        save_progress(source, prog)


def download_wavcaps(source_name, limit=None):
    """
    WavCaps from HuggingFace (cvssp/WavCaps).
    Downloads zip files from Zip_files/ directory in the repo.
    """
    source = source_name  # "wavcaps_soundbible" or "wavcaps_audioset_sl"
    needed = get_needed_paths(source)
    if not needed:
        print(f"  No {source} paths needed. Skipping.")
        return

    prog = load_progress(source)
    done_set = set(prog["completed"])
    todo = [p for p in needed if p not in done_set and not os.path.exists(p)]
    print(f"  {source}: {len(needed):,} needed, {len(todo):,} to download")

    if not todo:
        print(f"  ✓ All {source} files already present.")
        return

    from huggingface_hub import hf_hub_download

    # Map source to zip file path in repo
    zip_map = {
        "wavcaps_soundbible": "Zip_files/SoundBible/",
        "wavcaps_audioset_sl": "Zip_files/AudioSet_SL/",
    }

    zip_prefix = zip_map.get(source, "")
    out_dir = AUDIO_DIR / source
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n  Downloading {source} from cvssp/WavCaps ({zip_prefix})...")
    print(f"  This may take a while for AudioSet_SL (~33 GB)...")

    # List zip files in the repo
    from huggingface_hub import list_repo_tree

    try:
        tree = list(list_repo_tree("cvssp/WavCaps", path_in_repo=zip_prefix,
                                   repo_type="dataset"))
        zip_files = [f.rfilename for f in tree if f.rfilename.endswith('.zip')]
    except Exception as e:
        print(f"  ⚠ Failed to list WavCaps repo: {e}")
        print(f"    Try: pip install huggingface_hub --upgrade")
        return

    print(f"  Found {len(zip_files)} zip files")

    for zf in tqdm(zip_files, desc=f"  {source} zips", ncols=80):
        if _pause_requested:
            break

        try:
            local_zip = hf_hub_download(
                repo_id="cvssp/WavCaps",
                filename=zf,
                repo_type="dataset",
            )

            with zipfile.ZipFile(local_zip, 'r') as z:
                for member in z.namelist():
                    if not member.endswith(('.wav', '.flac', '.mp3', '.ogg')):
                        continue
                    basename = os.path.basename(member)
                    out_path = str(out_dir / basename)

                    if os.path.exists(out_path):
                        prog["skip"] += 1
                        continue

                    z.extract(member, path=str(out_dir / "_tmp"))
                    extracted = str(out_dir / "_tmp" / member)

                    if extracted.endswith('.wav'):
                        shutil.move(extracted, out_path)
                    else:
                        convert_to_wav(extracted, out_path)
                        os.remove(extracted)

                    prog["completed"].append(out_path)
                    prog["success"] += 1

            # Cleanup tmp
            tmp_dir = out_dir / "_tmp"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

        except Exception as e:
            print(f"\n  ⚠ Failed on {zf}: {e}")
            prog["fail"] += 1

    save_progress(source, prog)


def download_nsynth(limit=None):
    """
    NSynth from Google Magenta.
    Downloads tar.gz files for valid + test splits.
    """
    source = "nsynth"
    needed = get_needed_paths(source)
    if not needed:
        print(f"  No {source} paths needed. Skipping.")
        return

    prog = load_progress(source)
    done_set = set(prog["completed"])
    todo = [p for p in needed if p not in done_set and not os.path.exists(p)]
    print(f"  {source}: {len(needed):,} needed, {len(todo):,} to download")

    if not todo:
        print(f"  ✓ All {source} files already present.")
        return

    NSYNTH_URLS = {
        "valid": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz",
        "test": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz",
    }

    out_dir = AUDIO_DIR / source
    os.makedirs(out_dir, exist_ok=True)

    # Determine needed splits
    needed_basenames = {os.path.basename(p).replace(".wav", "") for p in todo}

    for split, url in NSYNTH_URLS.items():
        if _pause_requested:
            break

        print(f"\n  Downloading NSynth {split} ({url})...")
        print(f"  ⚠ This is a large download (~3 GB for valid, ~800 MB for test)")

        tar_path = out_dir / f"nsynth-{split}.tar.gz"

        # Download with wget
        if not tar_path.exists():
            try:
                r = subprocess.run(
                    ["wget", "-q", "--show-progress", "-O", str(tar_path), url],
                    timeout=3600
                )
                if r.returncode != 0:
                    print(f"  ⚠ Download failed for {split}")
                    continue
            except Exception as e:
                print(f"  ⚠ Download error: {e}")
                continue

        # Extract only the audio files we need
        print(f"  Extracting needed audio from {split}...")
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                for member in tqdm(tar.getmembers(), desc=f"  NSynth {split}"):
                    if not member.name.endswith('.wav'):
                        continue
                    basename = os.path.basename(member.name).replace(".wav", "")
                    if basename not in needed_basenames:
                        continue

                    out_path = str(out_dir / f"{basename}.wav")
                    if os.path.exists(out_path):
                        prog["skip"] += 1
                        continue

                    # Extract to temp, move to final location
                    tar.extract(member, path=str(out_dir / "_tmp"))
                    extracted = str(out_dir / "_tmp" / member.name)
                    shutil.move(extracted, out_path)
                    prog["completed"].append(out_path)
                    prog["success"] += 1

            # Cleanup
            tmp_dir = out_dir / "_tmp"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

        except Exception as e:
            print(f"  ⚠ Extract error: {e}")
            prog["fail"] += 1

    save_progress(source, prog)


def download_esc50(limit=None):
    """
    ESC-50 from GitHub.
    Clones the repo and copies audio files.
    """
    source = "esc50"
    needed = get_needed_paths(source)
    if not needed:
        print(f"  No {source} paths needed. Skipping.")
        return

    prog = load_progress(source)
    done_set = set(prog["completed"])
    todo = [p for p in needed if p not in done_set and not os.path.exists(p)]
    print(f"  {source}: {len(needed):,} needed, {len(todo):,} to download")

    if not todo:
        print(f"  ✓ All {source} files already present.")
        return

    out_dir = AUDIO_DIR / source
    os.makedirs(out_dir, exist_ok=True)

    # Clone ESC-50 repo
    repo_dir = DATA_DIR / "_repos" / "ESC-50"
    if not repo_dir.exists():
        print(f"  Cloning ESC-50 from GitHub...")
        r = subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/karolpiczak/ESC-50.git",
            str(repo_dir)
        ], capture_output=True)
        if r.returncode != 0:
            print(f"  ⚠ Git clone failed: {r.stderr.decode()}")
            return

    # Copy needed audio files
    audio_src = repo_dir / "audio"
    for out_path in tqdm(todo, desc=f"  {source}", ncols=80):
        fname = os.path.basename(out_path)
        src = audio_src / fname
        if src.exists():
            shutil.copy2(str(src), out_path)
            prog["completed"].append(out_path)
            prog["success"] += 1
        else:
            prog["failed"].append(out_path)
            prog["fail"] += 1

    save_progress(source, prog)


def download_urbansound8k(limit=None):
    """
    UrbanSound8K requires manual registration and download.
    """
    source = "urbansound8k"
    needed = get_needed_paths(source)
    if not needed:
        print(f"  No {source} paths needed. Skipping.")
        return

    prog = load_progress(source)
    done_set = set(prog["completed"])
    todo = [p for p in needed if p not in done_set and not os.path.exists(p)]

    if not todo:
        print(f"  ✓ All {source} files already present.")
        return

    print(f"\n  ⚠ UrbanSound8K requires manual download (registration required):")
    print(f"    1. Register at: https://urbansounddataset.weebly.com/urbansound8k.html")
    print(f"    2. Download UrbanSound8K.tar.gz")
    print(f"    3. Extract audio to: data/audio/urbansound8k/")
    print(f"       Each file should be at: data/audio/urbansound8k/<fold>/<filename>.wav")
    print(f"    4. Re-run this script to verify.")
    print(f"    Need: {len(todo):,} files")

    # Check if any already there
    found = sum(1 for p in todo if os.path.exists(p))
    if found:
        print(f"    Found {found:,} already on disk.")


def download_jamendo(limit=None):
    """
    JamendoMaxCaps from HuggingFace.
    """
    from datasets import load_dataset
    import soundfile as sf
    import numpy as np

    source = "jamendo"
    needed = get_needed_paths(source)
    if not needed:
        print(f"  No {source} paths needed. Skipping.")
        return

    prog = load_progress(source)
    done_set = set(prog["completed"])
    todo = [p for p in needed if p not in done_set and not os.path.exists(p)]
    print(f"  {source}: {len(needed):,} needed, {len(todo):,} to download")

    if not todo:
        print(f"  ✓ All {source} files already present.")
        return

    needed_ids = {os.path.basename(p).replace(".wav", ""): p for p in todo}

    print(f"  Loading JamendoMaxCaps from HuggingFace...")
    try:
        ds = load_dataset("amaai-lab/JamendoMaxCaps", split="train",
                          trust_remote_code=True, streaming=True)
    except Exception as e:
        print(f"  ⚠ Failed to load JamendoMaxCaps: {e}")
        return

    saved = 0
    for item in tqdm(ds, desc=f"  {source}", total=len(needed_ids), ncols=80):
        if _pause_requested:
            break
        if limit and saved >= limit:
            break
        if not needed_ids:
            break

        file_id = str(item.get("id", item.get("track_id", "")))
        if file_id not in needed_ids:
            continue

        out_path = needed_ids.pop(file_id)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        try:
            audio = item["audio"]
            waveform = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
            if sr != SAMPLE_RATE:
                import torchaudio, torch
                wav_t = torch.tensor(waveform).unsqueeze(0)
                wav_t = torchaudio.functional.resample(wav_t, sr, SAMPLE_RATE)
                waveform = wav_t.squeeze(0).numpy()
                sr = SAMPLE_RATE
            sf.write(out_path, waveform, sr)
            prog["completed"].append(out_path)
            prog["success"] += 1
            saved += 1
        except Exception:
            prog["failed"].append(out_path)
            prog["fail"] += 1

    save_progress(source, prog)


# ═══════════════════════════════════════════════════════════════════════
# Status report
# ═══════════════════════════════════════════════════════════════════════

def show_status():
    """Show download progress for all sources."""
    ALL_SOURCES = [
        "librispeech", "audiocaps", "fsd50k", "wavcaps_audioset_sl",
        "clotho", "musiccaps", "nsynth", "urbansound8k", "jamendo",
        "esc50", "wavcaps_soundbible",
    ]

    print(f"\n{'='*70}")
    print(f"  Audio Download Status")
    print(f"{'='*70}")
    print(f"  {'Source':<25} {'Needed':>8} {'OnDisk':>8} {'Missing':>8} {'Status'}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")

    total_needed = 0
    total_on_disk = 0

    for src in ALL_SOURCES:
        needed = get_needed_paths(src)
        on_disk = sum(1 for p in needed if os.path.exists(p))
        missing = len(needed) - on_disk
        total_needed += len(needed)
        total_on_disk += on_disk

        if len(needed) == 0:
            status = "—"
        elif missing == 0:
            status = "✅ Done"
        elif on_disk > 0:
            pct = on_disk / len(needed) * 100
            status = f"🔄 {pct:.0f}%"
        else:
            status = "❌ None"

        print(f"  {src:<25} {len(needed):>8,} {on_disk:>8,} {missing:>8,} {status}")

    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8}")
    print(f"  {'TOTAL':<25} {total_needed:>8,} {total_on_disk:>8,} "
          f"{total_needed - total_on_disk:>8,}")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

DOWNLOADERS = {
    "librispeech": download_librispeech,
    "audiocaps": download_audiocaps,
    "musiccaps": download_musiccaps,
    "fsd50k": download_fsd50k,
    "clotho": download_clotho,
    "wavcaps_soundbible": lambda limit=None: download_wavcaps("wavcaps_soundbible", limit),
    "wavcaps_audioset_sl": lambda limit=None: download_wavcaps("wavcaps_audioset_sl", limit),
    "nsynth": download_nsynth,
    "esc50": download_esc50,
    "urbansound8k": download_urbansound8k,
    "jamendo": download_jamendo,
}


def main():
    global _pause_requested

    import argparse
    ap = argparse.ArgumentParser(description="Download audio for ALM training")
    ap.add_argument("--source", default=None,
                    help=f"Source to download: {', '.join(DOWNLOADERS.keys())}, or 'all'")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max files to download per source (for testing)")
    ap.add_argument("--status", action="store_true",
                    help="Show download status for all sources")
    args = ap.parse_args()

    if args.status:
        show_status()
        return

    if args.source is None:
        print("\n  Audio Downloader for ALM Training")
        print("  " + "─" * 40)
        print(f"\n  Usage:")
        print(f"    python download_audio.py --source <name>")
        print(f"    python download_audio.py --source all")
        print(f"    python download_audio.py --source all --limit 10")
        print(f"    python download_audio.py --status")
        print(f"\n  Available sources:")
        for name in DOWNLOADERS:
            print(f"    {name}")
        print()
        return

    # Install signal handler
    signal.signal(signal.SIGINT, _signal_handler)

    if args.source == "all":
        # Download recommended order (smallest/easiest first)
        order = [
            "esc50", "wavcaps_soundbible", "nsynth",
            "clotho", "urbansound8k", "jamendo",
            "musiccaps", "fsd50k", "librispeech",
            "audiocaps", "wavcaps_audioset_sl",
        ]
        for src in order:
            if _pause_requested:
                break
            print(f"\n{'='*70}")
            print(f"  Downloading: {src}")
            print(f"{'='*70}")
            DOWNLOADERS[src](limit=args.limit)
    elif args.source in DOWNLOADERS:
        print(f"\n{'='*70}")
        print(f"  Downloading: {args.source}")
        print(f"{'='*70}")
        DOWNLOADERS[args.source](limit=args.limit)
    else:
        print(f"\n  Unknown source: {args.source}")
        print(f"  Available: {', '.join(DOWNLOADERS.keys())}")
        return

    # Final status
    show_status()


if __name__ == "__main__":
    main()