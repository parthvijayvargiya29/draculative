#!/usr/bin/env python3
"""
Transcribe YouTube Playlist → transcriptions/ICT2
===================================================
Downloads every video in the playlist as M4A audio (via pytubefix),
then transcribes each one with OpenAI Whisper (medium model).

Output layout:
  transcriptions/ICT2/
      audio/          ← raw M4A files
      transcripts/    ← .txt / .srt / .tsv / .vtt / .json per video

Usage:
    python transcribe_playlist_ICT2.py
    python transcribe_playlist_ICT2.py --model large
    python transcribe_playlist_ICT2.py --skip-download   # re-transcribe existing audio
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

from pytubefix import Playlist

PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLVgHx4Z63paYiFGQ56PjTF1PGePL3r69s"
BASE_DIR     = Path(__file__).parent / "transcriptions" / "ICT2"
AUDIO_DIR    = BASE_DIR / "audio"
TRANSCRIPT_DIR = BASE_DIR / "transcripts"
WHISPER_BIN  = Path(__file__).parent / ".venv" / "bin" / "whisper"


# ── helpers ──────────────────────────────────────────────────────────────────

def safe_filename(name: str) -> str:
    """Replace filesystem-unsafe characters."""
    keep = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
               '0123456789 ._-|()')
    return "".join(c if c in keep else "_" for c in name).strip()


def download_playlist_audio(url: str, outdir: Path) -> list[Path]:
    """Download all videos in the playlist as M4A and return their paths."""
    outdir.mkdir(parents=True, exist_ok=True)
    pl = Playlist(url)
    print(f"\n📋 Playlist : {pl.title if hasattr(pl, 'title') else url}")
    print(f"   Videos   : fetching list…")

    videos = list(pl.videos)
    total = len(videos)
    print(f"   Total    : {total} videos\n")

    audio_files: list[Path] = []

    for idx, video in enumerate(videos, start=1):
        try:
            title = safe_filename(video.title or f"video_{video.video_id}")
        except Exception:
            title = f"video_{idx:02d}_{video.video_id}"

        filename = f"{idx:02d} - {title}.m4a"
        dest = outdir / filename

        if dest.exists():
            print(f"[{idx:02d}/{total}] ⏭  Already downloaded: {filename}")
            audio_files.append(dest)
            continue

        print(f"[{idx:02d}/{total}] ⬇  Downloading: {title}")
        try:
            streams = [s for s in video.fmt_streams if s.includes_audio_track]
            if not streams:
                print(f"          ⚠️  No audio streams — skipping")
                continue
            best = max(streams, key=lambda s: s.abr or 0)
            out_path = best.download(output_path=str(outdir),
                                     filename=filename,
                                     skip_existing=True)
            print(f"          ✅ Saved → {Path(out_path).name}")
            audio_files.append(Path(out_path))
        except Exception as e:
            print(f"          ❌ Error: {e}")
            continue

    return audio_files


def transcribe_audio(audio_path: Path, model: str, out_dir: Path) -> None:
    """Run Whisper on a single audio file, saving outputs to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base_stem = audio_path.stem
    txt_out = out_dir / f"{base_stem}.txt"

    if txt_out.exists():
        print(f"   ⏭  Transcript exists — skipping")
        return

    whisper_cmd = str(WHISPER_BIN) if WHISPER_BIN.exists() else "whisper"
    cmd = [
        whisper_cmd,
        str(audio_path),
        "--model", model,
        "--output_format", "all",
        "--output_dir", str(out_dir),
        "--verbose", "False",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"   ✅ Transcribed → {txt_out.name}")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Whisper failed for {audio_path.name}")
        print(f"      {e.stderr[-400:] if e.stderr else '(no stderr)'}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download & transcribe the ICT2 YouTube playlist into transcriptions/ICT2/")
    parser.add_argument("--model", default="medium",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model (default: medium)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading; transcribe existing audio files only")
    args = parser.parse_args()

    print("=" * 70)
    print("  ICT2 PLAYLIST TRANSCRIPTION PIPELINE")
    print("=" * 70)
    print(f"  Playlist : {PLAYLIST_URL}")
    print(f"  Output   : {BASE_DIR}")
    print(f"  Model    : {args.model}")
    print("=" * 70)

    # Step 1 – download audio
    if args.skip_download:
        audio_files = sorted(AUDIO_DIR.glob("*.m4a"))
        print(f"\n⏭  Skipping download — found {len(audio_files)} audio files in {AUDIO_DIR}")
    else:
        audio_files = download_playlist_audio(PLAYLIST_URL, AUDIO_DIR)

    if not audio_files:
        print("\n⚠️  No audio files found. Exiting.")
        sys.exit(1)

    # Step 2 – transcribe each file
    print(f"\n🎤 Transcribing {len(audio_files)} audio files with Whisper ({args.model})…\n")
    for idx, audio_path in enumerate(sorted(audio_files), start=1):
        print(f"[{idx:02d}/{len(audio_files)}] {audio_path.name}")
        transcribe_audio(audio_path, args.model, TRANSCRIPT_DIR)

    print("\n" + "=" * 70)
    print(f"  ✅ ALL DONE")
    print(f"  📁 Audio files  : {AUDIO_DIR}")
    print(f"  📄 Transcripts  : {TRANSCRIPT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
