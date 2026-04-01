#!/usr/bin/env python3
"""
scripts/transcribe_ph_macro.py
================================
Fetches YouTube captions for 15 PH macro videos and saves them into:

    transcriptions/PH_macro/transcripts/
        01 - <title>.txt    ← plain text corpus
        01 - <title>.json   ← structured (start, duration, text)

Uses youtube-transcript-api (no download, no Whisper — uses YouTube's
built-in caption data).  Falls back to a placeholder if a video has
no English captions.

Usage
-----
    python scripts/transcribe_ph_macro.py
    python scripts/transcribe_ph_macro.py --refresh   # overwrite existing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

# ── Repo root ──────────────────────────────────────────────────────────────────
_ROOT          = Path(__file__).parent.parent
TRANSCRIPT_DIR = _ROOT / "transcriptions" / "PH_macro" / "transcripts"

# ── The 15 video IDs + URLs (order determines filename prefix 01-15) ──────────
VIDEOS = [
    ("mk4vcHtawSo", "https://youtu.be/mk4vcHtawSo"),
    ("zAer-Mqe7tQ", "https://youtu.be/zAer-Mqe7tQ"),
    ("nOQqGy4boBY", "https://youtu.be/nOQqGy4boBY"),
    ("aQSDSqdlFxk", "https://youtu.be/aQSDSqdlFxk"),
    ("spg58Glfz68", "https://youtu.be/spg58Glfz68"),
    ("fz-Dan7NRss", "https://youtu.be/fz-Dan7NRss"),
    ("t5oisJiorsU", "https://youtu.be/t5oisJiorsU"),
    ("jIS2eB-rGv0", "https://youtu.be/jIS2eB-rGv0"),
    ("axqDLhWs93Q", "https://youtu.be/axqDLhWs93Q"),
    ("ijnkCt1QK6k", "https://youtu.be/ijnkCt1QK6k"),
    ("CbamEcNuDXo", "https://youtu.be/CbamEcNuDXo"),
    ("ybufqRY77PQ", "https://youtu.be/ybufqRY77PQ"),
    ("35HRPLVyF0g", "https://youtu.be/35HRPLVyF0g"),
    ("MX93U4KzA28", "https://youtu.be/MX93U4KzA28"),
    ("kS-muAuq62E", "https://youtu.be/kS-muAuq62E"),
]

LANG_PREF = ["en", "en-US", "en-GB", "en-CA", "en-AU"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe(name: str) -> str:
    name = re.sub(r'[^\w\s\-\(\)\|\.]+', '_', name)
    return re.sub(r'\s+', ' ', name).strip()[:80]


def _get_title(video_id: str) -> str:
    """Fetch video title via yt-dlp Python API (info extraction, no download)."""
    try:
        import yt_dlp
        opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}", download=False
            )
            return _safe(info.get("title", "")) if info else ""
    except Exception:
        return ""


def _fetch_transcript(video_id: str) -> Optional[List[dict]]:
    """Fetch caption segments via youtube-transcript-api (v1 API)."""
    from youtube_transcript_api import YouTubeTranscriptApi
    api = YouTubeTranscriptApi()
    # Try preferred languages first, then auto-generated
    for langs in [LANG_PREF, None]:
        try:
            fetched = api.fetch(video_id, languages=langs) if langs else api.fetch(video_id)
            return [
                {"text": s.text, "start": round(s.start, 2), "duration": round(s.duration, 2)}
                for s in fetched
            ]
        except Exception:
            continue
    return None


def _write_outputs(segments: List[dict], stem: str) -> None:
    # Plain text — one continuous corpus string
    plain = " ".join(s["text"].strip().replace("\n", " ") for s in segments if s["text"].strip())
    (TRANSCRIPT_DIR / f"{stem}.txt").write_text(plain, encoding="utf-8")
    # Structured JSON
    (TRANSCRIPT_DIR / f"{stem}.json").write_text(json.dumps(segments, indent=2), encoding="utf-8")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch YouTube captions for 15 PH macro videos "
                    "→ transcriptions/PH_macro/transcripts/",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Overwrite existing transcript files",
    )
    args = parser.parse_args()

    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(VIDEOS)

    print("=" * 70)
    print("  PH MACRO TRANSCRIPT FETCH  (youtube-transcript-api)")
    print("=" * 70)
    print(f"  Videos  : {total}")
    print(f"  Output  : {TRANSCRIPT_DIR.relative_to(_ROOT)}")
    print("=" * 70)

    ok = skip = fail = 0

    for idx, (vid_id, url) in enumerate(VIDEOS, start=1):
        print(f"\n[{idx:02d}/{total}]  {vid_id}  ({url})")

        print("    🔍 Title … ", end="", flush=True)
        title = _get_title(vid_id) or f"video_{idx:02d}"
        print(title[:65])

        stem     = f"{idx:02d} - {title}"
        txt_path = TRANSCRIPT_DIR / f"{stem}.txt"

        if txt_path.exists() and not args.refresh:
            sz = txt_path.stat().st_size
            print(f"    ⏭  Exists ({sz:,} bytes) — skipping")
            skip += 1
            continue

        print("    📥 Fetching captions … ", end="", flush=True)
        segments = _fetch_transcript(vid_id)

        if not segments:
            print("NO CAPTIONS AVAILABLE")
            txt_path.write_text(
                f"[NO CAPTIONS] Video: {url}\n"
                "No English captions found via YouTube API.\n",
                encoding="utf-8",
            )
            fail += 1
            continue

        _write_outputs(segments, stem)
        words = len(txt_path.read_text().split())
        print(f"✅  {len(segments)} segments, {words:,} words → {stem}.txt")
        ok += 1
        time.sleep(0.4)   # brief pause between API calls

    print()
    print("=" * 70)
    print(f"  ✅ Fetched       : {ok}")
    print(f"  ⏭  Skipped       : {skip}")
    print(f"  ❌ No captions   : {fail}")
    txts = sorted(TRANSCRIPT_DIR.glob("*.txt"))
    print(f"  .txt files      : {len(txts)}")
    print(f"  Folder          : {TRANSCRIPT_DIR.relative_to(_ROOT)}")
    print("=" * 70)

    if fail == total and ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()