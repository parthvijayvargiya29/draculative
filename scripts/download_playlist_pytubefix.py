#!/usr/bin/env python3
"""Download best audio for every video in a YouTube playlist using pytubefix.

Usage:
    python scripts/download_playlist_pytubefix.py PLAYLIST_URL OUTPUT_DIR

Example:
    python scripts/download_playlist_pytubefix.py "https://www.youtube.com/playlist?list=PL..." transcriptions/playlist_xxx
"""
import os
import sys
import argparse
from pathlib import Path

from pytubefix import Playlist


def safe_filename(name: str) -> str:
    # keep it simple: replace path separators
    return "".join(c if c not in "\\/" else "_" for c in name)


def download_playlist(url: str, outdir: str):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pl = Playlist(url)
    print(f"Playlist id: {pl.playlist_id}")
    i = 0
    for video in pl.videos:
        i += 1
        try:
            title = safe_filename(video.title or f"video_{video.video_id}")
        except Exception:
            title = f"video_{i:02d}_{video.video_id}"
        print(f"[{i:02d}] Downloading: {title} ({video.video_id})")
        try:
            streams = [s for s in video.fmt_streams if s.includes_audio_track]
            if not streams:
                print(f"  no audio streams found for {video.video_id}, skipping")
                continue
            # pick best abr
            best = max(streams, key=lambda s: s.abr or 0)
            filename = f"{i:02d} - {title}.m4a"
            outpath = best.download(output_path=str(outdir), filename=filename, skip_existing=True)
            print(f"  saved -> {outpath}")
        except Exception as e:
            print(f"  ERROR downloading {video.video_id}: {e}")
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('url', help='Playlist URL or watch URL with &list=...')
    parser.add_argument('out', nargs='?', default='transcriptions/playlist', help='Output directory')
    args = parser.parse_args()
    download_playlist(args.url, args.out)
