#!/usr/bin/env python3
"""
YouTube Transcription Tool
===========================
Downloads audio from YouTube videos and transcribes them using OpenAI Whisper.

Usage:
  python transcribe_youtube.py "https://www.youtube.com/watch?v=..." [--model medium]
  python transcribe_youtube.py "https://www.youtube.com/watch?v=..." --model large

Models:
  tiny     — fastest, ~1GB VRAM, ~97% accuracy of large
  base     — balanced, ~1GB VRAM, ~99% accuracy
  small    — good accuracy, ~2GB VRAM
  medium   — high accuracy, ~5GB VRAM (recommended)
  large    — best accuracy, ~10GB VRAM (gold standard)

Requirements:
  pip install yt-dlp openai-whisper

Features:
  - Downloads audio from YouTube (MP3 format)
  - Transcribes locally using Whisper
  - No API keys, no cost, runs offline
  - Outputs SRT (subtitles) and TXT (plain text)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

def download_youtube_audio(url: str, output_dir: str = ".", browser: str = "chrome", cookies_file: str = None) -> str:
    """
    Download audio from YouTube video using yt-dlp.
    
    Args:
        url: YouTube URL
        output_dir: Directory to save MP3
        browser: Browser to extract cookies from (chrome, firefox, safari)
        cookies_file: Optional path to cookies.txt file
    
    Returns:
        Path to downloaded audio file
    """
    print(f"\n📥 Downloading audio from YouTube...")
    print(f"   URL: {url}")
    
    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")
    
    cmd = [
        "yt-dlp",
        "-x",                           # Extract audio only
        "--audio-format", "mp3",        # MP3 format
        "--audio-quality", "192",       # 192kbps (sufficient for speech)
        "--user-agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "-o", output_template,          # Output path
    ]
    
    # Add cookies if available
    if cookies_file and os.path.exists(cookies_file):
        print(f"   🍪 Using cookies from: {cookies_file}")
        cmd.extend(["--cookies", cookies_file])
    else:
        print(f"   🍪 Attempting to extract cookies from {browser}...")
        cmd.extend(["--cookies-from-browser", browser])
    
    cmd.append(url)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"   ✅ Download successful")
        
        # Find the downloaded file
        mp3_files = list(Path(output_dir).glob("*.mp3"))
        if mp3_files:
            latest = max(mp3_files, key=os.path.getctime)
            print(f"   📁 Audio: {latest.name} ({latest.stat().st_size / 1024 / 1024:.1f} MB)")
            return str(latest)
    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.lower()
        
        # Provide helpful error messages
        if "403" in stderr_text or "forbidden" in stderr_text:
            print(f"   ⚠️  YouTube returned 403 Forbidden")
            print(f"      This video might be:")
            print(f"      • Age-restricted (requires login)")
            print(f"      • Geo-blocked in your region")
            print(f"      • Copyright-protected")
            print(f"\n      💡 Try with cookies from your browser:")
            print(f"         python transcribe_youtube.py '{url}' --browser chrome")
            print(f"         # or use: firefox, safari")
        else:
            print(f"   ❌ Download failed: {e.stderr}")
        
        sys.exit(1)
    
    return None


def transcribe_audio(audio_path: str, model: str = "medium", output_dir: str = ".") -> str:
    """
    Transcribe audio file using OpenAI Whisper.
    
    Args:
        audio_path: Path to audio file
        model: Whisper model size (tiny, base, small, medium, large)
        output_dir: Directory to save outputs
    
    Returns:
        Path to transcription (TXT file)
    """
    print(f"\n🎤 Transcribing audio with Whisper ({model} model)...")
    print(f"   Audio: {audio_path}")
    print(f"   ⏳ This may take a few minutes (depending on model and hardware)...")
    
    cmd = [
        "whisper",
        audio_path,
        "--model", model,
        "--output_format", "all",       # Generate TXT, SRT, VTT, JSON
        "--output_dir", output_dir,
        "--verbose", "False"            # Suppress debug output
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Find output files
        base_name = Path(audio_path).stem
        txt_file = Path(output_dir) / f"{base_name}.txt"
        srt_file = Path(output_dir) / f"{base_name}.srt"
        json_file = Path(output_dir) / f"{base_name}.json"
        
        if txt_file.exists():
            print(f"   ✅ Transcription complete")
            print(f"   📄 Text: {txt_file.name}")
            if srt_file.exists():
                print(f"   📺 Subtitles: {srt_file.name}")
            if json_file.exists():
                print(f"   📊 JSON: {json_file.name}")
            
            # Preview first few lines
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                preview = content[:300] + "..." if len(content) > 300 else content
                print(f"\n   Preview:\n   {preview.replace(chr(10), chr(10) + '   ')}\n")
            
            return str(txt_file)
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Transcription failed: {e.stderr}")
        sys.exit(1)
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe YouTube videos using yt-dlp + OpenAI Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe_youtube.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  python transcribe_youtube.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --model large
  python transcribe_youtube.py "https://youtu.be/dQw4w9WgXcQ" --model tiny
  
Cookies (for age-restricted/protected videos):
  python transcribe_youtube.py "URL" --browser chrome
  python transcribe_youtube.py "URL" --browser firefox
  python transcribe_youtube.py "URL" --cookies cookies.txt
        """
    )
    
    parser.add_argument(
        "url",
        help="YouTube URL (long or short format)"
    )
    parser.add_argument(
        "--model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: medium)"
    )
    parser.add_argument(
        "--browser",
        default="chrome",
        choices=["chrome", "firefox", "safari", "edge"],
        help="Browser to extract cookies from (default: chrome)"
    )
    parser.add_argument(
        "--cookies",
        default=None,
        help="Path to cookies.txt file (optional)"
    )
    parser.add_argument(
        "--output-dir",
        default="transcriptions",
        help="Output directory (default: transcriptions/)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Download and transcribe
    print("=" * 70)
    print("  YOUTUBE TRANSCRIPTION PIPELINE")
    print("=" * 70)
    
    audio_file = download_youtube_audio(args.url, args.output_dir, args.browser, args.cookies)
    if audio_file:
        txt_file = transcribe_audio(audio_file, args.model, args.output_dir)
        if txt_file:
            print("=" * 70)
            print(f"  ✅ SUCCESS")
            print(f"  📄 Transcription saved to: {txt_file}")
            print("=" * 70)


if __name__ == "__main__":
    main()
