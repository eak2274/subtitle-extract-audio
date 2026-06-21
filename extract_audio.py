#!/usr/bin/env python3
"""
Script for extracting audio tracks from video files.
Optimized for Whisper ASR: converts to WAV format, 16kHz, mono.

This script processes video files (.avi, .mkv) in a specified directory
and extracts their audio tracks using ffmpeg. The output format is
optimized for OpenAI's Whisper speech recognition model.

CLI parameters:

    input_dir
    Path to directory containing video files (.avi, .mkv).
    Required: Yes

    -o, --output
    Path to directory for saving extracted audio files.
    If not specified, audio files are saved in the same directory as video files.
    Required: No
    Default: same as input_dir

Output format:
    - Format: WAV (PCM)
    - Sample rate: 16 kHz
    - Channels: mono
    - Codec: pcm_s16le

Logging:
    - Logs are saved to ./logs/extract_audio_<timestamp>.log
    - Logs are also displayed in the console
    - One script run = one log file

Behavior:
    - Only processes .avi and .mkv files in the root of input_dir (no recursion)
    - Skips files if corresponding audio file already exists
    - Stops execution on any error (non-zero exit code)

Run command examples:

    # 1. Basic usage - save audio to same directory as videos
    python extract_audio.py /path/to/videos

    # 2. Save audio to separate directory
    python extract_audio.py /path/to/videos -o /path/to/audio

    # 3. Using full parameter name
    python extract_audio.py /path/to/videos --output /path/to/audio

    # 4. Using relative paths
    python extract_audio.py ./videos -o ./audio

    # 5. Show help
    python extract_audio.py --help

Requirements:
    - ffmpeg must be installed and available in PATH
    - Install ffmpeg:
        Ubuntu/Debian: sudo apt-get install ffmpeg
        macOS: brew install ffmpeg
        Windows: download from https://ffmpeg.org/download.html
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime


DEFAULT_FFMPEG_PATH = Path(r"C:\Tools\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe")


def setup_logging() -> logging.Logger:
    """
    Configure logging to both file and console.
    
    Returns:
        Configured logger instance
    """
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"extract_audio_{timestamp}.log"
    
    # Setup logger
    logger = logging.getLogger("AudioExtractor")
    logger.setLevel(logging.INFO)
    
    # Log format
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Log file: {log_file}")
    
    return logger


def find_ffmpeg() -> str | None:
    """
    Find ffmpeg executable from FFMPEG_BINARY, PATH, or the local Windows tools path.
    """
    env_binary = os.environ.get("FFMPEG_BINARY")
    if env_binary and Path(env_binary).is_file():
        return env_binary

    path_binary = shutil.which("ffmpeg")
    if path_binary:
        return path_binary

    if DEFAULT_FFMPEG_PATH.is_file():
        return str(DEFAULT_FFMPEG_PATH)

    return None


def check_ffmpeg() -> str | None:
    """
    Check if ffmpeg is installed and usable.
    
    Returns:
        Path to ffmpeg executable if available, otherwise None
    """
    ffmpeg_binary = find_ffmpeg()
    if not ffmpeg_binary:
        return None

    try:
        subprocess.run(
            [ffmpeg_binary, '-version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return ffmpeg_binary
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def extract_audio(video_path: Path, audio_path: Path, ffmpeg_binary: str, logger: logging.Logger) -> None:
    """
    Extract audio from video file using ffmpeg.
    
    Converts to WAV format optimized for Whisper:
    - 16 kHz sampling rate
    - Mono channel
    - PCM codec
    
    Args:
        video_path: Path to source video file
        audio_path: Path for saving audio file
        logger: Logger instance
        
    Raises:
        subprocess.CalledProcessError: If ffmpeg extraction fails
    """
    # ffmpeg parameters optimized for Whisper
    # -vn: no video
    # -acodec pcm_s16le: WAV format
    # -ar 16000: 16 kHz sample rate
    # -ac 1: mono (1 channel)
    cmd = [
        ffmpeg_binary,
        '-i', str(video_path),
        '-vn',  # no video
        '-acodec', 'pcm_s16le',  # WAV codec
        '-ar', '16000',  # 16 kHz
        '-ac', '1',  # mono
        '-y',  # overwrite without asking
        str(audio_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        logger.info(f"✓ Successfully extracted: {video_path.name} → {audio_path.name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Error processing {video_path.name}")
        logger.error(f"ffmpeg output: {e.stderr}")
        raise


def process_videos(input_dir: Path, output_dir: Path, ffmpeg_binary: str, logger: logging.Logger) -> None:
    """
    Process all video files in the specified directory.
    
    Args:
        input_dir: Directory containing video files
        output_dir: Directory for saving audio files
        logger: Logger instance
        
    Raises:
        Exception: Any error during processing (script stops on errors)
    """
    # Supported video formats
    video_extensions = {'.avi', '.mkv'}
    
    # Find all video files
    video_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ]
    
    if not video_files:
        logger.warning(f"No video files (.avi, .mkv) found in {input_dir}")
        return
    
    logger.info(f"Found video files: {len(video_files)}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Audio format: WAV, 16kHz, mono")
    logger.info("-" * 60)
    
    processed = 0
    skipped = 0
    
    for video_file in video_files:
        # Create audio file path
        audio_file = output_dir / f"{video_file.stem}.wav"
        
        # Check if audio file already exists
        if audio_file.exists():
            logger.info(f"⊘ Skipped (already exists): {audio_file.name}")
            skipped += 1
            continue
        
        # Extract audio
        extract_audio(video_file, audio_file, ffmpeg_binary, logger)
        processed += 1
    
    logger.info("-" * 60)
    logger.info(f"Processing completed!")
    logger.info(f"Processed: {processed}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Total: {len(video_files)}")


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Extract audio tracks from video files for Whisper ASR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/videos
  %(prog)s /path/to/videos -o /path/to/audio
  %(prog)s ./videos --output ./audio
        """
    )
    
    parser.add_argument(
        'input_dir',
        type=str,
        help='Path to directory containing video files (.avi, .mkv)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to directory for saving audio files (default: same as input_dir)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Audio Extraction Script")
    logger.info("=" * 60)
    
    # Check for ffmpeg
    logger.info("Checking for ffmpeg...")
    ffmpeg_binary = check_ffmpeg()
    if not ffmpeg_binary:
        logger.error("✗ ffmpeg not found in system!")
        logger.error("Install ffmpeg: https://ffmpeg.org/download.html")
        sys.exit(1)
    logger.info(f"ffmpeg found: {ffmpeg_binary}")
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"✗ Input directory does not exist: {input_dir}")
        sys.exit(1)
    if not input_dir.is_dir():
        logger.error(f"✗ Specified path is not a directory: {input_dir}")
        sys.exit(1)
    
    # Determine output directory
    output_dir = Path(args.output) if args.output else input_dir
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process video files
    try:
        process_videos(input_dir, output_dir, ffmpeg_binary, logger)
    except Exception as e:
        logger.error(f"✗ Critical error: {e}")
        logger.exception("Traceback:")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("Execution completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
