import os
import json
import shlex
import subprocess
import re
import time
from groq import Groq
from groq._base_client import APIStatusError

# Directories configuration
VIDEO_DIR = "./"
AUDIO_DIR = os.path.join(VIDEO_DIR, "audio")
JSON_DIR = os.path.join(VIDEO_DIR, "json")
SRT_DIR = os.path.join(VIDEO_DIR, "srt")

# Ensure output directories exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(SRT_DIR, exist_ok=True)


def run_command(command):
    """Run a shell command."""
    try:
        subprocess.run(shlex.split(command), check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {command}")
        raise e


def convert_video_to_audio(video_file_path):
    """Convert video to audio in WebM format, optimized for size and quality."""
    base_name = os.path.splitext(os.path.basename(video_file_path))[0]
    audio_path = os.path.join(AUDIO_DIR, f"{base_name}.webm")
    ffmpeg_command = (
        f"ffmpeg -i {shlex.quote(video_file_path)} -vn -ar 16000 -ac 1 -b:a 46k "
        f"{shlex.quote(audio_path)}"
    )
    print(f"Extracting audio: {video_file_path} -> {audio_path}")
    run_command(ffmpeg_command)
    return audio_path


def extract_retry_time(error_message):
    """Extract retry time from rate limit error message."""
    match = re.search(r"Please try again in (\d+h)?(\d+m)?(\d+\.\d+s)?", error_message)
    if match:
        hours = int(match.group(1)[:-1]) if match.group(1) else 0
        minutes = int(match.group(2)[:-1]) if match.group(2) else 0
        seconds = float(match.group(3)[:-1]) if match.group(3) else 0
        return int(hours * 3600 + minutes * 60 + seconds)
    return 60  # Default to 60 seconds if no time is found


def transcribe_audio_with_groq(audio_path, json_output_path):
    """Send audio file to Groq API for transcription."""
    print(f"Transcribing {audio_path} -> {json_output_path}...")
    client = Groq()
    retry_count = 0
    max_retries = 5

    while retry_count < max_retries:
        try:
            with open(audio_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=(audio_path, audio_file.read()),
                    model="whisper-large-v3",
                    language="ko",
                    response_format="verbose_json",
                )
                with open(json_output_path, "w", encoding="utf-8") as json_file:
                    json.dump(transcription.to_dict(), json_file, ensure_ascii=False, indent=4)
                return  # Exit the function if successful
        except APIStatusError as e:
            error_message = str(e)
            print(f"Error: {error_message}")
            if "rate limit reached" in error_message.lower():
                retry_after = extract_retry_time(error_message)
                print(f"Rate limit reached. Retrying in {retry_after} seconds...")
                time.sleep(retry_after)
            else:
                retry_count += 1
                wait_time = 60 * retry_count
                print(f"Unexpected APIStatusError. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        except Exception as e:
            print(f"Unhandled error: {e}. Skipping file.")
            break
    print(f"Failed to transcribe {audio_path} after {max_retries} retries.")


def seconds_to_srt_timestamp(seconds):
    """Convert seconds to SRT timestamp format."""
    milliseconds = int((seconds % 1) * 1000)
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def convert_json_to_srt(json_path, srt_path):
    """Convert a JSON transcription file to SRT format."""
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    segments = data.get("segments", [])
    srt_lines = []

    for idx, segment in enumerate(segments, start=1):
        start_time = seconds_to_srt_timestamp(segment.get("start", 0))
        end_time = seconds_to_srt_timestamp(segment.get("end", 0))
        text = segment.get("text", "")
        srt_lines.append(f"{idx}\n{start_time} --> {end_time}\n{text}\n")

    with open(srt_path, "w", encoding="utf-8") as file:
        file.writelines(srt_lines)


def process_video_files():
    """Process all video files in the directory."""
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4", ".mkv", ".avi"))]
    if not video_files:
        print("No video files found in the directory.")
        return

    for video_file in video_files:
        video_file_path = os.path.join(VIDEO_DIR, video_file)
        base_name = os.path.splitext(video_file)[0]
        srt_output_path = os.path.join(SRT_DIR, f"{base_name}.srt")

        if os.path.exists(srt_output_path):
            print(f"Skipping {video_file}, SRT already exists.")
            continue

        print(f"Processing {video_file}...")
        audio_path = convert_video_to_audio(video_file_path)
        json_output_path = os.path.join(JSON_DIR, f"{base_name}.json")
        transcribe_audio_with_groq(audio_path, json_output_path)

        convert_json_to_srt(json_output_path, srt_output_path)
        print(f"SRT saved to {srt_output_path}.")

        # Cleanup
        os.remove(audio_path)
        os.remove(json_output_path)

    print("Processing complete.")
    # Cleanup remaining directories
    for directory in [AUDIO_DIR, JSON_DIR]:
        if os.path.exists(directory) and not os.listdir(directory):
            os.rmdir(directory)


if __name__ == "__main__":
    process_video_files()
