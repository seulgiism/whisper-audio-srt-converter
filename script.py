import os
import json
import shlex
import subprocess
import re
import time
from groq import Groq

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
        f"ffmpeg -i {shlex.quote(video_file_path)} -vn -ar 24000 -ac 1 -b:a 48k "
        f"{shlex.quote(audio_path)}"
    )
    print(f"Extracting audio: {video_file_path} -> {audio_path}")
    run_command(ffmpeg_command)
    return audio_path


def transcribe_audio_with_groq(audio_path, json_output_path):
    """Send audio file to Groq API for transcription."""
    print(f"Transcribing {audio_path} -> {json_output_path}...")
    client = Groq()
    with open(audio_path, "rb") as audio_file:
        while True:
            try:
                transcription = client.audio.transcriptions.create(
                    file=(audio_path, audio_file.read()),
                    model="whisper-large-v3-turbo",
                    language="ko",
                    response_format="verbose_json",
                )
                with open(json_output_path, "w", encoding="utf-8") as json_file:
                    json.dump(transcription.to_dict(), json_file, ensure_ascii=False, indent=4)
                break  # Successful transcription
            except Exception as e:
                error_message = str(e)
                print(f"Error: {error_message}")
                
                # Check if it's a rate limit error
                if "rate limit reached" in error_message.lower():
                    # Extract the "Please try again in X" time
                    retry_match = re.search(r"Please try again in (\d+m\d+\.\d+s)", error_message)
                    if retry_match:
                        retry_time = retry_match.group(1)
                        minutes, seconds = map(float, retry_time[:-1].split('m'))
                        total_wait_time = int(minutes * 60 + seconds)
                        print(f"Rate limit reached. Retrying in {total_wait_time} seconds...")
                        time.sleep(total_wait_time)
                    else:
                        print("Rate limit reached but no retry time specified. Retrying in 60 seconds...")
                        time.sleep(60)
                else:
                    raise e  # Re-raise other exceptions


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
    if os.path.exists(AUDIO_DIR):
        os.rmdir(AUDIO_DIR)
    if os.path.exists(JSON_DIR):
        os.rmdir(JSON_DIR)


if __name__ == "__main__":
    process_video_files()
