import os
import json
import shlex
import subprocess
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
        f"ffmpeg -i {shlex.quote(video_file_path)} -ar 16000 -ac 1 -b:a 24k "
        f"{shlex.quote(audio_path)}"
    )
    print(f"Extracting audio: {video_file_path} -> {audio_path}")
    run_command(ffmpeg_command)
    return audio_path


def split_audio_if_needed(audio_path):
    """Split audio into smaller parts if it exceeds the size limit."""
    max_size_mb = 3.9
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)

    if file_size_mb <= max_size_mb:
        return [audio_path]

    print(f"Splitting {audio_path} ({file_size_mb:.2f} MB) into smaller parts...")

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    split_dir = os.path.join(AUDIO_DIR, "split")
    os.makedirs(split_dir, exist_ok=True)
    split_pattern = os.path.join(split_dir, f"{base_name}-split%03d.webm")

    split_command = (
        f"ffmpeg -i {shlex.quote(audio_path)} -ar 16000 -ac 1 -b:a 48k "
        f"-f segment -segment_time 600 -reset_timestamps 1 {shlex.quote(split_pattern)}"
    )
    run_command(split_command)

    split_files = sorted(
        [os.path.join(split_dir, f) for f in os.listdir(split_dir)],
        key=lambda x: x.lower(),
    )
    return split_files


def transcribe_audio_with_groq(audio_path, json_output_path):
    """Send audio file to Groq API for transcription."""
    print(f"Transcribing {audio_path} -> {json_output_path}...")
    client = Groq()
    with open(audio_path, "rb") as audio_file:
        retry_count = 0
        while retry_count < 5:
            try:
                transcription = client.audio.transcriptions.create(
                    file=(audio_path, audio_file.read()),
                    model="whisper-large-v3-turbo",
                    language="ko",
                    response_format="verbose_json",
                )
                with open(json_output_path, "w", encoding="utf-8") as json_file:
                    json.dump(transcription.to_dict(), json_file, ensure_ascii=False, indent=4)
                break
            except Exception as e:
                retry_count += 1
                wait_time = 60 * retry_count
                print(f"Error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)


def combine_json_segments(json_files, combined_json_path):
    """Combine multiple JSON transcription files into one."""
    print(f"Combining JSON files: {json_files} -> {combined_json_path}")
    combined_segments = []
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)
            combined_segments.extend(data.get("segments", []))

    combined_data = {"segments": combined_segments}
    with open(combined_json_path, "w", encoding="utf-8") as file:
        json.dump(combined_data, file, ensure_ascii=False, indent=4)


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
        split_files = split_audio_if_needed(audio_path)

        json_files = []
        for idx, split_file in enumerate(split_files, start=1):
            json_output_path = os.path.join(JSON_DIR, f"{base_name}-split{idx:03d}.json")
            transcribe_audio_with_groq(split_file, json_output_path)
            json_files.append(json_output_path)

        combined_json_path = os.path.join(JSON_DIR, f"{base_name}.json")
        combine_json_segments(json_files, combined_json_path)

        convert_json_to_srt(combined_json_path, srt_output_path)
        print(f"SRT saved to {srt_output_path}.")

        # Cleanup
        for file in split_files + json_files:
            os.remove(file)

    print("Processing complete.")


if __name__ == "__main__":
    process_video_files()
