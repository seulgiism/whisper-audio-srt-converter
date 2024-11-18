import os
import json
import time
import re  # For parsing retry times
from groq import Groq

# Paths configuration
AUDIO_DIR = "/Volumes/Yoruichi/Movies/Bagel Girl/mp3/convert/"
JSON_OUTPUT_DIR = "/Volumes/Yoruichi/Movies/Bagel Girl/subs/json/"
SRT_OUTPUT_DIR = "/Volumes/Yoruichi/Movies/Bagel Girl/subs/srt/"

# Ensure output directories exist
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
os.makedirs(SRT_OUTPUT_DIR, exist_ok=True)

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

def extract_retry_time(error_message):
    """Extract retry time from error message (supports minutes and seconds)."""
    match = re.search(r"Please try again in (\d+)m([\d.]+)s", error_message)
    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        return int(minutes * 60 + seconds)
    match = re.search(r"Please try again in ([\d.]+)s", error_message)
    if match:
        return int(float(match.group(1)))
    return 60  # Default to 60 seconds if parsing fails

def process_audio_files():
    # Initialize the Groq client
    client = Groq()

    # List all audio files in the directory
    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith((".mp3", ".wav", ".m4a"))]

    if not audio_files:
        print("No audio files found in the directory.")
        return

    for audio_file in audio_files:
        audio_path = os.path.join(AUDIO_DIR, audio_file)
        json_output_path = os.path.join(JSON_OUTPUT_DIR, os.path.splitext(audio_file)[0] + ".json")
        srt_output_path = os.path.join(SRT_OUTPUT_DIR, os.path.splitext(audio_file)[0] + ".srt")

        # Skip already processed files
        if os.path.exists(json_output_path) and os.path.exists(srt_output_path):
            print(f"Skipping {audio_file}, already processed.")
            continue

        print(f"Processing {audio_file}...")

        while True:
            try:
                # Open the audio file
                with open(audio_path, "rb") as file:
                    # Create transcription
                    transcription = client.audio.transcriptions.create(
                        file=(audio_file, file.read()),  # Required audio file
                        model="whisper-large-v3-turbo",  # Model to use for transcription
                        language="ko",  # Language code
                        response_format="verbose_json",  # Response format
                        temperature=0.0  # Optional parameter
                    )
                
                # Convert transcription to dictionary
                transcription_data = transcription.to_dict()  # Replace with actual method if different

                # Save the JSON output
                with open(json_output_path, "w", encoding="utf-8") as json_file:
                    json.dump(transcription_data, json_file, ensure_ascii=False, indent=4)
                print(f"Saved JSON to {json_output_path}.")

                # Convert JSON to SRT
                convert_json_to_srt(json_output_path, srt_output_path)
                print(f"Saved SRT to {srt_output_path}.")

                # Add a delay to avoid hitting rate limits
                time.sleep(2)  # Adjust as necessary
                break  # Exit the retry loop after successful processing

            except Exception as e:
                if "rate_limit_exceeded" in str(e):
                    retry_time = extract_retry_time(str(e))
                    print(f"Rate limit hit. Retrying in {retry_time} seconds...")
                    time.sleep(retry_time)
                else:
                    print(f"Error processing {audio_file}: {e}")
                    break

    print("Processing complete.")

if __name__ == "__main__":
    process_audio_files()

