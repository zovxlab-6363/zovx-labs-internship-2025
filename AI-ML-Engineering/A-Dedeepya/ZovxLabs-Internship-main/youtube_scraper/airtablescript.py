import requests
import json
import csv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

# Airtable setup
AIRTABLE_API_KEY = 'patyJk5KHtDiJhkWi.bfc5f9603718aeead85702392cb77e343cac74ef238854d695a38bf74e48c572'
AIRTABLE_BASE_ID = 'appfrUu523Ee2Xlaq'
AIRTABLE_TABLE_NAME = 'Transcripts'
AIRTABLE_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"

HEADERS = {
    'Authorization': f'Bearer {AIRTABLE_API_KEY}',
    'Content-Type': 'application/json'
}

CSV_FILE_PATH = r"C:\Users\akula\OneDrive\Desktop\youtube_scraper\Transcripts-Grid view.csv"

def load_existing_transcripts_from_csv(csv_path):
    transcripts_set = set()
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                transcript = row.get("transcript", "").strip()
                if transcript:
                    transcripts_set.add(transcript)
        print(f"✅ Loaded {len(transcripts_set)} unique transcripts from CSV.")
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
    return transcripts_set

def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return ' '.join([line['text'] for line in transcript])
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        print(f"❌ Transcript not available for {video_id}: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error for {video_id}: {e}")
        return None

def add_to_airtable(video_id, title, transcript):
    data = {
        "fields": {
            "video_id": video_id,
            "title": title,
            "transcript": transcript
        }
    }
    response = requests.post(AIRTABLE_URL, headers=HEADERS, json=data)
    if response.status_code == 200 or response.status_code == 201:
        print(f"✅ Added transcript for: {title} ({video_id})")
    else:
        print(f"❌ Error adding to Airtable for {video_id}: {response.status_code} - {response.text}")

def main():
    try:
        videos = []
        with open('varun_mayya_transcripts_with_titles.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    video = json.loads(line.strip())
                    videos.append(video)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Skipping invalid JSON line: {e}")
    except Exception as e:
        print(f"❌ Could not load JSONL file: {e}")
        return

    print(f"📚 Loaded {len(videos)} videos from JSONL.")
    existing_transcripts = load_existing_transcripts_from_csv(CSV_FILE_PATH)

    added_count = 0
    for video in videos:
        video_id = video.get("video_id")
        title = video.get("title", "Untitled Video")

        transcript = get_transcript(video_id)
        if transcript:
            if transcript in existing_transcripts:
                print(f"⚠️ Skipping (duplicate transcript): {title} ({video_id})")
            else:
                add_to_airtable(video_id, title, transcript)
                existing_transcripts.add(transcript)  # Add it to avoid future repeats
                added_count += 1

    print(f"\n✅ Finished. Total transcripts added: {added_count}")

if __name__ == "__main__":
    main()
