import requests
from youtube_transcript_api import YouTubeTranscriptApi
import json
import time

API_KEY = "AIzaSyAxwV8PG5buFCmW4dqEiSwC0UHeFZxUuMA"  # 🔁 Replace with your actual key
CHANNEL_ID = "UCsQoiOrh7jzKmE8NBofhTnQ"  # Varun Mayya's channel
MAX_RESULTS = 50  # Max allowed per request

def fetch_all_videos_with_titles(api_key, channel_id):
    videos = []
    url = f"https://www.googleapis.com/youtube/v3/search"
    params = {
        "key": api_key,
        "channelId": channel_id,
        "part": "snippet",
        "order": "date",
        "maxResults": MAX_RESULTS,
        "type": "video"
    }

    while True:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"❌ Error fetching video list: {response.status_code} - {response.text}")
            break

        data = response.json()
        items = data.get("items", [])
        for item in items:
            video_id = item["id"].get("videoId")
            title = item["snippet"].get("title")
            if video_id and title:
                videos.append({"video_id": video_id, "title": title})

        next_page = data.get("nextPageToken")
        if not next_page:
            break

        params["pageToken"] = next_page
        time.sleep(0.5)  # be polite to the API

    print(f"📦 Total videos found: {len(videos)}")
    return videos

def fetch_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        print(f"❌ Could not get transcript for {video_id}: {e}")
        return None

def main():
    videos = fetch_all_videos_with_titles(API_KEY, CHANNEL_ID)
    data = []

    for video in videos:
        video_id = video["video_id"]
        title = video["title"]
        transcript = fetch_transcript(video_id)
        if transcript:
            data.append({
                "video_id": video_id,
                "title": title,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "transcript": transcript
            })

    # Save to JSONL
    with open("varun_mayya_transcripts_with_titles.jsonl", "w", encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ Saved transcripts with titles for {len(data)} videos.")

if __name__ == "__main__":
    main()
