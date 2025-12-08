"""
YouTube video tools for transcript fetching
"""
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from typing import Dict, List, Optional
import re
import time


def extract_video_id(url: str) -> Optional[str]:
    """
    Extract video ID from various YouTube URL formats

    Supports:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    """
    patterns = [
        r'(?:v=|/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be/)([0-9A-Za-z_-]{11})'
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def fetch_transcript(video_id: str, languages: Optional[List[str]] = None) -> Dict:
    """
    Fetch transcript for a YouTube video using youtube-transcript-api 1.2.x

    Args:
        video_id: YouTube video ID
        languages: List of language codes to try (default: English)

    Returns:
        Dict with transcript data:
        {
            'text': str,          # Full transcript text
            'segments': List[Dict],  # Individual segments with timestamps
            'duration': float,    # Total video duration in seconds
            'success': bool,
            'error': str or None
        }
    """
    if languages is None:
        languages = ["en"]

    try:
        # New API: create an instance, then use .list() and .fetch()
        ytt_api = YouTubeTranscriptApi()

        # This replaces the old YouTubeTranscriptApi.list_transcripts(video_id)
        transcript_list = ytt_api.list(video_id)

        # Try to find transcript in preferred languages
        transcript = None
        try:
            # Try manual transcripts first
            transcript = transcript_list.find_transcript(languages)
        except Exception:
            # Fall back to auto-generated
            try:
                transcript = transcript_list.find_generated_transcript(languages)
            except Exception:
                # Finally, pick any available transcript
                for t in transcript_list:
                    transcript = t
                    break

        if transcript is None:
            raise NoTranscriptFound(f"No transcript available for video {video_id}")

        # New API: transcript.fetch() returns a FetchedTranscript object.
        # Use .to_raw_data() to get the classic list[dict] with text/start/duration.
        fetched = transcript.fetch()
        segments = fetched.to_raw_data()

        # Combine all text
        full_text = " ".join(segment["text"] for segment in segments)

        # Get duration from last segment
        duration = (
            segments[-1]["start"] + segments[-1]["duration"]
            if segments
            else 0.0
        )

        return {
            "text": full_text,
            "segments": segments,
            "duration": duration,
            "success": True,
            "error": None,
        }

    except TranscriptsDisabled:
        return {
            "text": None,
            "segments": None,
            "duration": None,
            "success": False,
            "error": "Transcripts are disabled for this video",
        }
    except NoTranscriptFound:
        return {
            "text": None,
            "segments": None,
            "duration": None,
            "success": False,
            "error": f"No transcript found in languages: {languages}",
        }
    except Exception as e:
        return {
            "text": None,
            "segments": None,
            "duration": None,
            "success": False,
            "error": f"Error fetching transcript: {str(e)}",
        }


def format_transcript_with_timestamps(segments: List[Dict], max_length: int = 10000) -> str:
    """
    Format transcript segments with timestamps for LLM analysis

    Args:
        segments: List of transcript segments
        max_length: Maximum character length (for token limits)

    Returns:
        Formatted string with timestamps
    """
    formatted_lines = []

    for segment in segments:
        timestamp = segment["start"]
        text = segment["text"].strip()

        # Format as [MM:SS] text
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)

        formatted_lines.append(f"[{minutes:02d}:{seconds:02d}] {text}")

    formatted_text = "\n".join(formatted_lines)

    # Truncate if too long
    if len(formatted_text) > max_length:
        formatted_text = formatted_text[:max_length] + "\n... (transcript truncated)"

    return formatted_text


def format_time(seconds: float) -> str:
    """
    Convert seconds to MM:SS format

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "03:45")
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def get_video_info(video_id: str) -> Dict:
    """
    Get basic video information

    Args:
        video_id: YouTube video ID

    Returns:
        Dict with video info
    """
    return {
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "thumbnail_url": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
    }


# Testing
if __name__ == "__main__":
    # Test with one video
    test_url = "https://www.youtube.com/watch?v=FE-hM1kRK4Y"

    print("=" * 60)
    print("SnapScholar YouTube Tools Test")
    print("=" * 60)
    print(f"\nTest URL: {test_url}\n")

    # Extract video ID
    video_id = extract_video_id(test_url)
    print(f"📹 Video ID: {video_id}\n")

    if not video_id:
        print("❌ Failed to extract video ID")
        raise SystemExit(1)

    # Fetch transcript
    print("Fetching transcript...")
    result = fetch_transcript(video_id)

    if result["success"]:
        print(f"\n✅ SUCCESS!")
        print(f"   Duration: {format_time(result['duration'])}")
        print(f"   Segments: {len(result['segments'])}")
        print(f"   Text length: {len(result['text'])} characters")
        print(f"\n   First 200 characters:")
        print(f"   {result['text'][:200]}...")

        # Show first 3 segments with timestamps
        if len(result["segments"]) >= 3:
            print(f"\n   First 3 segments with timestamps:")
            formatted = format_transcript_with_timestamps(result["segments"][:3])
            for line in formatted.split("\n"):
                print(f"   {line}")
    else:
        print(f"\n❌ FAILED")
        print(f"   Error: {result['error']}")
