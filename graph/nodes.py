"""
Node (agent) implementations for the SnapScholar LangGraph workflow.
"""

from typing import cast, List, Dict
import json
from pathlib import Path
import re
import cv2
import os
import google.generativeai as genai

from tools.youtube_tools import (
    extract_video_id,
    fetch_transcript,
    format_transcript_with_timestamps,
)
from tools.screenshot_tools import (
    extract_screenshots,
    validate_frame_quality,
    find_best_nearby_frame,
    extract_frame,
)
from tools.cache_tools import get_from_cache, save_to_cache
from config.prompts import (
    SUMMARIZATION_PROMPT,
    SCREENSHOT_PLANNING_PROMPT,
    TOPIC_EXTRACTION_PROMPT,
    TOPIC_TIMESTAMP_SELECTION_PROMPT,
    TOPIC_TIMESTAMP_SELECTION_BATCH_PROMPT,
)
from config.settings import settings
from .state import SnapScholarState


# Configure Gemini
genai.configure(api_key=settings.GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel(settings.MODEL_NAME)


def init_state_node(state: SnapScholarState) -> SnapScholarState:
    """
    First step in the workflow.
    Ensure errors list exists, set current_step, extract video_id
    """
    if "errors" not in state or state["errors"] is None:
        state["errors"] = []

    state["current_step"] = "init_state"

    if not state.get("video_id"):
        youtube_url = state.get("youtube_url")

        if not youtube_url:
            state["errors"].append("Missing youtube_url in state.")
            return state

        video_id = extract_video_id(cast(str, youtube_url))
        if video_id:
            state["video_id"] = video_id
        else:
            state["errors"].append("Could not extract video_id from youtube_url.")

    return state


def transcript_node(state: SnapScholarState) -> SnapScholarState:
    """
    Fetch the transcript for the video
    Stores: transcript_text, transcript_segments, transcript_with_timestamps, video_duration
    """
    state["current_step"] = "fetch_transcript"

    video_id = state.get("video_id")
    if not video_id:
        state["errors"].append("Cannot fetch transcript: missing video_id.")
        return state
    
    # --- Caching Logic ---
    cache_key = f"{video_id}:transcript"
    cached_data = get_from_cache(cache_key)
    if cached_data:
        print("  ✅ Using cached transcript.")
        state.update(cached_data)
        return state
    
    print("  Transcript not in cache, fetching from YouTube...")
    # --- End Caching Logic ---
    
    result = fetch_transcript(video_id)

    if not result["success"]:
        state["errors"].append(f"Transcript fetch failed: {result['error']}")
        return state

    # Prepare data to be used and cached
    transcript_data = {
        "transcript_text": result["text"],
        "transcript_segments": result["segments"],
        "video_duration": result["duration"],
        "transcript_with_timestamps": None, # Default value
    }
    if result["segments"]:
        transcript_data["transcript_with_timestamps"] = format_transcript_with_timestamps(
            result["segments"]
        )

    # Update state with the new data
    state.update(transcript_data)

    # --- Caching Logic ---
    # Save the prepared data to cache
    save_to_cache(cache_key, transcript_data)
    # --- End Caching Logic ---

    return state


def summarization_node(state: SnapScholarState) -> SnapScholarState:
    """
    Use Gemini to create structured study guide with ## Section ## markers
    Each section answers a question (What, How, Why)
    """
    state["current_step"] = "summarize"

    video_id = state.get("video_id")
    transcript_text = state.get("transcript_text")
    if not transcript_text:
        state["errors"].append("Cannot summarize: missing transcript_text.")
        return state

    # --- Caching Logic ---
    cache_key = f"{video_id}:summary"
    if video_id:
        cached_summary = get_from_cache(cache_key)
        if cached_summary:
            print("  ✅ Using cached summary.")
            state["summary"] = cached_summary
            return state
    
    print("  Summary not in cache, generating with Gemini...")
    # --- End Caching Logic ---

    prompt = SUMMARIZATION_PROMPT.format(transcript=transcript_text)

    try:
        response = gemini_model.generate_content(prompt)
        summary_text = response.text.strip()
        state["summary"] = summary_text
        
        # --- Caching Logic ---
        if video_id:
            save_to_cache(cache_key, summary_text)
        # --- End Caching Logic ---

    except Exception as e:
        state["errors"].append(f"Summarization failed: {e}")

    return state


def _safe_json_from_text(raw: str) -> any:
    """
    Parse JSON from Gemini response
    Tries direct parse first, then extracts JSON block
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Empty response from Gemini")

    # Try direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Extract JSON block
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find JSON object in response")

    json_str = raw[start : end + 1]
    return json.loads(json_str)


def filter_sections_for_visuals(sections: List[str], summary: str) -> List[str]:
    """
    Filter sections - only skip conclusion/takeaway sections
    Keep all question-based content sections
    
    Args:
        sections: All extracted section names
        summary: Full summary text
        
    Returns:
        Filtered list of sections for screenshots
    """
    
    # Only skip conclusion/summary sections
    skip_keywords = ['takeaway', 'takeaways', 'conclusion', 'summary', 'recap', 'wrap up', 'wrap-up']
    
    filtered = []
    
    for section in sections:
        section_lower = section.lower()
        
        # Skip only conclusion-type sections
        if any(keyword in section_lower for keyword in skip_keywords):
            print(f"     ⏭️  Skip: {section} (conclusion/summary)")
            continue
        
        # Keep all other sections
        filtered.append(section)
        print(f"     ✅ Keep: {section}")
    
    return filtered


def topic_extraction_node(state: SnapScholarState) -> SnapScholarState:
    """
    Extract sections from summary
    Each section represents a key question answered (What, How, Why)
    """
    state["current_step"] = "extract_topics"

    summary = state.get("summary")
    if not summary:
        state["errors"].append("Cannot extract sections: missing summary.")
        return state

    prompt = TOPIC_EXTRACTION_PROMPT.format(summary=summary)

    try:
        response = gemini_model.generate_content(prompt)
        raw = response.text or ""

        parsed = _safe_json_from_text(raw)
        
        # Try both "sections" and "topics" keys
        sections = parsed.get("sections") or parsed.get("topics", [])

        if not sections or not isinstance(sections, list):
            print("  ⚠️ LLM extraction failed, using regex fallback")
            sections = re.findall(r'##\s*([^#]+?)\s*##', summary)
            sections = [s.strip() for s in sections if s.strip()]

        if not sections:
            state["errors"].append("No sections extracted from summary")
            return state
        
        print(f"  📚 Extracted {len(sections)} sections from summary")
        
        # Filter sections (remove only conclusions)
        filtered = filter_sections_for_visuals(sections, summary)
        
        # Ensure minimum sections
        if len(filtered) < settings.MIN_SECTIONS_FOR_VISUALS and len(sections) >= settings.MIN_SECTIONS_FOR_VISUALS:
            print(f"  ⚠️ Only {len(filtered)} sections kept, using first {settings.MIN_SECTIONS_FOR_VISUALS}-{settings.MAX_SECTIONS_FOR_VISUALS}")
            # Keep first N sections, skip only last if it's a conclusion
            filtered = sections[:settings.MAX_SECTIONS_FOR_VISUALS]
            if filtered and any(kw in filtered[-1].lower() for kw in ['conclusion', 'takeaway', 'summary']):
                filtered = filtered[:-1]
        
        # Cap at maximum
        if len(filtered) > settings.MAX_SECTIONS_FOR_VISUALS:
            print(f"  ℹ️ Capped at {settings.MAX_SECTIONS_FOR_VISUALS} sections (had {len(filtered)})")
            filtered = filtered[:settings.MAX_SECTIONS_FOR_VISUALS]
        
        # Fallback: use all if still empty
        if not filtered and sections:
            print(f"  ⚠️ No sections after filtering, using all {len(sections)}")
            filtered = sections[:settings.MAX_SECTIONS_FOR_VISUALS]
        
        state["topics"] = filtered
        print(f"  ✅ Selected {len(filtered)} sections for visual extraction")

    except Exception as e:
        state["errors"].append(f"Section extraction failed: {e}")

    return state


def topic_timestamp_selection_node(state: SnapScholarState) -> SnapScholarState:
    """
    For each section, find the best timestamp in transcript
    Uses batch processing for efficiency
    Handles both numeric timestamps and time format (MM:SS)
    """
    state["current_step"] = "select_timestamps"

    topics = state.get("topics")
    transcript_ts = state.get("transcript_with_timestamps")

    if not topics:
        state["errors"].append("Cannot select timestamps: missing sections.")
        return state

    if not transcript_ts:
        state["errors"].append("Cannot select timestamps: missing transcript.")
        return state

    # Use batch prompt for efficiency
    prompt = TOPIC_TIMESTAMP_SELECTION_BATCH_PROMPT.format(
        topics=json.dumps(topics),
        transcript_with_timestamps=transcript_ts[:8000]
    )

    try:
        response = gemini_model.generate_content(prompt)
        raw = response.text or ""
        
        # Enhanced JSON extraction - handle markdown code blocks
        raw = re.sub(r'```json\s*', '', raw)
        raw = re.sub(r'```\s*', '', raw)
        raw = raw.strip()

        try:
            parsed = _safe_json_from_text(raw)
        except Exception as parse_err:
            print("\n==== JSON PARSE ERROR ====")
            print(f"Error: {parse_err}")
            print(f"Raw response (first 500 chars):\n{raw[:500]}")
            print("==========================\n")
            
            if raw.startswith('['):
                try:
                    parsed = {"screenshots": json.loads(raw)}
                except:
                    raise parse_err
            else:
                raise parse_err
        
        screenshots = parsed.get("screenshots", [])

        if not isinstance(screenshots, list):
            raise ValueError("Response is not a list")

        # Normalize timestamps
        topic_timestamps = []
        for item in screenshots:
            if not isinstance(item, dict):
                continue

            ts = item.get("timestamp")
            
            # Handle different timestamp formats
            try:
                # Try direct float conversion
                ts = float(ts)
            except (TypeError, ValueError):
                # Try parsing time format (MM:SS or HH:MM:SS)
                if isinstance(ts, str) and ':' in ts:
                    try:
                        parts = ts.split(':')
                        if len(parts) == 2:  # MM:SS
                            ts = int(parts[0]) * 60 + float(parts[1])
                        elif len(parts) == 3:  # HH:MM:SS
                            ts = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                        else:
                            print(f"  ⚠️ Invalid timestamp format: {ts}")
                            continue
                    except:
                        print(f"  ⚠️ Could not parse timestamp: {ts}")
                        continue
                else:
                    print(f"  ⚠️ Invalid timestamp: {ts}")
                    continue

            # Handle both "section" and "topic" keys
            section_name = item.get("section") or item.get("topic", "").strip()
            
            if not section_name:
                print(f"  ⚠️ Missing section name for timestamp {ts}")
                continue

            topic_timestamps.append({
                "topic": section_name,
                "timestamp": ts,
                "caption": item.get("caption", "Visual representation").strip(),
                "reason": item.get("reason", "Selected from transcript").strip()
            })

        if not topic_timestamps:
            state["errors"].append("No valid timestamps extracted")
            
            # Fallback: create default timestamps
            print("  ⚠️ No timestamps from LLM, using fallback strategy")
            video_duration = state.get("video_duration", 600)
            
            num_topics = len(topics)
            for i, topic in enumerate(topics):
                timestamp = 30 + (video_duration - 60) * (i + 1) / (num_topics + 1)
                topic_timestamps.append({
                    "topic": topic,
                    "timestamp": float(timestamp),
                    "caption": f"Visual for {topic}",
                    "reason": "Fallback: evenly distributed"
                })
            
            print(f"  ✅ Created {len(topic_timestamps)} fallback timestamps")
        else:
            print(f"  ✅ Selected {len(topic_timestamps)} timestamps for sections")
            for item in topic_timestamps:
                print(f"     - {item['topic'][:50]}: {item['timestamp']:.1f}s")

        state["topic_timestamps"] = topic_timestamps

    except Exception as e:
        state["errors"].append(f"Timestamp selection failed: {e}")
        
        # Final fallback
        print(f"  ❌ Timestamp selection failed: {e}")
        print("  ⚠️ Using emergency fallback: evenly distributed timestamps")
        
        topics = state.get("topics", [])
        video_duration = state.get("video_duration", 600)
        
        if topics and video_duration:
            fallback_timestamps = []
            num_topics = len(topics)
            
            for i, topic in enumerate(topics):
                timestamp = 30 + (video_duration - 60) * (i + 1) / (num_topics + 1)
                fallback_timestamps.append({
                    "topic": topic,
                    "timestamp": float(timestamp),
                    "caption": f"Visual for {topic}",
                    "reason": "Emergency fallback"
                })
            
            state["topic_timestamps"] = fallback_timestamps
            print(f"  ✅ Created {len(fallback_timestamps)} emergency fallback timestamps")

    return state


def frame_validation_node(state: SnapScholarState) -> SnapScholarState:
    """
    Validate frame quality using CV
    If frame is bad, try to find better nearby frame
    """
    # Suppress OpenCV warnings
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'loglevel;quiet'
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
    try:
        cv2.setLogLevel(0)
    except:
        pass
    
    state["current_step"] = "validate_frames"

    if not settings.ENABLE_CV_VALIDATION:
        print("  ⚠️ CV validation disabled, skipping")
        return state

    topic_timestamps = state.get("topic_timestamps")
    video_id = state.get("video_id")

    if not topic_timestamps:
        state["errors"].append("Cannot validate frames: missing topic_timestamps.")
        return state

    if not video_id:
        state["errors"].append("Cannot validate frames: missing video_id.")
        return state

    # Download video if needed
    from tools.screenshot_tools import download_video
    download_result = download_video(video_id)

    if not download_result['success']:
        state["errors"].append(f"Video download failed: {download_result['error']}")
        return state

    video_path = download_result['video_path']

    # Validate each frame
    validation_results = []
    video = cv2.VideoCapture(str(video_path))

    for item in topic_timestamps:
        timestamp = item['timestamp']
        topic = item['topic']

        # Extract frame
        video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        success, frame = video.read()

        if not success:
            validation_results.append({
                "topic": topic,
                "timestamp": timestamp,
                "is_valid": False,
                "reason": "Could not extract frame",
                "final_timestamp": timestamp
            })
            continue

        # Validate quality
        validation = validate_frame_quality(frame, settings)

        if validation['is_valid']:
            # Frame is good
            validation_results.append({
                "topic": topic,
                "timestamp": timestamp,
                "is_valid": True,
                "reason": "Good quality",
                "metrics": validation['metrics'],
                "final_timestamp": timestamp
            })
            print(f"  ✅ {topic}: Frame at {timestamp:.1f}s is good")
        else:
            # Frame is bad, try to find better one
            print(f"  ⚠️ {topic}: Frame at {timestamp:.1f}s is bad - {validation['reasons']}")
            print(f"     Searching for better frame nearby...")

            better = find_best_nearby_frame(
                video_path,
                timestamp,
                settings.FRAME_SEARCH_OFFSETS,
                settings
            )

            if better:
                # Found better frame
                validation_results.append({
                    "topic": topic,
                    "timestamp": timestamp,
                    "is_valid": True,
                    "reason": f"Used alternative at {better['timestamp']:.1f}s",
                    "final_timestamp": better['timestamp'],
                    "original_issues": validation['reasons']
                })
                print(f"     ✅ Found better frame at {better['timestamp']:.1f}s")

                # Update timestamp for this topic
                item['timestamp'] = better['timestamp']
            else:
                # No better frame found, keep original
                validation_results.append({
                    "topic": topic,
                    "timestamp": timestamp,
                    "is_valid": False,
                    "reason": "No better alternative found",
                    "original_issues": validation['reasons'],
                    "final_timestamp": timestamp
                })
                print(f"     ⚠️ No better frame found, keeping original")

    video.release()

    state["frame_validation"] = validation_results
    print(f"  ✅ Validated {len(validation_results)} frames")

    return state


def screenshot_extraction_node(state: SnapScholarState) -> SnapScholarState:
    """
    Extract frames at validated timestamps
    """
    state["current_step"] = "extract_screenshots"

    video_id = state.get("video_id")
    topic_timestamps = state.get("topic_timestamps")

    if not video_id:
        state["errors"].append("Cannot extract screenshots: missing video_id.")
        return state

    if not topic_timestamps:
        state["errors"].append("Cannot extract screenshots: missing topic_timestamps.")
        return state

    # Collect timestamps
    timestamps = [item['timestamp'] for item in topic_timestamps]

    if not timestamps:
        state["errors"].append("No valid timestamps to extract.")
        return state

    result = extract_screenshots(video_id=video_id, timestamps=timestamps)

    if not result["success"]:
        state["errors"].append(f"Screenshot extraction failed: {result['error']}")
        return state

    # Add topic information to screenshots
    screenshots_with_topics = []
    for i, screenshot in enumerate(result["screenshots"]):
        if i < len(topic_timestamps):
            screenshot['topic'] = topic_timestamps[i]['topic']
            screenshot['caption'] = topic_timestamps[i]['caption']
        screenshots_with_topics.append(screenshot)

    state["screenshots"] = screenshots_with_topics
    state["video_path"] = str(result["video_path"])
    state["video_dir"] = str(result["video_dir"])

    return state


def document_assembly_node(state: SnapScholarState) -> SnapScholarState:
    """
    Build Markdown study guide with inline screenshots
    """
    state["current_step"] = "assemble_document"

    summary = state.get("summary") or ""
    screenshots = state.get("screenshots") or []
    video_id = state.get("video_id") or "video"

    if not summary:
        state["errors"].append("Document assembly skipped: missing summary")
        return state

    # Target directory
    base_dir: Path = settings.TEMP_DIR / video_id
    base_dir.mkdir(parents=True, exist_ok=True)

    doc_path = base_dir / "study_guide.md"

    # Build markdown with inline images
    lines = []

    # Title
    lines.append(f"# SnapScholar Study Guide — {video_id}\n")

    # Split summary by sections
    sections = re.split(r'(##[^#]+##)', summary)

    # Process sections and insert images
    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Check if this is a section header
        match = re.match(r'##\s*([^#]+?)\s*##', section)
        if match:
            section_name = match.group(1).strip()
            lines.append(f"\n## {section_name}\n")

            # Find screenshot for this section
            matching_screenshots = [
                s for s in screenshots
                if s.get('topic', '').lower() == section_name.lower()
            ]

            if matching_screenshots:
                shot = matching_screenshots[0]
                lines.append(f"![{shot.get('caption', 'Visual')}]({shot['path'].name})\n")
                lines.append(f"*{shot.get('caption', 'Visual aid')} (t={shot['timestamp']:.1f}s)*\n")
        else:
            # Regular content
            lines.append(section + "\n")

    # Write file
    doc_text = "\n".join(lines)
    doc_path.write_text(doc_text, encoding="utf-8")

    state["document_link"] = str(doc_path)
    print(f"  ✅ Document assembled: {doc_path}")

    return state


# Fallback: Original screenshot planning (for comparison)
def screenshot_planning_node(state: SnapScholarState) -> SnapScholarState:
    """
    Original method: Plan screenshots based on transcript only
    """
    state["current_step"] = "plan_screenshots"

    transcript_ts = state.get("transcript_with_timestamps") or ""
    summary = state.get("summary") or ""
    max_shots = getattr(settings, "MAX_SCREENSHOTS", 8)

    if not transcript_ts or not summary:
        state["errors"].append(
            "Screenshot planning skipped: missing transcript_with_timestamps or summary"
        )
        return state

    prompt = SCREENSHOT_PLANNING_PROMPT.format(
        transcript_with_timestamps=transcript_ts[:8000],
        summary=summary[:4000],
        max_screenshots=max_shots,
    )

    try:
        response = gemini_model.generate_content(prompt)
        raw = response.text or ""

        try:
            parsed = _safe_json_from_text(raw)
        except Exception as parse_err:
            print("\n==== RAW GEMINI RESPONSE FOR SCREENSHOT PLANNING (truncated) ====")
            print(raw[:1000])
            print("=================================================================\n")
            raise ValueError(f"JSON parse failed: {parse_err}")

        if isinstance(parsed, dict):
            plan = parsed.get("screenshots", [])
        else:
            plan = parsed

        if not isinstance(plan, list):
            raise ValueError("Parsed JSON is not a list or missing 'screenshots'")

        normalized: List[Dict[str, any]] = []
        for item in plan:
            if not isinstance(item, dict):
                continue

            ts = item.get("timestamp")
            try:
                ts = float(ts)
            except (TypeError, ValueError):
                continue

            normalized.append(
                {
                    "timestamp": ts,
                    "caption": (item.get("caption") or "").strip(),
                    "summary_section": (item.get("summary_section") or "").strip(),
                    "concept": (item.get("concept") or "").strip(),
                }
            )

        if not normalized:
            state["errors"].append("Screenshot planning produced no valid entries")
        else:
            state["screenshot_plan"] = normalized

    except Exception as e:
        state["errors"].append(f"Screenshot planning failed: {e}")

    return state