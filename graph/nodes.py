"""
Node (agent) implementations for the SnapScholar LangGraph workflow.
"""

from typing import cast, List, Dict
import json

import google.generativeai as genai

from tools.youtube_tools import (
    extract_video_id,
    fetch_transcript,
    format_transcript_with_timestamps,
)
from tools.screenshot_tools import extract_screenshots
from config.prompts import (
    SUMMARIZATION_PROMPT,
    SCREENSHOT_PLANNING_PROMPT,
)
from config.settings import settings
from .state import SnapScholarState


# Configure Gemini once using settings.py
genai.configure(api_key=settings.GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel(settings.MODEL_NAME)


def init_state_node(state: SnapScholarState) -> SnapScholarState:
    """
    First step in the workflow.

    - Ensure 'errors' list exists
    - Set 'current_step' for debugging
    - If 'video_id' is missing, extract it from 'youtube_url'
    """
    # Ensure we have an errors list
    if "errors" not in state or state["errors"] is None:
        state["errors"] = []

    # Mark current step
    state["current_step"] = "init_state"

    # Extract video_id if not already present
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
    Fetch the transcript for the video and store:
    - transcript_text
    - transcript_segments
    - transcript_with_timestamps
    - video_duration
    """
    state["current_step"] = "fetch_transcript"

    video_id = state.get("video_id")
    if not video_id:
        state["errors"].append("Cannot fetch transcript: missing video_id.")
        return state

    result = fetch_transcript(video_id)

    if not result["success"]:
        state["errors"].append(f"Transcript fetch failed: {result['error']}")
        return state

    # Raw transcript info
    state["transcript_text"] = result["text"]
    state["transcript_segments"] = result["segments"]
    state["video_duration"] = result["duration"]

    # Formatted [MM:SS] transcript for screenshot planning
    if result["segments"]:
        state["transcript_with_timestamps"] = format_transcript_with_timestamps(
            result["segments"]
        )

    return state


def summarization_node(state: SnapScholarState) -> SnapScholarState:
    """
    Use Gemini + SUMMARIZATION_PROMPT to create a structured study guide.

    Input:
        state['transcript_text']
    Output:
        state['summary']
    """
    state["current_step"] = "summarize"

    transcript_text = state.get("transcript_text")
    if not transcript_text:
        state["errors"].append("Cannot summarize: missing transcript_text.")
        return state

    prompt = SUMMARIZATION_PROMPT.format(transcript=transcript_text)

    try:
        response = gemini_model.generate_content(prompt)
        summary_text = response.text.strip()
        state["summary"] = summary_text
    except Exception as e:
        state["errors"].append(f"Summarization failed: {e}")

    return state


def _safe_json_from_text(raw: str) -> any:
    """
    Try hard to parse JSON from a Gemini response:
    1) try json.loads directly
    2) if that fails, extract substring between first '{' and last '}' and parse that
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Empty response from Gemini")

    # 1) direct attempt
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2) extract JSON-looking block
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find JSON object in response")

    json_str = raw[start : end + 1]
    return json.loads(json_str)


def screenshot_planning_node(state: SnapScholarState) -> SnapScholarState:
    """Plan which moments to screenshot based on transcript + summary."""
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
            # Debug help: show what Gemini actually returned
            print("\n==== RAW GEMINI RESPONSE FOR SCREENSHOT PLANNING (truncated) ====")
            print(raw[:1000])
            print("=================================================================\n")
            raise ValueError(f"JSON parse failed: {parse_err}")

        # Either {"screenshots": [...]} or directly a list
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


def screenshot_extraction_node(state: SnapScholarState) -> SnapScholarState:
    """
    Use screenshot_tools.extract_screenshots to:
    - download the video (if needed)
    - extract frames at the chosen timestamps

    Inputs:
        state['video_id']
        state['screenshot_plan']

    Outputs:
        state['screenshots']  (list of {timestamp, path})
        state['video_path']
        state['video_dir']
    """
    state["current_step"] = "extract_screenshots"

    video_id = state.get("video_id")
    screenshot_plan = state.get("screenshot_plan")

    if not video_id:
        state["errors"].append("Cannot extract screenshots: missing video_id.")
        return state

    if not screenshot_plan:
        state["errors"].append("Cannot extract screenshots: missing screenshot_plan.")
        return state

    # Collect timestamps from the plan
    timestamps: List[float] = []
    for item in screenshot_plan:
        try:
            timestamps.append(float(item["timestamp"]))
        except Exception:
            # Skip malformed items
            continue

    if not timestamps:
        state["errors"].append("Screenshot plan contained no valid timestamps.")
        return state

    result = extract_screenshots(video_id=video_id, timestamps=timestamps)

    if not result["success"]:
        state["errors"].append(f"Screenshot extraction failed: {result['error']}")
        return state

    # Store results in state
    state["screenshots"] = result["screenshots"]           # [{timestamp, path}]
    state["video_path"] = str(result["video_path"])        # Path -> str
    state["video_dir"] = str(result["video_dir"])          # Path -> str

    return state
