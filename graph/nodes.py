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


def screenshot_planning_node(state: SnapScholarState) -> SnapScholarState:
    """
    Use Gemini + SCREENSHOT_PLANNING_PROMPT to decide:
    - which timestamps to capture
    - what each screenshot represents

    Inputs:
        state['transcript_with_timestamps']
        state['summary']

    Output:
        state['screenshot_plan']  (list of dicts)
    """
    state["current_step"] = "plan_screenshots"

    transcript_with_ts = state.get("transcript_with_timestamps")
    summary = state.get("summary")

    if not transcript_with_ts or not summary:
        state["errors"].append(
            "Cannot plan screenshots: missing transcript_with_timestamps or summary."
        )
        return state

    prompt = SCREENSHOT_PLANNING_PROMPT.format(
        transcript_with_timestamps=transcript_with_ts,
        summary=summary,
        max_screenshots=settings.MAX_SCREENSHOTS,
    )

    try:
        response = gemini_model.generate_content(prompt)
        raw_text = response.text.strip()

        # Expect JSON with {"screenshots": [ ... ]}
        plan_obj = json.loads(raw_text)
        screenshots: List[Dict] = plan_obj.get("screenshots", [])
        state["screenshot_plan"] = screenshots

    except json.JSONDecodeError as e:
        state["errors"].append(f"Screenshot planning JSON parse failed: {e}")
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
