"""
LangGraph workflow for the SnapScholar pipeline.

Nodes:
- init_state       -> fill in video_id, set defaults
- fetch_transcript -> get transcript + timestamps
- summarize        -> Gemini summary
- extract_topics   -> Parse topics from summary
- select_timestamps -> Find best timestamp per topic
- validate_frames  -> CV quality checks
- extract_screenshots -> yt-dlp + OpenCV frame extraction
- assemble_document -> Create markdown with inline images
"""

from typing import Iterable

from langgraph.graph import StateGraph, END

from graph.state import SnapScholarState
from graph.nodes import (
    init_state_node,
    transcript_node,
    summarization_node,
    topic_extraction_node,
    topic_timestamp_selection_node,
    frame_validation_node,
    screenshot_extraction_node,
    document_assembly_node,
)
from config.settings import settings


def build_snapscholar_graph():
    """
    Define and compile the LangGraph workflow.
    """
    workflow = StateGraph(SnapScholarState)

    # Register nodes
    workflow.add_node("init_state", init_state_node)
    workflow.add_node("fetch_transcript", transcript_node)
    workflow.add_node("summarize", summarization_node)
    
    if settings.USE_TOPIC_BASED_SELECTION:
        # Topic-based pipeline
        workflow.add_node("extract_topics", topic_extraction_node)
        workflow.add_node("select_timestamps", topic_timestamp_selection_node)
        workflow.add_node("validate_frames", frame_validation_node)
    
    workflow.add_node("extract_screenshots", screenshot_extraction_node)
    workflow.add_node("assemble_document", document_assembly_node)

    # Wire edges
    workflow.set_entry_point("init_state")
    workflow.add_edge("init_state", "fetch_transcript")
    workflow.add_edge("fetch_transcript", "summarize")
    
    if settings.USE_TOPIC_BASED_SELECTION:
        # Topic-based flow
        workflow.add_edge("summarize", "extract_topics")
        workflow.add_edge("extract_topics", "select_timestamps")
        workflow.add_edge("select_timestamps", "validate_frames")
        workflow.add_edge("validate_frames", "extract_screenshots")
    else:
        # Original flow (fallback)
        workflow.add_edge("summarize", "extract_screenshots")
    
    workflow.add_edge("extract_screenshots", "assemble_document")
    workflow.add_edge("assemble_document", END)

    return workflow.compile()


# Create compiled app
snapscholar_app = build_snapscholar_graph()


def make_initial_state(youtube_url: str) -> SnapScholarState:
    """
    Helper to construct a fresh SnapScholarState from just a YouTube URL.
    """
    return {
        "youtube_url": youtube_url,
        "video_id": "",
        "transcript_text": None,
        "transcript_segments": None,
        "video_duration": None,
        "summary": None,
        "topics": None,
        "topic_timestamps": None,
        "frame_validation": None,
        "screenshot_plan": None,
        "screenshots": None,
        "video_path": None,
        "video_dir": None,
        "document_link": None,
        "errors": [],
        "current_step": "",
    }


def run_snapscholar(youtube_url: str) -> SnapScholarState:
    """
    Convenience wrapper: run the full graph and return the final state.
    """
    state = make_initial_state(youtube_url)
    final_state = snapscholar_app.invoke(state)
    return final_state


def run_snapscholar_stream(youtube_url: str) -> Iterable[SnapScholarState]:
    """
    Optional: stream intermediate states (useful later in Streamlit UI).
    """
    state = make_initial_state(youtube_url)
    for event in snapscholar_app.stream(state):
        for _node_name, node_state in event.items():
            yield node_state