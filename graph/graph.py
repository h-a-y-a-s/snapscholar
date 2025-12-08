"""
LangGraph workflow for the SnapScholar pipeline.

Nodes:
- init_state       -> fill in video_id, set defaults
- fetch_transcript -> get transcript + timestamps
- summarize        -> Gemini summary
- plan_screenshots -> Gemini screenshot plan (timestamps + captions)
- extract_screenshots -> yt-dlp + OpenCV frame extraction
"""

from typing import Iterable

from langgraph.graph import StateGraph, END

from graph.state import SnapScholarState
from graph.nodes import (
    init_state_node,
    transcript_node,
    summarization_node,
    screenshot_planning_node,
    screenshot_extraction_node,
    document_assembly_node,
)


def build_snapscholar_graph():
    """
    Define and compile the LangGraph workflow.
    """
    workflow = StateGraph(SnapScholarState)

    # --- Register nodes ---
    workflow.add_node("init_state", init_state_node)
    workflow.add_node("fetch_transcript", transcript_node)
    workflow.add_node("summarize", summarization_node)
    workflow.add_node("plan_screenshots", screenshot_planning_node)
    workflow.add_node("extract_screenshots", screenshot_extraction_node)
    workflow.add_node("assemble_document", document_assembly_node)  

    # --- Wire the edges (linear pipeline for now) ---
    workflow.set_entry_point("init_state")
    workflow.add_edge("init_state", "fetch_transcript")
    workflow.add_edge("fetch_transcript", "summarize")
    workflow.add_edge("summarize", "plan_screenshots")
    workflow.add_edge("plan_screenshots", "extract_screenshots")
    workflow.add_edge("extract_screenshots", "assemble_document")
    workflow.add_edge("extract_screenshots", END)

    return workflow.compile()


# Create a compiled app we can import and reuse
snapscholar_app = build_snapscholar_graph()


def make_initial_state(youtube_url: str) -> SnapScholarState:
    """
    Helper to construct a fresh SnapScholarState from just a YouTube URL.
    Same structure you used in your manual tests.
    """
    return {
        "youtube_url": youtube_url,
        "video_id": "",
        "transcript_text": None,
        "transcript_segments": None,
        "video_duration": None,
        "summary": None,
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
        # event is a dict: {node_name: state_after_node}
        for _node_name, node_state in event.items():
            yield node_state
