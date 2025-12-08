# test_screenshot_nodes.py
"""
Manual test for screenshot-related nodes:
- init_state_node
- transcript_node
- summarization_node
- screenshot_planning_node
- screenshot_extraction_node
"""

from graph.state import SnapScholarState
from graph.nodes import (
    init_state_node,
    transcript_node,
    summarization_node,
    screenshot_planning_node,
    screenshot_extraction_node,
)

TEST_URL = "https://www.youtube.com/watch?v=wjZofJX0v4M"


def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70 + "\n")


def main():
    # ------------------------------------------------------------------
    # 1) Initial state
    # ------------------------------------------------------------------
    state: SnapScholarState = {
        "youtube_url": TEST_URL,
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

    print_header("INITIAL STATE")
    print(state)

    # ------------------------------------------------------------------
    # 2) Init node
    # ------------------------------------------------------------------
    state = init_state_node(state)
    print_header("AFTER init_state_node")
    print("keys:", state.keys())
    print("errors:", state["errors"])
    print("video_id:", state["video_id"])

    # ------------------------------------------------------------------
    # 3) Transcript node
    # ------------------------------------------------------------------
    state = transcript_node(state)
    print_header("AFTER transcript_node")
    print("keys:", state.keys())
    print("errors:", state["errors"])

    if state["transcript_text"]:
        print("\nFirst 200 chars of transcript_text:")
        print(state["transcript_text"][:200])

    # ------------------------------------------------------------------
    # 4) Summarization node
    # ------------------------------------------------------------------
    state = summarization_node(state)
    print_header("AFTER summarization_node")
    print("keys:", state.keys())
    print("errors:", state["errors"])

    if state.get("summary"):
        print("\nFirst 400 chars of summary:")
        print(state["summary"][:400])
    else:
        print("No summary produced, stopping.")
        return

    # ------------------------------------------------------------------
    # 5) Screenshot planning node (Gemini finds good timestamps)
    # ------------------------------------------------------------------
    state = screenshot_planning_node(state)
    print_header("AFTER screenshot_planning_node")
    print("keys:", state.keys())
    print("errors:", state["errors"])

    plan = state.get("screenshot_plan")
    if not plan:
        print("No screenshot_plan produced, stopping.")
        return

    print(f"\nScreenshot plan entries: {len(plan)}")
    for shot in plan:
        print(
            f"- t={shot.get('timestamp')}s | "
            f"section='{shot.get('summary_section')}' | "
            f"concept='{shot.get('concept')}'"
        )

    # ------------------------------------------------------------------
    # 6) Screenshot extraction node (uses yt-dlp + OpenCV)
    # ------------------------------------------------------------------
    state = screenshot_extraction_node(state)
    print_header("AFTER screenshot_extraction_node")
    print("keys:", state.keys())
    print("errors:", state["errors"])

    shots = state.get("screenshots") or []
    print(f"\nExtracted screenshots: {len(shots)}")
    for s in shots:
        print(f"- t={s['timestamp']}s -> {s['path']}")

    print("\nVideo path:", state.get("video_path"))
    print("Video dir:", state.get("video_dir"))


if __name__ == "__main__":
    main()
