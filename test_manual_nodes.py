"""
Manual test runner for the SnapScholar nodes.
Run with:
    python test_manual_nodes.py
"""

from graph.nodes import init_state_node, transcript_node, summarization_node
from graph.state import SnapScholarState


def line():
    print("\n" + "="*70 + "\n")


def main():

    # 1. Create a fresh clean state
    state: SnapScholarState = {
        #"youtube_url": "https://www.youtube.com/watch?v=FE-hM1kRK4Y",
        "youtube_url": "https://youtu.be/qJeaCHQ1k2w?si=AYmsm7Z5uL7B6zjf",
        "errors": [],
    }

    line()
    print("INITIAL STATE:")
    print(state)

    # 2. Test init_state_node
    state = init_state_node(state)

    line()
    print("AFTER init_state_node:")
    print("keys:", state.keys())
    print("errors:", state["errors"])
    print("video_id:", state.get("video_id"))

    # 3. Test transcript_node
    # Caching is now handled inside the node itself.
    state = transcript_node(state)

    line()
    print("AFTER transcript_node:")
    print("keys:", state.keys())
    print("errors:", state["errors"])

    print("\nFirst 300 chars of transcript_text:")
    print((state.get("transcript_text") or "")[:300])

    print("\nFirst 300 chars of transcript_with_timestamps:")
    print((state.get("transcript_with_timestamps") or "")[:300])

    # 4. Test summarization_node (Gemini)
    # Caching is now handled inside the node itself.
    state = summarization_node(state)

    line()
    print("AFTER summarization_node:")
    print("keys:", state.keys())
    print("errors:", state["errors"])

    if state.get("summary"):
        print("\nFirst 800 chars of summary:")
        print(state["summary"][:800])
    else:
        print("No summary produced.")


if __name__ == "__main__":
    main()
