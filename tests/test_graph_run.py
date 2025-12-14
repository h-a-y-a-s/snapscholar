from graph.graph import run_snapscholar

TEST_URL = "https://www.youtube.com/watch?v=wjZofJX0v4M"

if __name__ == "__main__":
    state = run_snapscholar(TEST_URL)

    print("\nErrors:", state["errors"])

    print("\nSummary (first 400 chars):")
    print((state.get("summary") or "")[:400])

    print("\nScreenshot files:")
    for s in state.get("screenshots") or []:
        print(f"- t={s['timestamp']}s -> {s['path']}")

    print("\nDocument link:")
    print(state.get("document_link"))