"""
Test script for topic-based visual selection pipeline
Tests only the new approach on a single video
"""

from graph.graph import run_snapscholar
from tools.cache_tools import clear_cache
from config.settings import settings

# Test video
#TEST_VIDEO = "https://www.youtube.com/watch?v=8wAwLwJAGHs"
#TEST_VIDEO = "https://youtu.be/tadUeiNe5-g?si=trB5GJ1sZxqrXntA"
TEST_VIDEO = "https://youtu.be/qJeaCHQ1k2w?si=pMfWaI271_wVYyC-"
def print_separator(title=""):
    """Print a nice separator"""
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)
    print()


def print_state_summary(state):
    """Print a summary of the final state"""
    print_separator("TEST RESULTS")
    
    # Basic info
    print(f"Video ID: {state.get('video_id')}")
    print(f"Video Duration: {state.get('video_duration', 0):.1f}s")
    print(f"Errors: {len(state.get('errors', []))}")
    
    if state.get('errors'):
        print("\n❌ Errors:")
        for error in state['errors']:
            print(f"  - {error}")
    else:
        print("\n✅ No errors!")
    
    # Topics
    if state.get('topics'):
        print(f"\n📚 Topics Extracted: {len(state['topics'])}")
        for i, topic in enumerate(state['topics'], 1):
            print(f"  {i}. {topic}")
    
    # Topic timestamps
    if state.get('topic_timestamps'):
        print(f"\n⏱️  Timestamps Selected: {len(state['topic_timestamps'])}")
        for item in state['topic_timestamps']:
            print(f"  - {item['topic']}: {item['timestamp']:.1f}s")
    
    # Frame validation
    if state.get('frame_validation'):
        valid_count = sum(1 for v in state['frame_validation'] if v.get('is_valid'))
        total = len(state['frame_validation'])
        print(f"\n🔍 Frame Validation: {valid_count}/{total} passed CV checks")
        
        for v in state['frame_validation']:
            status = "✅" if v.get('is_valid') else "⚠️"
            reason = v.get('reason', 'N/A')
            print(f"  {status} {v['topic'][:50]}: {reason}")
    
    # Screenshots
    if state.get('screenshots'):
        print(f"\n📸 Screenshots Extracted: {len(state['screenshots'])}")
        for shot in state['screenshots']:
            print(f"  - t={shot['timestamp']:.1f}s: {shot.get('topic', 'N/A')[:50]}")
    
    # Document
    if state.get('document_link'):
        print(f"\n📄 Study Guide: {state['document_link']}")
    
    # Summary preview
    if state.get('summary'):
        # Count sections in summary
        import re
        sections = re.findall(r'##\s*([^#]+?)\s*##', state['summary'])
        print(f"\n📝 Summary: {len(sections)} sections found in text")


def run_test():
    """Run single test"""
    print_separator("SNAPSCHOLAR TOPIC-BASED VISUALS TEST")
    
    print(f"Test video: {TEST_VIDEO}")
    print(f"Settings: MIN={settings.MIN_SECTIONS_FOR_VISUALS}, MAX={settings.MAX_SECTIONS_FOR_VISUALS}")
    print()
    
    # Ensure new approach is enabled
    settings.USE_TOPIC_BASED_SELECTION = True
    
    # Run pipeline
    print("🚀 Running pipeline...\n")
    state = run_snapscholar(TEST_VIDEO)
    
    # Print results
    print_state_summary(state)
    
    # Final verdict
    print_separator("TEST COMPLETE")
    
    if state.get('errors'):
        print("❌ Test completed with errors")
        return False
    else:
        topics_count = len(state.get('topics', []))
        screenshots_count = len(state.get('screenshots', []))
        
        print(f"✅ Test completed successfully!")
        print(f"   Sections: {topics_count}")
        print(f"   Screenshots: {screenshots_count}")
        
        if topics_count >= settings.MIN_SECTIONS_FOR_VISUALS and topics_count <= settings.MAX_SECTIONS_FOR_VISUALS:
            print(f"   ✅ Section count within expected range ({settings.MIN_SECTIONS_FOR_VISUALS}-{settings.MAX_SECTIONS_FOR_VISUALS})")
        else:
            print(f"   ⚠️  Section count outside range (expected {settings.MIN_SECTIONS_FOR_VISUALS}-{settings.MAX_SECTIONS_FOR_VISUALS})")
        
        return True


if __name__ == "__main__":
    import sys
    
    if "--clear-cache" in sys.argv:
        print_separator("CLEARING CACHE")
        clear_cache()
        print("Cache cleared. Proceeding with test run...")

    success = run_test()
    sys.exit(0 if success else 1)