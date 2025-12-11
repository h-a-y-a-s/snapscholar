import streamlit as st
import re
from pathlib import Path
import time

# Internal imports from your existing codebase
from graph.graph import run_snapscholar_stream
from config.settings import settings

# --- Page Config ---
st.set_page_config(
    page_title="SnapScholar",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    div.stButton > button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    div.stButton > button:hover {
        background-color: #FF2B2B;
        border-color: #FF2B2B;
    }
    </style>
""", unsafe_allow_html=True)

def render_study_guide(file_path_str: str):
    """
    Parses the generated Markdown file and renders it in Streamlit.
    Detects local image links e.g. ![Caption](image.jpg) and displays
    them using st.image() to ensure they load correctly.
    """
    path = Path(file_path_str)
    if not path.exists():
        st.error(f"Study guide file not found at: {path}")
        return

    # Base directory for images (same folder as the md file)
    base_dir = path.parent
    
    # Read content
    content = path.read_text(encoding="utf-8")

    # Regex to find markdown images: ![AltText](ImageURL)
    # This splits the text into: [Pre-text, AltText, ImageURL, Post-text...]
    image_pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')
    
    last_idx = 0
    
    for match in image_pattern.finditer(content):
        # 1. Render text occurring before the image
        pre_text = content[last_idx:match.start()]
        if pre_text.strip():
            st.markdown(pre_text)
            
        # 2. Render the image
        alt_text = match.group(1)
        img_filename = match.group(2)
        
        img_path = base_dir / img_filename
        
        if img_path.exists():
            st.image(str(img_path), caption=alt_text, use_container_width=True)
        else:
            st.warning(f"Image not found: {img_filename}")
            
        last_idx = match.end()

    # 3. Render remaining text
    if last_idx < len(content):
        st.markdown(content[last_idx:])


def get_step_message(step_name: str) -> str:
    """Maps internal node names to user-friendly UI messages."""
    steps = {
        "init_state": "Starting initialization...",
        "fetch_transcript": "📥 Fetching video transcript...",
        "summarize": "🧠 Generating AI summary and narrative...",
        "extract_topics": "🔍 Identifying key visual concepts...",
        "select_timestamps": "⏱️ Locating best moments in video...",
        "validate_frames": "👁️ Validating image quality (CV check)...",
        "extract_screenshots": "📸 Extracting high-res screenshots...",
        "assemble_document": "📝 Assembling final study guide...",
        "plan_screenshots": "🗺️ Planning screenshots..."
    }
    return steps.get(step_name, f"Processing: {step_name}...")


def main():
    # --- Sidebar ---
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/youtube-play.png", width=60)
        st.title("SnapScholar")
        st.caption("Turn Videos into Visual Study Guides")
        
        st.markdown("---")
        st.markdown("### How it works")
        st.markdown("""
        1. **Paste** a YouTube URL.
        2. **Process** the video.
        3. **Read** the visual summary.
        """)
        
        st.markdown("---")
        st.markdown("### Settings")
        # You could expose settings here if needed
        st.info(f"Model: {settings.MODEL_NAME}")
        if settings.USE_TOPIC_BASED_SELECTION:
            st.success("✨ Enhanced Visual Selection Active")

    # --- Main Content ---
    st.title("🎓 Video to Study Guide")
    st.markdown("Generate a comprehensive summary with context-aware screenshots automatically.")

    # Input area
    url = st.text_input("Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")

    if st.button("Generate Guide", type="primary"):
        if not url:
            st.warning("Please enter a valid YouTube URL.")
            return

        # Initialize Session State for results
        st.session_state.result_state = None
        
        # Container for results
        result_container = st.container()
        
        try:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.status("Running SnapScholar Pipeline...", expanded=True) as status:
                
                final_state = None
                
                # Stream the graph execution
                step_count = 0
                total_estimated_steps = 8
                
                for state in run_snapscholar_stream(url):
                    final_state = state
                    current_step = state.get("current_step", "")
                    
                    # Update status
                    msg = get_step_message(current_step)
                    status.write(msg)
                    
                    # Update progress bar
                    step_count += 1
                    progress = min(step_count / total_estimated_steps, 0.95)
                    progress_bar.progress(progress)
                
                # Completion
                progress_bar.progress(1.0)
                status.update(label="✅ Processing Complete!", state="complete", expanded=False)
                
            # Handle Errors or Success
            if final_state and final_state.get("errors"):
                st.error("Errors occurred during processing:")
                for err in final_state["errors"]:
                    st.error(f"- {err}")
            
            elif final_state and final_state.get("document_link"):
                st.session_state.result_state = final_state
                st.balloons()

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

    # --- Display Results (Persistent) ---
    if "result_state" in st.session_state and st.session_state.result_state:
        state = st.session_state.result_state
        doc_path = state.get("document_link")
        
        st.divider()
        st.subheader("📚 Generated Study Guide")
        
        # Download Button
        with open(doc_path, "rb") as file:
            btn = st.download_button(
                label="📥 Download Markdown File",
                data=file,
                file_name="study_guide.md",
                mime="text/markdown"
            )

        # Render content within a styled container
        with st.container(border=True):
            render_study_guide(doc_path)

if __name__ == "__main__":
    main()