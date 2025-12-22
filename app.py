import streamlit as st
import re
from pathlib import Path
import time
import pypandoc
import os
import random

# Internal imports from your existing codebase
from graph.graph import run_snapscholar_stream
from config.settings import settings
from config.prompts import SUMMARIZATION_PROMPT, TOPIC_EXTRACTION_PROMPT

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
        height: 3.6rem;
    }
    div.stButton > button:hover {
        background-color: #FF2B2B;
        border-color: #FF2B2B;
    }
    </style>
""", unsafe_allow_html=True)

def convert_md_to_docx(md_file_path: str) -> str:
    """
    Converts a Markdown file to a DOCX file using Pandoc, ensuring images are embedded.
    """
    md_file_path = os.path.abspath(md_file_path)
    docx_file_path = os.path.splitext(md_file_path)[0] + ".docx"
    
    original_cwd = os.getcwd()
    md_dir = os.path.dirname(md_file_path)
    md_filename = os.path.basename(md_file_path)

    try:
        os.chdir(md_dir)
        pypandoc.convert_file(md_filename, 'docx', outputfile=docx_file_path)
        return docx_file_path
    except Exception as e:
        st.error(f"Error converting to DOCX: {e}")
        return None
    finally:
        os.chdir(original_cwd)

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
        
        # --- API Key Input ---
        st.markdown("---")

        # Check if the API key is already loaded and valid
        api_key_loaded = "google_api_key" in st.session_state and st.session_state.google_api_key
        if api_key_loaded:
            st.success("✅ Google API Key Loaded")
            # Update env and settings on every run to be safe
            os.environ["GOOGLE_API_KEY"] = st.session_state.google_api_key
            settings.GOOGLE_API_KEY = st.session_state.google_api_key

        # Expander for API key settings
        with st.expander("🔑 API Key Configuration", expanded=not api_key_loaded):
            st.info(
                "Get your Google API key from [Google AI Studio](https://aistudio.google.com/). "
                "Click **'Get API Key'** in the top right and create a new key."
            )
            
            api_key = st.text_input(
                "Enter your Google API Key", 
                type="password", 
                key="api_key_input",
                placeholder="Enter your key and press 'Save Key'",
                label_visibility="collapsed"
            )

            if st.button("Save Key"):
                if api_key:
                    st.session_state.google_api_key = api_key
                    os.environ["GOOGLE_API_KEY"] = api_key
                    settings.GOOGLE_API_KEY = api_key
                    st.success("API Key saved!")
                    time.sleep(0.5)  # Short delay for UI update
                    st.rerun()
                else:
                    st.warning("Please enter a valid API key.")
        
        st.markdown("---")
        if settings.USE_TOPIC_BASED_SELECTION:
            st.success("✨ Enhanced Visual Selection Active")

    # --- Main Content ---
    st.title("🎓 Video to Study Guide")
    st.markdown("Generate a comprehensive summary with context-aware screenshots automatically.")

    with st.expander("ℹ️ About & How to Use", expanded=True):
        st.markdown("""
        **SnapScholar is designed to help you learn faster and more effectively from educational videos.**

        Simply paste a YouTube link, and the app will generate a comprehensive study guide complete with:
        - A full transcript and AI-powered summary.
        - Key topics and concepts explained.
        - Context-aware screenshots placed directly into the narrative to visualize important moments.

        You can read the guide directly in the app or download it as a DOCX file for offline use.

        **How to Use:**
        1. **Configure API Key:** Use the 'API Key Configuration' section in the sidebar to add your Google API key. You only need to do this once.
        2. **Paste YouTube URL:** Add the link to the video you want to summarize.
        3. **Click 'Generate Guide':** Let SnapScholar work its magic.
        4. **Review & Download:** Your visual study guide will appear, ready to be read or downloaded as a DOCX file.
        """)

    st.subheader("✨ Example Video")
    st.video("https://youtu.be/FgakZw6K1QQ?si=M4uKe5wLHTUlYbL2")

    # Input area
    url = st.text_input("Enter YouTube Video URL", value="https://youtu.be/FgakZw6K1QQ?si=M4uKe5wLHTUlYbL2", placeholder="https://www.youtube.com/watch?v=...")
    with st.popover("⚙️ Settings"):
        # Model Selection
        available_models = (
            "gemini-2.0-flash",
            "gemini-1.5-flash-latest", 
            "gemini-1.5-pro-latest", 
            "gemini-pro"
        )
        
        try:
            default_index = available_models.index(settings.MODEL_NAME)
        except ValueError:
            default_index = 0

        model_name = st.selectbox(
            "Choose a model:",
            available_models,
            index=default_index,
            help="Select the AI model to use for generation."
        )

        # Prompt Configuration
        summarization_prompt = st.text_area("Summarization Prompt", value=SUMMARIZATION_PROMPT, height=200)
        topic_extraction_prompt = st.text_area("Topic Extraction Prompt", value=TOPIC_EXTRACTION_PROMPT, height=200)

    # Disable button if API key is not provided
    api_key_present = hasattr(settings, 'GOOGLE_API_KEY') and settings.GOOGLE_API_KEY

    if not api_key_present:
        st.warning("Please enter and save your Google API Key in the sidebar to proceed.")

    if st.button("Generate Guide", type="primary", use_container_width=True, disabled=not api_key_present):
        if not url:
            st.warning("Please enter a valid YouTube URL.")
            return

        st.session_state.result_state = None
        
        try:
            progress_bar = st.progress(0)
            
            with st.status("Running SnapScholar Pipeline...", expanded=True) as status:
                final_state = None
                step_count = 0
                total_estimated_steps = 8
                
                for state in run_snapscholar_stream(url, model_name, summarization_prompt, topic_extraction_prompt):
                    final_state = state
                    current_step = state.get("current_step", "")
                    
                    msg = get_step_message(current_step)
                    status.write(msg)
                    
                    step_count += 1
                    progress = min(step_count / total_estimated_steps, 0.95)
                    progress_bar.progress(progress)
                
                progress_bar.progress(1.0)
                status.update(label="✅ Processing Complete!", state="complete", expanded=False)
                
            if final_state and final_state.get("errors"):
                for err in final_state["errors"]:
                    st.error(f"- {err}")
            
            elif final_state and final_state.get("document_link"):
                st.session_state.result_state = final_state
                st.balloons()

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

    # --- Display Results ---
    if "result_state" in st.session_state and st.session_state.result_state:
        state = st.session_state.result_state
        doc_path = state.get("document_link")
        
        st.divider()
        st.subheader("📚 Generated Study Guide")
        
        col1, col2 = st.columns(2)
        with col1:
            docx_path = convert_md_to_docx(doc_path)
            if docx_path and os.path.exists(docx_path):
                with open(docx_path, "rb") as file:
                    st.download_button(
                        label="📥 Download DOCX File",
                        data=file,
                        file_name="study_guide.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True
                    )
        with col2:
            if st.button("🎉 Celebrate!", use_container_width=True):
                random.choice([st.balloons, st.snow, lambda: st.success("🎉🎉🎉")])()

        with st.container(border=True):
            render_study_guide(doc_path)

if __name__ == "__main__":
    main()