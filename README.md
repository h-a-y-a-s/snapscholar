# 🎓 SnapScholar

**AI-powered visual study guides from educational YouTube videos**

SnapScholar turns educational YouTube videos into **structured study guides** with **clear summaries and relevant screenshots**, making it easier to review, remember, and teach what you learned.

🔗 **Live App:** https://snapscholar.streamlit.app/

> 📚 Applied Language Models – Group Project  
> Google & Reichman Tech School (Dec 2025)  
> **Team:** Haya Salameh & Amal Zubidat

---

## ✨ Main Features

- 📄 AI-generated structured summaries  
- 🖼️ Automatic screenshot selection aligned with topics  
- 🤖 Agentic workflow using LangGraph  
- 🧠 Gemini-powered reasoning  
- 👁️ Computer-vision validation of screenshots  
- ⚡ Fast end-to-end processing  
- 📥 Downloadable study guide  

---

## 🧭 How It Works (High Level)

1. User pastes a **YouTube link**
2. Transcript is extracted with timestamps
3. AI generates a **topic-structured summary**
4. Key visual moments are selected per topic
5. Screenshots are validated and extracted
6. Text and visuals are combined into a study guide

---

## 🏗️ System Architecture

```
┌──────────────────────┐
│       User           │
│  (Streamlit UI)      │
└─────────┬────────────┘
          │ YouTube URL
          ▼
┌──────────────────────┐
│  LangGraph Engine    │
│ (Agent State Graph)  │
└─────────┬────────────┘
          │
          ▼
┌───────────────────────────────┐
│        init_state             │
│    - validate input           │
│    - extract video_id         │
└─────────┬─────────────────────┘
          ▼
┌───────────────────────────────┐
│     fetch_transcript          │
│  - YouTube transcript API     │
│  - timestamps + caching       │
└─────────┬─────────────────────┘
          ▼
┌───────────────────────────────┐
│        summarize              │
│      - Gemini LLM             │
│   - structured sections       │
└─────────┬─────────────────────┘
          ▼
┌───────────────────────────────┐
│     extract_topics            │
│    - section titles           │
│    - learning units           │
└─────────┬─────────────────────┘
          ▼
┌───────────────────────────────┐
│    select_timestamps          │
│  - best visual per topic      │
│   - semantic matching         │
└─────────┬─────────────────────┘
          ▼
┌───────────────────────────────┐
│     validate_frames           │
│     - OpenCV checks           │
│  - brightness / content       │
└─────────┬─────────────────────┘
          ▼
┌───────────────────────────────┐
│   extract_screenshots         │
│    - yt-dlp + OpenCV          │
└─────────┬─────────────────────┘
          ▼
┌───────────────────────────────┐
│    assemble_document          │
│    - text + visuals           │
│   - final study guide         │
└───────────────────────────────┘
```

---

## 🤖 LangGraph Agent Workflow

SnapScholar is implemented as a **LangGraph state machine**, where each step is a dedicated agent (node) operating on a shared state.

### Why LangGraph?

- Explicit and reproducible pipeline  
- Clear separation between reasoning, tools, and validation  
- Streaming intermediate states (used by the UI)  
- Easy to extend with new agents (e.g. quizzes, slides, RAG)

---

### Workflow Structure

```
init_state
    ↓
fetch_transcript
    ↓
summarize
    ↓
extract_topics
    ↓
select_timestamps
    ↓
validate_frames
    ↓
extract_screenshots
    ↓
assemble_document
```

Each node:
- Reads the current `SnapScholarState`
- Performs **one responsibility**
- Updates the state and passes it forward

---

### Agent Responsibilities

- **init_state** – validates input and extracts the video ID  
- **fetch_transcript** – fetches transcript + timestamps (with caching)  
- **summarize** – generates a structured summary using Gemini  
- **extract_topics** – extracts section titles from the summary  
- **select_timestamps** – finds the best visual moment for each topic  
- **validate_frames** – rejects low-quality frames using computer vision  
- **extract_screenshots** – extracts screenshots from the video  
- **assemble_document** – builds the final study guide with text + images  

---

## 🚀 How to Use (Online)

1. Open: https://snapscholar.streamlit.app/
2. Paste a YouTube link
3. Click **Generate**
4. Download your study guide

---

## 💻 Run Locally

### Requirements
- Python **3.11+**
- Google API key (Gemini)
- `ffmpeg` installed (recommended)

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/snapscholar.git
cd snapscholar
```

2. **Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
venv\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set environment variables**

Create a `.env` file:
```
GOOGLE_API_KEY=your_api_key_here
```

5. **Run the app**
```bash
streamlit run app.py
```

6. Open in your browser:
```
http://localhost:8501
```

---

## 🎯 Why SnapScholar?

- Watching videos alone leads to **low retention**
- Manual note-taking and screenshots are **slow**
- Visual explanations significantly improve learning

**SnapScholar automates the full learning-to-notes pipeline.**

---

## 🧠 Technologies

- LangGraph  
- Google Gemini  
- Streamlit  
- OpenCV  
- YouTube Transcript API  

---

## 👩‍💻 Team

**Haya Salameh**  
**Amal Zubidat**
