# 🎓 SnapScholar

**AI-powered visual study guides from educational YouTube videos**

SnapScholar turns educational YouTube videos into **structured study guides** with **clear summaries and relevant screenshots**, making it easier to review, remember, and teach what you learned.

🔗 **Live App:** https://snapscholar.streamlit.app/

> Applied Language Models – Group Project  
> Google & Reichman Tech School (Dec 2025)  
> **Team:** Haya Salameh & Amal Zubidat

---

## Why SnapScholar?

**The Problem:**
- Watching videos alone leads to **low retention** - you forget most of what you watch
- Visual explanations improve learning, but **finding the right moments is hard**

**Our Solution:**
SnapScholar **automates the full learning-to-notes pipeline** - from any educational YouTube video to a comprehensive study guide in under 60 seconds.

## Key Features

- **AI-Powered Summaries** – Gemini 2.0 Flash generates structured, topic-based content organized into 5-8 logical sections
- **Intelligent Screenshot Selection** – Automatically extracts the most relevant visual frames based on transcript analysis and topic alignment
- **Agentic AI Workflow** – Built with LangGraph orchestrating 8 specialized processing nodes for robust end-to-end automation
- **Computer Vision Validation** – OpenCV ensures screenshot quality (brightness, blur detection, visual clarity)
- **Fast & Reliable** – Average processing time of 45-60 seconds per video
- **Professional Output** – Download ready-to-use DOCX study guides with timestamps linking back to original video moments

---
## How to Use

### Online (No Installation Required)

1. Open: **https://snapscholar.streamlit.app/**
2. Paste a YouTube link
3. Click **"Generate Study Guide"**
4. Download your DOCX file

That's it! No account needed.

### Run Locally

**Requirements:**
- Python **3.11+**
- Google API key ([get free key](https://makersuite.google.com/app/apikey))
- `ffmpeg` 

**Setup:**

1. **Clone the repository**
```bash
git clone https://github.com/h-a-y-a-s/snapscholar.git
cd snapscholar
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Add your API key**

Create `.env` file:
```
GOOGLE_API_KEY=your_api_key_here
```

5. **Run the app**
```bash
streamlit run app.py
```

6. **Open browser** to `http://localhost:8501`

---
## Best Results With

**Ideal videos:**
- Educational lectures with slides or diagrams
- Tutorials with step-by-step visuals
- 10-30 minute length (optimal)
- Videos with captions/subtitles

**Avoid:**
- Videos without captions
- Purely conversational content (podcasts)
- Music videos or entertainment content
---

## How It Works

**Simple 6-step process:**

1. **User pastes a YouTube link** into the web app
2. **Transcript is extracted** with timestamps from YouTube
3. **AI generates a structured summary** organized by topics (Gemini 2.0)
4. **Key visual moments are selected** - one screenshot per topic
5. **Screenshots are validated** using computer vision (brightness, contrast, content)
6. **Study guide is assembled** - text + visuals combined into downloadable DOCX

**Processing time:** 45-60 seconds | **Success rate:** 95%


## Powered by LangGraph (Agentic AI Workflow)

SnapScholar uses **8 specialized AI agents** working in sequence - each handling one specific task:

```
        ┌─────────────────────┐
        │   init_state        │  Validate input, extract video ID
        └──────────┬──────────┘
                   ↓
        ┌─────────────────────┐
        │  fetch_transcript   │  Get captions + timestamps (cached)
        └──────────┬──────────┘
                   ↓
        ┌─────────────────────┐
        │    summarize        │  Gemini creates structured summary
        └──────────┬──────────┘
                   ↓
        ┌─────────────────────┐
        │  extract_topics     │  Parse section titles
        └──────────┬──────────┘
                   ↓
        ┌─────────────────────┐
        │ select_timestamps   │  Find best visual per topic
        └──────────┬──────────┘
                   ↓
        ┌─────────────────────┐
        │  validate_frames    │  OpenCV quality checks
        └──────────┬──────────┘
                   ↓
        ┌─────────────────────┐
        │ extract_screenshots │  Save high-quality images
        └──────────┬──────────┘
                   ↓
        ┌─────────────────────┐
        │ assemble_document   │  Build final study guide
        └─────────────────────┘
```

**Each agent:**
- Reads the current state
- Performs **one responsibility**
- Updates state and passes forward

**Why LangGraph?**
- Clear, reproducible pipeline
- Easy to debug and extend
- Streaming progress updates (used by UI)
- Separation between reasoning, tools, and validation

**Core Stack:**
- **LangGraph** - Agentic workflow orchestration
- **Google Gemini 2.0 Flash** - AI summarization and reasoning
- **Streamlit** - User-friendly web interface
- **OpenCV** - Computer vision for screenshot validation
- **YouTube Transcript API** - Caption extraction
- **yt-dlp** - Video processing

---

**Haya Salameh** & **Amal Zubidat**

Applied Language Models Course  
Google & Reichman Tech School  
December 2025
