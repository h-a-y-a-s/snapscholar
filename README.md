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
- Manual note-taking and screenshots are **time-consuming** (as long as the video itself!)
- Visual explanations improve learning, but **finding the right moments is hard**

**Our Solution:**
SnapScholar **automates the full learning-to-notes pipeline** - from any educational YouTube video to a comprehensive study guide in under 60 seconds.

---

## Main Features

- **AI-generated structured summaries** - Gemini creates organized, topic-based content
- **Smart screenshot selection** - Automatically finds the most relevant visual moments based on video transcript
- **Agentic workflow** - 8 specialized AI agents working in sequence (LangGraph)
- **Computer vision validation** - Validate screenshots 
- **Fast processing** - 45-60 seconds average per video
- 📥 **Downloadable format** - Professional DOCX study guides

---

## What You Get

Your study guide includes:
- **5-8 organized sections** with AI-generated explanations
- **Screenshots** aligned with each topic
- **Timestamp references** linking back to the original video
- **Downloadable DOCX** - ready to review, print, or share

**Example Output:**
```
Study Guide: "Supply and Demand Economics"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Introduction to Economic Principles
[Screenshot: Market basics diagram - t=45s]

Supply and demand are fundamental concepts that determine 
market prices...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Market Equilibrium
[Screenshot: Supply/demand curves - t=312s]

Equilibrium occurs when quantity supplied equals quantity 
demanded...
```

---

## How to Use

### Online (No Installation Required)

1. Open: **https://snapscholar.streamlit.app/**
2. Paste a YouTube link
3. Click **"Generate Study Guide"**
4. Download your DOCX file

That's it! No account needed.

---

### Run Locally

**Requirements:**
- Python **3.11+**
- Google API key ([get free key](https://makersuite.google.com/app/apikey))
- `ffmpeg` installed (recommended)

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

---

## Technologies

**Core Stack:**
- **LangGraph** - Agentic workflow orchestration
- **Google Gemini 2.0 Flash** - AI summarization and reasoning
- **Streamlit** - User-friendly web interface
- **OpenCV** - Computer vision for screenshot validation
- **YouTube Transcript API** - Caption extraction
- **yt-dlp** - Video processing

**Key Innovation:**
Topic-based visual selection (95% accuracy) vs traditional transcript-based approach (60% accuracy)

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

## Team

**Haya Salameh** & **Amal Zubidat**

Applied Language Models Course  
Google & Reichman Tech School  
December 2025

---

**[Try SnapScholar Now](https://snapscholar.streamlit.app/)**
