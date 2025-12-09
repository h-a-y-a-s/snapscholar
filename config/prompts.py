"""
Prompt templates for SnapScholar agents
"""

from config.settings import settings


SUMMARIZATION_PROMPT = """You are an expert educational content summarizer. Your task is to create a study guide with EXACTLY 5-8 sections.

VIDEO TRANSCRIPT:
{transcript}

CRITICAL REQUIREMENTS:
1. Create EXACTLY 5-8 sections (not 1, not 2, MUST be 5-8)
2. EVERY section header MUST be a question starting with: What, How, Why, When, or Where
3. Format section headers EXACTLY as: ## Question Here? ##
4. Each section: 100-250 words

SECTION HEADER FORMAT (copy this style):
## What Is [Main Concept]? ##
## How Does [Process] Work? ##
## Why Is [Aspect] Important? ##
## What Are The Key [Components]? ##
## How Do [Parts] Interact? ##

EXAMPLE OUTPUT STRUCTURE:
## What Are Transformers? ##
[100-250 words explaining what transformers are...]

## How Do Transformers Process Text? ##
[100-250 words explaining the process...]

## Why Are Attention Mechanisms Important? ##
[100-250 words explaining importance...]

## What Are The Key Components? ##
[100-250 words describing components...]

## How Does Training Work? ##
[100-250 words about training...]

FORBIDDEN:
❌ Generic headers: "Brief Overview", "Introduction", "Main Topic"
❌ Non-question headers: "Core Concepts", "Background"  
❌ Only 1-2 sections (MUST create 5-8)

YOUR OUTPUT STRUCTURE:
1. Write 2-3 sentence overview
2. Create 5-8 sections with ## Question? ## headers
3. Add ## Key Takeaways ## at the end (3-5 bullets)

BEGIN STUDY GUIDE:"""


TOPIC_EXTRACTION_PROMPT = """Extract all the section headers from this study guide.

SUMMARY:
{summary}

Find all section headers formatted as ## Section Name ##

These sections represent key questions answered in the content.

Respond with a JSON list of section names (without the ## markers):

{{
  "sections": ["What Is The Core Concept", "How Does The Process Work", "Why Is This Approach Used"]
}}

Rules:
- Extract ALL sections with ## markers
- Remove the ## markers from names
- Keep exact section names as written
- Preserve order from the summary
- Skip "Key Takeaways" or "Summary" sections

JSON RESPONSE:"""


TOPIC_TIMESTAMP_SELECTION_BATCH_PROMPT = """Find the best visual moments for these sections from the video.

SECTIONS TO ILLUSTRATE:
{topics}

VIDEO TRANSCRIPT WITH TIMESTAMPS:
{transcript_with_timestamps}

VIDEO DURATION: The video is approximately {{duration}} seconds long.

For EACH section, find ONE timestamp where visual content is shown.

CRITICAL RULES:
1. Timestamps MUST be numbers in seconds (e.g., 125.5, 340.0, 23.0)
2. DO NOT use time format (e.g., 00:23, 01:38)
3. Timestamps MUST be between 10s and {{duration}}-10s (avoid first/last 10 seconds)
4. Prefer timestamps in the middle range of when the topic is discussed
5. DO NOT select timestamps near the very end of the video

Respond with valid JSON only (no markdown, no code blocks):

{{"screenshots": [
  {{"section": "Section Name", "timestamp": 125.5, "caption": "Brief description", "reason": "Why this moment"}},
  {{"section": "Another Section", "timestamp": 340.0, "caption": "Brief description", "reason": "Why this moment"}}
]}}

OTHER RULES:
- Return ONLY the JSON object
- No markdown code blocks (no ```)
- No extra text before or after
- Use double quotes for all strings
- Ensure valid JSON syntax
- One timestamp per section
- Prefer moments with phrases like "as you can see", "look at", "here we have"

JSON RESPONSE:"""

TOPIC_TIMESTAMP_SELECTION_PROMPT = """Find the best visual moment for this section.

SECTION: {topic}

VIDEO TRANSCRIPT WITH TIMESTAMPS:
{transcript_with_timestamps}

Find the ONE best timestamp (in seconds) where visual content for this section is most likely shown.

Look for moments that:
- Explicitly discuss this topic
- Mention visual elements (diagrams, charts, visuals, examples)
- Use phrases like "as you can see", "look at this", "here we have"
- Show relevant visual content on screen

Respond with JSON:
{{
  "timestamp": 150.5,
  "reason": "Transcript indicates visual explanation at this moment",
  "caption": "Visual representation of the concept"
}}

CRITICAL:
- Timestamp MUST be a number in seconds (e.g., 150.5, NOT 02:30)
- Choose the SINGLE best moment for this section
- Prefer moments with explicit visual references
- If unclear, choose where the topic is discussed most

JSON RESPONSE:"""


# Original fallback prompt
SCREENSHOT_PLANNING_PROMPT = """You are an expert at identifying the most important visual moments in educational videos.

VIDEO TRANSCRIPT WITH TIMESTAMPS:
{transcript_with_timestamps}

SUMMARY:
{summary}

Your task: Identify {max_screenshots} timestamps where screenshots would be most valuable for understanding the content.

Choose moments that show:
- Diagrams, charts, or visual explanations
- Key definitions or formulas written on screen
- Important demonstrations or examples
- Critical visual concepts

For each screenshot, provide:
1. Timestamp (in seconds as a number, e.g., 45.5, NOT 00:45)
2. A caption explaining what the visual shows
3. Which section of the summary this visual relates to (match the section header EXACTLY from the summary, WITHOUT the ## markers)
4. What concept it illustrates

Respond in this exact JSON format:
{{
  "screenshots": [
    {{
      "timestamp": 45.5,
      "caption": "Visual diagram showing the main concept",
      "summary_section": "How Does It Work",
      "concept": "Process explanation"
    }}
  ]
}}

IMPORTANT: 
- Timestamps must be numbers in seconds (45.5, NOT 00:45)
- The "summary_section" must match a section header from the summary above (without the ## markers)
- Only suggest screenshots for moments with actual visual content (diagrams, charts, demos)
- Provide diverse timestamps spread across the video

JSON RESPONSE:"""
