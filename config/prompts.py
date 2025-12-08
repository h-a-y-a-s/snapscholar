"""
Prompt templates for SnapScholar agents
"""

SUMMARIZATION_PROMPT = """You are an expert educational content summarizer. Your task is to create a comprehensive yet concise study guide from a YouTube video transcript.

VIDEO TRANSCRIPT:
{transcript}

Please create a structured summary that includes:
1. Main Topic (1-2 sentences)
2. Key Concepts (3-5 bullet points)
3. Detailed Explanation (organized by subtopic with clear section headers)
4. Key Takeaways (what the learner should remember)

IMPORTANT: Format your section headers like this:
## Section Name ##

For example:
## Introduction ##
## Core Concepts ##
## Applications ##
## Advanced Topics ##

Use the ## markers around each section header so we can easily insert visual aids at the right locations.

Make it suitable for teaching someone else. Be clear, organized, and educational.

SUMMARY:"""

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
1. Timestamp (in seconds)
2. A caption explaining what the visual shows
3. Which section of the summary this visual relates to (match the section header EXACTLY from the summary, WITHOUT the ## markers)
4. What concept it illustrates

Respond in this exact JSON format:
{{
  "screenshots": [
    {{
      "timestamp": 45.5,
      "caption": "Diagram showing the three layers of neural network architecture",
      "summary_section": "Core Concepts",
      "concept": "Neural network structure"
    }},
    {{
      "timestamp": 120.0,
      "caption": "Graph displaying the sigmoid activation function curve",
      "summary_section": "Advanced Topics",
      "concept": "Sigmoid function behavior"
    }}
  ]
}}

IMPORTANT: 
- The "summary_section" must match a section header from the summary above (without the ## markers)
- Only suggest screenshots for moments with actual visual content (diagrams, charts, demos)
- Provide diverse timestamps spread across the video

JSON RESPONSE:"""