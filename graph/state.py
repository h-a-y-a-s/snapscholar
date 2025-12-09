"""
Agent state definition for SnapScholar workflow
"""
from typing import TypedDict, List, Dict, Optional


class SnapScholarState(TypedDict, total=False):
    """
    State that flows through the agent workflow.
    
    Contains all data needed by agents:
    - Input: YouTube URL/video_id
    - Transcript data
    - AI-generated summary
    - Screenshot plan
    - Extracted screenshots
    - Final output
    """
    
    # Input
    youtube_url: str
    video_id: str
    
    # Transcript data
    transcript_text: Optional[str]
    transcript_segments: Optional[List[Dict]]
    transcript_with_timestamps: Optional[str]
    video_duration: Optional[float]
    
    # AI-generated summary
    summary: Optional[str]
    
    # Topic-based visual selection
    # topics: ["Introduction", "Core Concepts", "Applications"]
    topics: Optional[List[str]]
    
    # topic_timestamps: [{"topic": "Core Concepts", "timestamp": 150.5, "caption": "..."}]
    topic_timestamps: Optional[List[Dict]]
    
    # frame_validation: [{"timestamp": 150.5, "is_valid": True, "validation_details": {...}}]
    frame_validation: Optional[List[Dict]]
    
    # Screenshot planning
    # [{timestamp, caption, summary_section, concept}]
    screenshot_plan: Optional[List[Dict]]
    
    # Extracted screenshots
    # [{timestamp, path, topic}]
    screenshots: Optional[List[Dict]]
    video_path: Optional[str]
    video_dir: Optional[str]
    
    # Final output
    document_link: Optional[str]
    
    # Error handling
    errors: List[str]
    current_step: Optional[str]