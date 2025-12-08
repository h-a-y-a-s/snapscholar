"""
Agent state definition for SnapScholar workflow
"""
from typing import TypedDict, List, Dict, Optional


class SnapScholarState(TypedDict, total=False):
    """
    State that flows through the agent workflow.
    
    This contains all data needed by agents:
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
    
    # Screenshot planning
    # e.g. [{timestamp, caption, summary_section, concept}]
    screenshot_plan: Optional[List[Dict]]
    
    # Extracted screenshots
    # e.g. [{timestamp, path}]
    screenshots: Optional[List[Dict]]
    video_path: Optional[str]
    video_dir: Optional[str]
    
    # Final output (e.g. Google Doc/Slides link)
    document_link: Optional[str]
    
    # Error handling / tracking
    errors: List[str]
    current_step: Optional[str]
