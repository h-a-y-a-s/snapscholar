"""
Configuration settings for SnapScholar
"""
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Settings:
    """Application settings"""
    
    # API Keys - Initialize as None to force user input
    GOOGLE_API_KEY = None
    
    # Google Drive
    GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    TEMP_DIR = BASE_DIR / "data" / "temp"
    
    # Model Settings
    MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    TEMPERATURE = 0.7
    MAX_TOKENS = 4000

    # Whisper Settings
    WHISPER_MODEL = "base"
    USE_WHISPER_FALLBACK = True
    
    # Screenshot Settings
    MAX_SCREENSHOTS = 10
    MIN_SCREENSHOTS = 3
    SCREENSHOT_QUALITY = 95
    
    # Topic-based visual selection
    USE_TOPIC_BASED_SELECTION = True
    ONE_VISUAL_PER_TOPIC = True
    
    # Section selection - UPDATED VALUES
    MIN_SECTIONS_FOR_VISUALS = 3
    MAX_SECTIONS_FOR_VISUALS = 10
    
    # CV frame validation - Optimized thresholds
    ENABLE_CV_VALIDATION = True
    
    # Face detection: Reject if face > 30% of frame (talking head)
    FACE_AREA_THRESHOLD = 0.30
    
    # Color variance: Reject if < 10.0 (blank/uniform slides)
    COLOR_VARIANCE_THRESHOLD = 10.0
    
    # Edge density: Reject if < 2% (no visual information)
    EDGE_DENSITY_THRESHOLD = 0.02
    
    # Text density: Prefer frames with > 1% text regions
    TEXT_DENSITY_THRESHOLD = 0.01
    
    # Black pixel ratio: Reject if > 70% black pixels
    BLACK_PIXEL_THRESHOLD = 0.70
    
    # Frame search - expanded range
    FRAME_SEARCH_OFFSETS = [-10, -7, -5, -3, -1, 1, 3, 5, 7, 10, 15]
    MAX_FRAME_VALIDATION_ATTEMPTS = 11
    
    # Document Settings
    DOC_TITLE_PREFIX = "SnapScholar Study Guide"
    
    def __init__(self):
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    def validate(self):
        # We will validate the API key within the app logic itself
        # if not self.GOOGLE_API_KEY:
        #     raise ValueError("GOOGLE_API_KEY not found in .env file")
        return True

settings = Settings()