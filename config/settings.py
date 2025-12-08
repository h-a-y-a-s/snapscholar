"""
Configuration settings for SnapScholar
"""
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Settings:
    """Application settings"""
    
    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Google Drive
    GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    TEMP_DIR = BASE_DIR / "data" / "temp"
    
    # Model Settings - USING GEMINI
    MODEL_NAME = "gemini-1.5-flash"  # Fast and FREE
    # MODEL_NAME = "gemini-1.5-pro"  # More powerful (also FREE!)
    TEMPERATURE = 0.7
    MAX_TOKENS = 4000
    
    # Screenshot Settings
    MAX_SCREENSHOTS = 8
    MIN_SCREENSHOTS = 3
    SCREENSHOT_QUALITY = 95
    
    # Document Settings
    DOC_TITLE_PREFIX = "SnapScholar Study Guide"
    
    def __init__(self):
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    def validate(self):
        if not self.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        return True

settings = Settings()