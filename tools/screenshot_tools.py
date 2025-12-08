"""
Screenshot extraction tools for video frames
Organizes files by video ID: data/temp/{video_id}/video.mp4 and screenshots
"""
import cv2
import yt_dlp
from pathlib import Path
from typing import List, Dict, Optional
import os


def download_video(video_id: str, output_dir: Optional[Path] = None) -> Dict:
    """
    Download YouTube video for screenshot extraction
    Creates a directory for each video: data/temp/{video_id}/
    
    Args:
        video_id: YouTube video ID
        output_dir: Base directory (default: data/temp/)
        
    Returns:
        Dict with download info:
        {
            'success': bool,
            'video_path': Path or None,
            'video_dir': Path,  # Directory for this video
            'error': str or None
        }
    """
    if output_dir is None:
        output_dir = Path("data/temp")
    
    # Create video-specific directory
    video_dir = output_dir / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    
    video_path = video_dir / "video.mp4"
    
    # If video already exists, skip download
    if video_path.exists():
        print(f"  ✅ Video already exists: {video_path}")
        return {
            'success': True,
            'video_path': video_path,
            'video_dir': video_dir,
            'error': None
        }
    
    try:
        ydl_opts = {
            'format': 'best[height<=720]',  # 720p max for speed
            'outtmpl': str(video_dir / 'video'),  # Save as 'video' (no extension)
            'quiet': True,
            'no_warnings': True,
        }
        
        print(f"  📥 Downloading video {video_id}...")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        
        # Find downloaded file (might have different extension or none)
        possible_files = list(video_dir.glob("video*"))
        
        if possible_files:
            actual_path = possible_files[0]
            
            # Rename to video.mp4 if needed
            if actual_path != video_path:
                actual_path = actual_path.rename(video_path)
            
            print(f"  ✅ Video downloaded: {video_path}")
            return {
                'success': True,
                'video_path': video_path,
                'video_dir': video_dir,
                'error': None
            }
        else:
            return {
                'success': False,
                'video_path': None,
                'video_dir': video_dir,
                'error': 'Video file not found after download'
            }
            
    except Exception as e:
        return {
            'success': False,
            'video_path': None,
            'video_dir': video_dir,
            'error': f"Download failed: {str(e)}"
        }


def extract_frame(video_path: Path, timestamp: float, output_path: Path) -> Dict:
    """
    Extract a single frame from video at specific timestamp
    
    Args:
        video_path: Path to video file
        timestamp: Time in seconds
        output_path: Where to save the screenshot
        
    Returns:
        Dict with extraction info:
        {
            'success': bool,
            'image_path': Path or None,
            'error': str or None
        }
    """
    try:
        # Open video
        video = cv2.VideoCapture(str(video_path))
        
        if not video.isOpened():
            return {
                'success': False,
                'image_path': None,
                'error': 'Could not open video file'
            }
        
        # Set position to timestamp (in milliseconds)
        video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        
        # Read frame
        success, frame = video.read()
        
        if not success:
            video.release()
            return {
                'success': False,
                'image_path': None,
                'error': f'Could not read frame at {timestamp}s'
            }
        
        # Save frame as JPG
        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        video.release()
        
        return {
            'success': True,
            'image_path': output_path,
            'error': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'image_path': None,
            'error': f"Frame extraction failed: {str(e)}"
        }


def extract_screenshots(video_id: str, timestamps: List[float], 
                       output_dir: Optional[Path] = None) -> Dict:
    """
    Extract multiple screenshots from a video
    Saves everything in data/temp/{video_id}/ directory
    
    Args:
        video_id: YouTube video ID
        timestamps: List of timestamps in seconds
        output_dir: Base directory (default: data/temp/)
        
    Returns:
        Dict with extraction results:
        {
            'success': bool,
            'screenshots': List[Dict],  # List of {timestamp, path}
            'video_path': Path,
            'video_dir': Path,
            'error': str or None
        }
    """
    if output_dir is None:
        output_dir = Path("data/temp")
    
    # Download video (or use existing)
    download_result = download_video(video_id, output_dir)
    
    if not download_result['success']:
        return {
            'success': False,
            'screenshots': [],
            'video_path': None,
            'video_dir': None,
            'error': download_result['error']
        }
    
    video_path = download_result['video_path']
    video_dir = download_result['video_dir']
    
    # Extract screenshots
    screenshots = []
    failed = []
    
    print(f"  📸 Extracting {len(timestamps)} screenshots...")
    
    for timestamp in timestamps:
        # Save screenshot in video directory
        output_path = video_dir / f"screenshot_{timestamp:.1f}.jpg"
        
        # Skip if screenshot already exists
        if output_path.exists():
            screenshots.append({
                'timestamp': timestamp,
                'path': output_path
            })
            print(f"     ✅ Screenshot at {timestamp:.1f}s (already exists)")
            continue
        
        # Extract frame
        result = extract_frame(video_path, timestamp, output_path)
        
        if result['success']:
            screenshots.append({
                'timestamp': timestamp,
                'path': result['image_path']
            })
            print(f"     ✅ Screenshot at {timestamp:.1f}s")
        else:
            failed.append({
                'timestamp': timestamp,
                'error': result['error']
            })
            print(f"     ❌ Failed at {timestamp:.1f}s: {result['error']}")
    
    if screenshots:
        print(f"  ✅ Extracted {len(screenshots)}/{len(timestamps)} screenshots")
        return {
            'success': True,
            'screenshots': screenshots,
            'video_path': video_path,
            'video_dir': video_dir,
            'failed': failed,
            'error': None
        }
    else:
        return {
            'success': False,
            'screenshots': [],
            'video_path': video_path,
            'video_dir': video_dir,
            'failed': failed,
            'error': 'No screenshots extracted successfully'
        }


def cleanup_video(video_path: Path) -> bool:
    """
    Delete video file after processing
    
    Args:
        video_path: Path to video file
        
    Returns:
        bool: True if deleted successfully
    """
    try:
        if video_path.exists():
            video_path.unlink()
            print(f"  🗑️  Cleaned up video: {video_path.name}")
            return True
        return False
    except Exception as e:
        print(f"  ⚠️  Could not delete video: {e}")
        return False


def cleanup_video_directory(video_dir: Path) -> bool:
    """
    Delete entire video directory (video + screenshots)
    
    Args:
        video_dir: Path to video directory
        
    Returns:
        bool: True if deleted successfully
    """
    try:
        if video_dir.exists():
            import shutil
            shutil.rmtree(video_dir)
            print(f"  🗑️  Cleaned up directory: {video_dir.name}")
            return True
        return False
    except Exception as e:
        print(f"  ⚠️  Could not delete directory: {e}")
        return False


def cleanup_screenshots(screenshot_paths: List[Path]) -> int:
    """
    Delete screenshot files
    
    Args:
        screenshot_paths: List of paths to screenshot files
        
    Returns:
        int: Number of files deleted
    """
    deleted = 0
    for path in screenshot_paths:
        try:
            if path.exists():
                path.unlink()
                deleted += 1
        except Exception as e:
            print(f"  ⚠️  Could not delete {path.name}: {e}")
    
    if deleted > 0:
        print(f"  🗑️  Cleaned up {deleted} screenshots")
    
    return deleted


# Testing
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Test with a short video
    test_video_id = "FE-hM1kRK4Y"  # Your working test video
    test_timestamps = [10.0, 30.0, 60.0]  # 10s, 30s, 1min
    
    print("=" * 60)
    print("SnapScholar Screenshot Tools Test")
    print("=" * 60)
    print(f"\nTest Video ID: {test_video_id}")
    print(f"Timestamps: {test_timestamps}\n")
    
    # Extract screenshots
    result = extract_screenshots(test_video_id, test_timestamps)
    
    if result['success']:
        print(f"\n✅ SUCCESS!")
        print(f"   Video directory: {result['video_dir']}")
        print(f"   Video: {result['video_path']}")
        print(f"   Screenshots extracted: {len(result['screenshots'])}")
        
        for screenshot in result['screenshots']:
            print(f"   - {screenshot['timestamp']}s → {screenshot['path'].name}")
        
        print(f"\n📂 All files saved in: {result['video_dir']}")
        
        # Test running again (should skip download)
        print("\n" + "=" * 60)
        print("Testing with existing video (should skip download)...")
        print("=" * 60 + "\n")
        
        result2 = extract_screenshots(test_video_id, test_timestamps)
        
        if result2['success']:
            print(f"\n✅ Reused existing files successfully!")
        
        # Optionally cleanup
        # cleanup_video_directory(result['video_dir'])
    else:
        print(f"\n❌ FAILED")
        print(f"   Error: {result['error']}")