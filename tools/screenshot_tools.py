"""
Screenshot extraction tools for video frames with CV validation
Organizes files by video ID: data/temp/{video_id}/video.mp4 and screenshots
"""
import os
import cv2
import yt_dlp
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

# Suppress OpenCV/FFmpeg warnings
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'loglevel;quiet'
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

# Suppress OpenCV logs
try:
    cv2.setLogLevel(0)
except:
    pass


def download_video(video_id: str, output_dir: Optional[Path] = None) -> Dict:
    """
    Download YouTube video for screenshot extraction.
    Creates a directory for each video: data/temp/{video_id}/
    
    Args:
        video_id: YouTube video ID
        output_dir: Base directory (default: data/temp/)
    Returns:
        Dict with download info
    """
    if output_dir is None:
        output_dir = Path("data/temp")
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
            'format': 'best[height<=720]',
            'outtmpl': str(video_dir / 'video'),
            'quiet': True,
            'no_warnings': True,
        }
        
        print(f"  📥 Downloading video {video_id}...")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        
        # Find downloaded file
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
        Dict with extraction info
    """
    try:
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


def calculate_black_pixel_ratio(frame: np.ndarray) -> float:
    """
    Calculate ratio of black pixels in frame
    
    Args:
        frame: OpenCV frame
        
    Returns:
        Ratio of black pixels (0-1)
    """
    try:
        # Pixels are "black" if all RGB values < 20
        black_pixels = np.sum(np.all(frame < 20, axis=2))
        total_pixels = frame.shape[0] * frame.shape[1]
        return black_pixels / total_pixels
    except Exception as e:
        print(f"  ⚠️ Black pixel calculation failed: {e}")
        return 0.0


def validate_frame_quality(frame: np.ndarray, settings=None) -> Dict:
    """
    Validate frame quality using computer vision
    
    Args:
        frame: OpenCV frame (numpy array)
        settings: Settings object with thresholds (optional)
        
    Returns:
        {
            'is_valid': bool,
            'reasons': List[str],
            'metrics': {
                'has_large_face': bool,
                'is_blank': bool,
                'edge_density': float,
                'color_variance': float,
                'black_pixel_ratio': float
            }
        }
    """
    if settings is None:
        from config.settings import settings
    
    reasons = []
    metrics = {}
    
    # Check 1: Face detection (talking head)
    has_large_face = detect_large_face(frame, settings.FACE_AREA_THRESHOLD)
    metrics['has_large_face'] = has_large_face
    if has_large_face:
        reasons.append("Large face detected (talking head)")
    
    # Check 2: Blank slide detection
    is_blank = detect_blank_frame(frame, settings.COLOR_VARIANCE_THRESHOLD)
    metrics['is_blank'] = is_blank
    if is_blank:
        reasons.append("Mostly blank (low color variance)")
    
    # Check 3: Edge density (visual information)
    edge_density = calculate_edge_density(frame)
    metrics['edge_density'] = edge_density
    if edge_density < settings.EDGE_DENSITY_THRESHOLD:
        reasons.append(f"Low visual information (edge density: {edge_density:.3f})")
    
    # Check 4: Color variance
    color_variance = calculate_color_variance(frame)
    metrics['color_variance'] = color_variance
    
    # Check 5: Black pixel ratio
    black_ratio = calculate_black_pixel_ratio(frame)
    metrics['black_pixel_ratio'] = black_ratio
    if black_ratio > settings.BLACK_PIXEL_THRESHOLD:
        reasons.append(f"Mostly black ({black_ratio*100:.1f}% black pixels)")
    
    # Frame is valid if it passes all checks
    is_valid = not (
        has_large_face or 
        is_blank or 
        edge_density < settings.EDGE_DENSITY_THRESHOLD or
        black_ratio > settings.BLACK_PIXEL_THRESHOLD
    )
    
    return {
        'is_valid': is_valid,
        'reasons': reasons if not is_valid else ["Good frame quality"],
        'metrics': metrics
    }


def detect_large_face(frame: np.ndarray, threshold: float = 0.30) -> bool:
    """
    Detect if frame has a large face (talking head)
    
    Args:
        frame: OpenCV frame
        threshold: Face area threshold (% of frame)
        
    Returns:
        True if large face detected
    """
    try:
        # Load Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return False
        
        # Calculate total face area
        frame_area = frame.shape[0] * frame.shape[1]
        face_area = sum(w * h for (x, y, w, h) in faces)
        face_ratio = face_area / frame_area
        
        return face_ratio > threshold
        
    except Exception as e:
        print(f"  ⚠️ Face detection failed: {e}")
        return False


def detect_blank_frame(frame: np.ndarray, threshold: float = 10.0) -> bool:
    """
    Detect if frame is mostly blank (uniform color)
    
    Args:
        frame: OpenCV frame
        threshold: Color variance threshold
        
    Returns:
        True if frame is mostly blank
    """
    try:
        # Calculate standard deviation across color channels
        std_dev = np.std(frame)
        return std_dev < threshold
        
    except Exception as e:
        print(f"  ⚠️ Blank detection failed: {e}")
        return False


def calculate_edge_density(frame: np.ndarray) -> float:
    """
    Calculate edge density (visual information content)
    
    Args:
        frame: OpenCV frame
        
    Returns:
        Edge density (0-1, higher = more visual info)
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate percentage of edge pixels
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        
        return edge_pixels / total_pixels
        
    except Exception as e:
        print(f"  ⚠️ Edge detection failed: {e}")
        return 0.0


def calculate_color_variance(frame: np.ndarray) -> float:
    """
    Calculate color variance across frame
    
    Args:
        frame: OpenCV frame
        
    Returns:
        Standard deviation of pixel values
    """
    try:
        return float(np.std(frame))
    except Exception as e:
        print(f"  ⚠️ Color variance calculation failed: {e}")
        return 0.0


def find_best_nearby_frame(video_path: Path, timestamp: float, 
                           offsets: List[float], settings=None) -> Optional[Dict]:
    """
    Try to find a better quality frame near the target timestamp
    
    Args:
        video_path: Path to video
        timestamp: Target timestamp
        offsets: Time offsets to try (e.g., [-5, -3, 3, 5])
        settings: Settings object
        
    Returns:
        Best frame info or None
    """
    if settings is None:
        from config.settings import settings
    
    video = cv2.VideoCapture(str(video_path))
    
    if not video.isOpened():
        return None
    
    best_frame = None
    best_score = -1
    best_timestamp = timestamp
    
    for offset in offsets:
        try_timestamp = timestamp + offset
        
        # Skip negative timestamps
        if try_timestamp < 0:
            continue
        
        # Set position and read frame
        video.set(cv2.CAP_PROP_POS_MSEC, try_timestamp * 1000)
        success, frame = video.read()
        
        if not success:
            continue
        
        # Validate frame
        validation = validate_frame_quality(frame, settings)
        
        if validation['is_valid']:
            # Calculate score (higher edge density = better)
            score = validation['metrics']['edge_density']
            
            if score > best_score:
                best_score = score
                best_frame = frame
                best_timestamp = try_timestamp
    
    video.release()
    
    if best_frame is not None:
        return {
            'frame': best_frame,
            'timestamp': best_timestamp,
            'score': best_score
        }
    
    return None


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
        Dict with extraction results
    """
    if output_dir is None:
        output_dir = Path("data/temp")
    
    # Download video
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
    """Delete video file after processing"""
    try:
        if video_path.exists():
            video_path.unlink()
            print(f"  🗑️ Cleaned up video: {video_path.name}")
            return True
        return False
    except Exception as e:
        print(f"  ⚠️ Could not delete video: {e}")
        return False


def cleanup_video_directory(video_dir: Path) -> bool:
    """Delete entire video directory (video + screenshots)"""
    try:
        if video_dir.exists():
            import shutil
            shutil.rmtree(video_dir)
            print(f"  🗑️ Cleaned up directory: {video_dir.name}")
            return True
        return False
    except Exception as e:
        print(f"  ⚠️ Could not delete directory: {e}")
        return False