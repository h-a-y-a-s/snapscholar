"""
Ultra-Simple URL Extractor
Just prints all video URLs from your Google Sheets export
"""

import pandas as pd
import sys
import os


def extract_urls_simple(csv_file):
    """Extract and print all URLs from CSV"""
    
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # Find URL column (it contains 'URL' or 'url' in name)
    url_column = None
    for col in df.columns:
        if 'url' in col.lower():
            url_column = col
            break
    
    if not url_column:
        print("❌ Could not find URL column")
        print(f"Available columns: {list(df.columns)}")
        return []
    
    # Extract unique URLs
    urls = df[url_column].dropna().unique().tolist()
    
    # Clean and filter
    clean_urls = []
    for url in urls:
        url = str(url).strip()
        if url.startswith('http'):
            clean_urls.append(url)
    
    return clean_urls


def main():
    # Find CSV file
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ No CSV file found")
        print("\nExport your Google Sheets as CSV first:")
        print("  File → Download → CSV")
        return
    
    csv_file = csv_files[0]
    print(f"Reading: {csv_file}\n")
    
    # Extract URLs
    urls = extract_urls_simple(csv_file)
    
    # Print results
    print(f"Found {len(urls)} video URLs:\n")
    print("="*70)
    
    for i, url in enumerate(urls, 1):
        print(f"{i}. {url}")
    
    print("="*70)
    
    # Save to file
    with open('video_urls.txt', 'w') as f:
        for url in urls:
            f.write(url + '\n')
    
    print(f"\n✓ Saved to video_urls.txt")
    
    # Also create JSON with video IDs
    import json
    
    video_data = []
    for url in urls:
        # Extract video ID
        if 'youtube.com/watch?v=' in url:
            video_id = url.split('watch?v=')[1].split('&')[0]
        elif 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[1].split('?')[0]
        else:
            video_id = url
        
        video_data.append({
            "video_id": video_id,
            "video_url": url
        })
    
    with open('video_urls.json', 'w') as f:
        json.dump(video_data, f, indent=2)
    
    print(f"✓ Saved to video_urls.json")
    
    print("\nUse these URLs to:")
    print("  1. Process with SnapScholar to generate study guides")
    print("  2. Then run evaluation")


if __name__ == "__main__":
    main()
