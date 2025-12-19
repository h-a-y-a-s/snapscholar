"""
Complete Automated Evaluation Workflow
From Google Sheets → URLs → Evaluation → Results

Loads GOOGLE_API_KEY from .env file in project root

Just run: python complete_workflow.py
"""

import pandas as pd
import json
import os
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env file in parent directory (project root)
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded .env from {env_path}")
    else:
        print(f"⚠ .env file not found at {env_path}")
except ImportError:
    print("⚠ python-dotenv not installed, using environment variables directly")
    print("  Install with: pip install python-dotenv")


def step1_extract_data(csv_file):
    """Step 1: Extract all data from Google Sheets"""
    
    print("\n" + "="*70)
    print("STEP 1: Extracting Data from Google Sheets")
    print("="*70)
    
    # Read CSV
    df = pd.read_csv(csv_file)
    print(f"✓ Loaded {len(df)} responses")
    
    # Find columns
    columns_found = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'url' in col_lower:
            columns_found['url'] = col
        elif 'title' in col_lower:
            columns_found['title'] = col
        elif 'summary' in col_lower and 'screenshot' not in col_lower:
            columns_found['summary'] = col
        elif 'screenshot' in col_lower:
            columns_found['screenshot'] = col
    
    print(f"✓ Found columns: {list(columns_found.values())}")
    
    # Extract data
    videos = []
    for idx, row in df.iterrows():
        # Get URL
        url = str(row[columns_found['url']]).strip()
        if not url.startswith('http'):
            continue
        
        # Extract video ID from URL
        if 'youtube.com/watch?v=' in url:
            video_id = url.split('watch?v=')[1].split('&')[0]
        elif 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[1].split('?')[0]
        else:
            # Use row index as fallback
            video_id = f"video_{idx}"
        
        # Get title
        title = str(row[columns_found['title']]).strip()
        
        # Get scores (handle "5 (Excellent)" format)
        summary_score = str(row[columns_found['summary']])
        screenshot_score = str(row[columns_found['screenshot']])
        
        # Extract just the number
        summary_score = summary_score.split('(')[0].strip() if '(' in summary_score else summary_score
        screenshot_score = screenshot_score.split('(')[0].strip() if '(' in screenshot_score else screenshot_score
        
        try:
            summary_score = int(summary_score)
            screenshot_score = int(screenshot_score)
        except:
            print(f"⚠ Could not parse scores for: {title}")
            continue
        
        videos.append({
            'video_id': video_id,
            'title': title,
            'url': url,
            'summary_score': summary_score,
            'screenshot_score': screenshot_score,
            'overall_score': (summary_score + screenshot_score) / 2.0
        })
    
    print(f"✓ Extracted {len(videos)} videos with scores")
    
    # Save URLs only
    with open('video_urls.txt', 'w') as f:
        for v in videos:
            f.write(f"{v['url']}\n")
    print(f"✓ Saved URLs to video_urls.txt")
    
    # Save complete data
    with open('videos_data.json', 'w') as f:
        json.dump(videos, f, indent=2)
    print(f"✓ Saved complete data to videos_data.json")
    
    return videos


def step2_check_study_guides(videos, study_guides_dir='../study_guides'):
    """Step 2: Check which videos have study guides"""
    
    print("\n" + "="*70)
    print("STEP 2: Checking Study Guides")
    print("="*70)
    
    if not os.path.exists(study_guides_dir):
        print(f"⚠ Directory not found: {study_guides_dir}")
        print("\nYou need to:")
        print("  1. Create study guides for these videos using SnapScholar")
        print("  2. Save them in the '../study_guides/' directory")
        print(f"  3. Name them: <video_id>.json")
        return []
    
    # Check which videos have study guides
    matched = []
    missing = []
    
    for video in videos:
        video_id = video['video_id']
        possible_files = [
            f"{video_id}.json",
            f"{video_id}.docx",
            f"{video['title'].lower().replace(' ', '_')}.json"
        ]
        
        found = False
        for filename in possible_files:
            filepath = os.path.join(study_guides_dir, filename)
            if os.path.exists(filepath):
                video['study_guide_path'] = filepath
                matched.append(video)
                found = True
                break
        
        if not found:
            missing.append(video)
    
    print(f"✓ Matched: {len(matched)} videos have study guides")
    
    if missing:
        print(f"⚠ Missing: {len(missing)} videos need study guides")
        print("\nMissing study guides for:")
        for v in missing[:5]:
            print(f"  - {v['title']} (ID: {v['video_id']})")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
    
    return matched


def step3_run_evaluation(matched_videos, api_key):
    """Step 3: Run LLM evaluation"""
    
    print("\n" + "="*70)
    print("STEP 3: Running LLM Evaluation")
    print("="*70)
    
    if not matched_videos:
        print("❌ No videos to evaluate")
        return []
    
    try:
        from google_forms_evaluation import LLMJudgeAdapted
    except ImportError:
        print("❌ Could not import evaluation module")
        print("Make sure google_forms_evaluation.py is in the same directory")
        return []
    
    judge = LLMJudgeAdapted(api_key)
    results = []
    
    for i, video in enumerate(matched_videos, 1):
        print(f"\n[{i}/{len(matched_videos)}] Evaluating: {video['title']}")
        
        # Load study guide
        with open(video['study_guide_path'], 'r', encoding='utf-8') as f:
            study_guide = json.load(f)
        
        # Create video info
        video_info = {
            'title': video['title'],
            'description': f"Video ID: {video['video_id']}"
        }
        
        # Evaluate
        try:
            llm_result = judge.evaluate(video_info, study_guide)
            
            result = {
                'video_id': video['video_id'],
                'title': video['title'],
                'url': video['url'],
                'human': {
                    'summary': video['summary_score'],
                    'screenshots': video['screenshot_score'],
                    'overall': video['overall_score']
                },
                'llm': {
                    'summary': llm_result.summary_accuracy,
                    'screenshots': llm_result.screenshots_quality,
                    'overall': llm_result.overall_score
                }
            }
            
            results.append(result)
            print(f"  ✓ Human: {video['overall_score']:.1f}, LLM: {llm_result.overall_score:.1f}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\n✓ Evaluated {len(results)} videos")
    
    # Save results
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved results to evaluation_results.json")
    
    return results


def step4_compute_correlation(results):
    """Step 4: Compute correlation statistics"""
    
    print("\n" + "="*70)
    print("STEP 4: Computing Correlation")
    print("="*70)
    
    if len(results) < 3:
        print("⚠ Need at least 3 samples for correlation")
        return None
    
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    
    # Extract scores
    human_overall = [r['human']['overall'] for r in results]
    llm_overall = [r['llm']['overall'] for r in results]
    
    human_summary = [r['human']['summary'] for r in results]
    llm_summary = [r['llm']['summary'] for r in results]
    
    human_screenshots = [r['human']['screenshots'] for r in results]
    llm_screenshots = [r['llm']['screenshots'] for r in results]
    
    # Compute correlations
    correlations = {
        'overall': {
            'pearson_r': float(pearsonr(human_overall, llm_overall)[0]),
            'pearson_p': float(pearsonr(human_overall, llm_overall)[1]),
            'spearman_r': float(spearmanr(human_overall, llm_overall)[0]),
            'mae': float(np.mean(np.abs(np.array(human_overall) - np.array(llm_overall))))
        },
        'summary': {
            'pearson_r': float(pearsonr(human_summary, llm_summary)[0])
        },
        'screenshots': {
            'pearson_r': float(pearsonr(human_screenshots, llm_screenshots)[0])
        }
    }
    
    # Print results
    print(f"\nOVERALL SCORES:")
    print(f"  Pearson r:  {correlations['overall']['pearson_r']:.3f}")
    print(f"  p-value:    {correlations['overall']['pearson_p']:.4f}")
    print(f"  Spearman ρ: {correlations['overall']['spearman_r']:.3f}")
    print(f"  MAE:        {correlations['overall']['mae']:.3f}")
    
    print(f"\nSUMMARY: r = {correlations['summary']['pearson_r']:.3f}")
    print(f"SCREENSHOTS: r = {correlations['screenshots']['pearson_r']:.3f}")
    
    # Interpret
    r = correlations['overall']['pearson_r']
    if r > 0.7:
        strength = "STRONG"
    elif r > 0.5:
        strength = "MODERATE"
    else:
        strength = "WEAK"
    
    print(f"\n✓ {strength} correlation detected!")
    
    if correlations['overall']['pearson_p'] < 0.05:
        print(f"✓ Statistically significant (p < 0.05)")
    else:
        print(f"⚠ Not statistically significant (p ≥ 0.05)")
    
    # Save
    with open('correlation_results.json', 'w', encoding='utf-8') as f:
        json.dump(correlations, f, indent=2)
    print(f"\n✓ Saved to correlation_results.json")
    
    return correlations


def step5_create_visualization(results, correlations):
    """Step 5: Create visualization"""
    
    print("\n" + "="*70)
    print("STEP 5: Creating Visualization")
    print("="*70)
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("⚠ matplotlib not installed, skipping visualization")
        return
    
    # Extract data
    human_overall = [r['human']['overall'] for r in results]
    llm_overall = [r['llm']['overall'] for r in results]
    
    # Create simple scatter plot
    plt.figure(figsize=(10, 8))
    
    plt.scatter(human_overall, llm_overall, s=100, alpha=0.6, edgecolors='black')
    
    # Add regression line
    z = np.polyfit(human_overall, llm_overall, 1)
    p = np.poly1d(z)
    x_line = np.linspace(1, 5, 100)
    plt.plot(x_line, p(x_line), "r--", linewidth=2, label='Best fit')
    
    # Add perfect correlation line
    plt.plot([1, 5], [1, 5], 'k--', alpha=0.3, linewidth=1, label='Perfect correlation')
    
    plt.xlabel('Human Scores', fontsize=14)
    plt.ylabel('LLM Scores', fontsize=14)
    plt.title(f'SnapScholar Evaluation: Human vs LLM\nr = {correlations["overall"]["pearson_r"]:.3f}, p = {correlations["overall"]["pearson_p"]:.4f}',
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0.5, 5.5)
    plt.ylim(0.5, 5.5)
    
    plt.tight_layout()
    plt.savefig('correlation_plot.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to correlation_plot.png")
    plt.close()


def main():
    """Run complete workflow"""
    
    print("="*70)
    print("SNAPSCHOLAR: COMPLETE EVALUATION WORKFLOW")
    print("="*70)
    
    # Check for CSV file
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("\n❌ No CSV file found in current directory")
        print("\nTo use this script:")
        print("1. Convert your Excel file to CSV:")
        print("   python -c \"import pandas as pd; pd.read_excel('SnapScholar Study Guide Evaluation Form (Responses).xlsx').to_csv('form_responses.csv', index=False)\"")
        print("2. Or manually: Open Excel → File → Save As → CSV")
        print("3. Run: python complete_workflow.py")
        return
    
    csv_file = csv_files[0]
    print(f"\n✓ Using CSV file: {csv_file}")
    
    # Check for API key - try GOOGLE_API_KEY first, then GEMINI_API_KEY
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("\n❌ API key not found")
        print("\nExpected GOOGLE_API_KEY in .env file")
        print("\nCreate/edit .env file in project root with:")
        print("GOOGLE_API_KEY=your_api_key_here")
        print("\nOr set environment variable:")
        print("export GOOGLE_API_KEY=your_api_key_here")
        return
    
    key_source = 'GOOGLE_API_KEY' if os.getenv('GOOGLE_API_KEY') else 'GEMINI_API_KEY'
    print(f"✓ API key found (from {key_source})")
    
    # Run all steps
    videos = step1_extract_data(csv_file)
    
    if not videos:
        print("\n❌ Failed to extract data")
        return
    
    matched = step2_check_study_guides(videos)
    
    if not matched:
        print("\n⚠ No study guides found")
        print("\nNext steps:")
        print("1. Use video_urls.txt to process videos with SnapScholar")
        print("2. Save study guides to ../study_guides/ directory")
        print("3. Run this script again")
        return
    
    results = step3_run_evaluation(matched, api_key)
    
    if not results:
        print("\n❌ Evaluation failed")
        return
    
    correlations = step4_compute_correlation(results)
    
    if correlations:
        step5_create_visualization(results, correlations)
    
    # Final summary
    print("\n" + "="*70)
    print("✓✓✓ WORKFLOW COMPLETE! ✓✓✓")
    print("="*70)
    print("\nGenerated files:")
    print("  • video_urls.txt - List of video URLs")
    print("  • videos_data.json - Complete data")
    print("  • evaluation_results.json - LLM vs Human scores")
    print("  • correlation_results.json - Statistics")
    print("  • correlation_plot.png - Visualization")
    print("\nYou're ready for your presentation! 🎉")
    print("="*70)


if __name__ == "__main__":
    main()
