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
    
    # Find columns - handle long Google Forms column names
    columns_found = {}
    for col in df.columns:
        col_lower = col.lower()
        
        # Video URL
        if 'video url' in col_lower or col == 'Video URL':
            columns_found['url'] = col
        
        # Video title
        elif 'video title' in col_lower or col == 'Video title':
            columns_found['title'] = col
        
        # Summary score - look for "Summary Accuracy" or "accuracy and clarity"
        elif ('summary' in col_lower and 'accuracy' in col_lower) or 'accuracy and clarity' in col_lower:
            columns_found['summary'] = col
        
        # Screenshot score - look for "Snapshots Quality" or "quality and relevance"  
        elif ('snapshot' in col_lower and 'quality' in col_lower) or 'quality and relevance' in col_lower or 'quality and releveance' in col_lower:
            columns_found['screenshot'] = col
    
    # Check we found all required columns
    required = ['url', 'title', 'summary', 'screenshot']
    missing = [r for r in required if r not in columns_found]
    
    if missing:
        print(f"❌ Could not find columns for: {missing}")
        print(f"\nAvailable columns:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        return []
    
    print(f"✓ Found required columns:")
    for key, col in columns_found.items():
        print(f"  - {key}: '{col}'")
    
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
    
    # Ensure we're saving in eval directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save URLs only
    urls_file = os.path.join(output_dir, 'video_urls.txt')
    with open(urls_file, 'w') as f:
        for v in videos:
            f.write(f"{v['url']}\n")
    print(f"✓ Saved URLs to {urls_file}")
    
    # Save complete data
    data_file = os.path.join(output_dir, 'videos_data.json')
    with open(data_file, 'w') as f:
        json.dump(videos, f, indent=2)
    print(f"✓ Saved complete data to {data_file}")
    
    return videos


def step2_check_study_guides(videos, study_guides_dir=None):
    """Step 2: Check which videos have study guides"""
    
    print("\n" + "="*70)
    print("STEP 2: Checking Study Guides")
    print("="*70)
    
    # Auto-detect study guides directory
    if study_guides_dir is None:
        # Try multiple possible paths
        possible_paths = [
            '../data/temp',
            'data/temp',
            os.path.join('..', 'data', 'temp'),
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'temp'))
        ]
        
        study_guides_dir = None
        for path in possible_paths:
            if os.path.exists(path):
                study_guides_dir = path
                print(f"✓ Found study guides directory: {os.path.abspath(path)}")
                break
    
    if study_guides_dir is None or not os.path.exists(study_guides_dir):
        print(f"⚠ Study guides directory not found")
        print(f"\nTried these paths:")
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            exists = "✓" if os.path.exists(path) else "✗"
            print(f"  {exists} {abs_path}")
        
        print("\nYou need to:")
        print("  1. Create study guides for these videos using SnapScholar")
        print("  2. Save them in the 'data/temp/' directory")
        print(f"  3. Each video in its own folder: data/temp/<video_id>/study_guide.docx")
        return []
    
    # List what's in the directory
    try:
        folders = [f for f in os.listdir(study_guides_dir) if os.path.isdir(os.path.join(study_guides_dir, f))]
        print(f"✓ Found {len(folders)} folders in {study_guides_dir}")
        if folders:
            print(f"  First few: {folders[:5]}")
    except Exception as e:
        print(f"⚠ Could not list directory: {e}")
        return []
    
    # Check which videos have study guides
    matched = []
    missing = []
    
    for video in videos:
        video_id = video['video_id']
        
        # Look for study guide in data/temp/{video_id}/ directory
        video_dir = os.path.join(study_guides_dir, video_id)
        
        if not os.path.exists(video_dir):
            missing.append(video)
            continue
        
        # Check for study_guide files (try multiple formats)
        possible_files = [
            os.path.join(video_dir, 'study_guide.docx'),
            os.path.join(video_dir, 'study_guide.md'),
            os.path.join(video_dir, 'study_guide.json'),
        ]
        
        found = False
        for filepath in possible_files:
            if os.path.exists(filepath):
                video['study_guide_path'] = filepath
                video['study_guide_format'] = os.path.splitext(filepath)[1]
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
        print(f"  Study guide: {video['study_guide_path']}")
        
        # Load study guide based on format
        try:
            study_guide_format = video.get('study_guide_format', '')
            
            if study_guide_format == '.json':
                # Load JSON
                with open(video['study_guide_path'], 'r', encoding='utf-8') as f:
                    study_guide = json.load(f)
            
            elif study_guide_format == '.md':
                # Load Markdown as text
                with open(video['study_guide_path'], 'r', encoding='utf-8') as f:
                    md_content = f.read()
                study_guide = {
                    'type': 'markdown',
                    'content': md_content
                }
            
            elif study_guide_format == '.docx':
                # Load DOCX as text
                from docx import Document
                doc = Document(video['study_guide_path'])
                docx_content = '\n'.join([para.text for para in doc.paragraphs])
                study_guide = {
                    'type': 'docx',
                    'content': docx_content
                }
            
            else:
                print(f"  ⚠ Unknown format: {study_guide_format}")
                continue
        
        except Exception as e:
            print(f"  ✗ Error loading study guide: {e}")
            continue
        
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
            print(f"  ✗ Error evaluating: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✓ Evaluated {len(results)} videos")
    
    # Save results in eval directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved results to {results_file}")
    
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
    
    # Save in eval directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    corr_file = os.path.join(output_dir, 'correlation_results.json')
    
    with open(corr_file, 'w', encoding='utf-8') as f:
        json.dump(correlations, f, indent=2)
    print(f"\n✓ Saved to {corr_file}")
    
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
    
    # Save in eval directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_file = os.path.join(output_dir, 'correlation_plot.png')
    
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to {plot_file}")
    plt.close()


def main():
    """Run complete workflow"""
    
    print("="*70)
    print("SNAPSCHOLAR: COMPLETE EVALUATION WORKFLOW")
    print("="*70)
    
    # Get the directory where this script is located (eval/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check for CSV file in the script's directory
    csv_files = [f for f in os.listdir(script_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("\n❌ No CSV file found in eval/ directory")
        print("\nTo use this script:")
        print("1. Convert your Excel file to CSV:")
        print("   cd eval")
        print("   python -c \"import pandas as pd; pd.read_excel('../SnapScholar Study Guide Evaluation Form (Responses).xlsx').to_csv('form_responses.csv', index=False)\"")
        print("2. Or manually: Open Excel → File → Save As → CSV (save in eval/ folder)")
        print("3. Run: python complete_workflow.py")
        return
    
    csv_file = os.path.join(script_dir, csv_files[0])
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
    
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"\nGenerated files in eval/ directory:")
    print(f"  • video_urls.txt - List of video URLs")
    print(f"  • videos_data.json - Complete data")
    print(f"  • evaluation_results.json - LLM vs Human scores")
    print(f"  • correlation_results.json - Statistics")
    print(f"  • correlation_plot.png - Visualization")
    print(f"\nLocation: {eval_dir}")
    print("\nYou're ready for your presentation! 🎉")
    print("="*70)


if __name__ == "__main__":
    main()