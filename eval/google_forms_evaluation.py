"""
SnapScholar Evaluation System - Adapted for Google Forms Data
Works with existing survey responses (1-5 scale, 2 criteria)
"""

import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Import LLM judges
import google.generativeai as genai
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Store evaluation results from an LLM judge"""
    summary_accuracy: int  # 1-5 scale
    screenshots_quality: int  # 1-5 scale
    overall_score: float  # Average (1-5 scale)
    reasoning: str


# Scoring criteria matching your Google Form
SCORING_CRITERIA = {
    "summary_accuracy": """
    Rate the summary accuracy (1-5):
    5: Summary is perfect
    4: Summary is clear
    3: Summary misses a minor point
    2: Summary has hallucinations (errors)
    1: Summary makes no sense
    """,
    
    "screenshots_quality": """
    Rate the screenshots quality (1-5):
    5: Every screenshot perfectly illustrates the concept text next to it
    4: Screenshots are relevant but maybe 1 is slightly blurry or mistimed
    3: Screenshots are okay but generic (e.g., showing the speaker instead of the slide)
    2: Screenshots are often irrelevant (black screens, transitions)
    1: All screenshots are unusable
    """
}


class GoogleFormsDataLoader:
    """Load and process Google Forms survey responses"""
    
    def __init__(self, csv_path: str):
        """
        Load Google Forms data from CSV export
        
        Args:
            csv_path: Path to exported CSV from Google Forms
        """
        self.df = pd.read_csv(csv_path)
        self._normalize_column_names()
    
    def _normalize_column_names(self):
        """Normalize column names to standard format"""
        # Google Forms exports have long question text as column names
        # Map them to simpler names
        
        column_mapping = {}
        for col in self.df.columns:
            if 'video title' in col.lower() or 'title' in col.lower():
                column_mapping[col] = 'video_title'
            elif 'video url' in col.lower() or 'url' in col.lower():
                column_mapping[col] = 'video_url'
            elif 'summary' in col.lower() and 'accu' in col.lower():
                column_mapping[col] = 'summary_accuracy'
            elif 'snapshot' in col.lower() or 'screenshot' in col.lower():
                column_mapping[col] = 'screenshots_quality'
            elif 'timestamp' in col.lower() or 'date' in col.lower():
                column_mapping[col] = 'timestamp'
        
        self.df.rename(columns=column_mapping, inplace=True)
        print(f"✓ Loaded {len(self.df)} responses from Google Forms")
    
    def get_human_scores(self) -> pd.DataFrame:
        """
        Get human scores in standardized format
        
        Returns:
            DataFrame with columns: video_id, summary_accuracy, screenshots_quality, overall
        """
        # Extract relevant columns
        required_cols = ['video_title', 'summary_accuracy', 'screenshots_quality']
        
        if not all(col in self.df.columns for col in required_cols):
            print("⚠ Warning: Not all required columns found")
            print(f"Available columns: {list(self.df.columns)}")
            return pd.DataFrame()
        
        scores_df = self.df[required_cols].copy()
        
        # Convert rating strings to integers if needed
        for col in ['summary_accuracy', 'screenshots_quality']:
            if scores_df[col].dtype == 'object':
                # Extract numbers from strings like "4 (Good)"
                scores_df[col] = scores_df[col].str.extract(r'(\d)')[0].astype(int)
        
        # Calculate overall score (average of two criteria)
        scores_df['overall'] = (scores_df['summary_accuracy'] + 
                                scores_df['screenshots_quality']) / 2.0
        
        # Create video_id from title (for matching with study guides)
        scores_df['video_id'] = scores_df['video_title'].str.replace(' ', '_').str.lower()
        
        return scores_df
    
    def export_for_correlation(self, output_file: str = "human_scores_processed.json"):
        """Export scores in format ready for correlation analysis"""
        scores_df = self.get_human_scores()
        
        scores_list = scores_df.to_dict('records')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(scores_list, f, indent=2)
        
        print(f"✓ Exported {len(scores_list)} human scores to {output_file}")
        return scores_list


class LLMJudgeAdapted:
    """LLM judge adapted for 1-5 scale with 2 criteria"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def create_evaluation_prompt(self, video_info: Dict, study_guide_content: Dict) -> str:
        """Create evaluation prompt matching Google Forms criteria"""
        
        prompt = f"""You are an expert educational content evaluator. Evaluate this AI-generated study guide from a YouTube video.

VIDEO INFORMATION:
- Title: {video_info.get('title', 'N/A')}
- Description: {video_info.get('description', 'N/A')}

GENERATED STUDY GUIDE:
{json.dumps(study_guide_content, indent=2)}

EVALUATION CRITERIA:

{SCORING_CRITERIA['summary_accuracy']}

{SCORING_CRITERIA['screenshots_quality']}

CRITICAL INSTRUCTIONS FOR SCORING:

1. USE THE FULL RANGE (1-5):
   - Be discriminating - not all study guides are the same quality
   - Only give 5 if truly excellent with NO issues whatsoever
   - Give 4 for good quality with only minor, forgivable issues
   - Give 3 for acceptable but with noticeable gaps or problems
   - Give 2 for poor quality with significant errors or missing content
   - Give 1 for completely unusable or severely flawed content

2. BE SPECIFIC:
   - Don't default to "middle" scores
   - Look for actual differences between study guides
   - Compare against the ideal, not against other examples

3. EVALUATE INDEPENDENTLY:
   - Rate summary accuracy separately from screenshots
   - One can be excellent while the other is poor

4. BE HONEST:
   - If something is truly excellent, say so (5)
   - If something has real problems, say so (2 or 3)
   - Most study guides should NOT all get the same score

RESPONSE FORMAT:
Respond in this EXACT JSON format:
{{
    "summary_accuracy": <integer 1-5>,
    "screenshots_quality": <integer 1-5>,
    "reasoning": {{
        "summary_accuracy": "<detailed explanation of score, mentioning specific strengths/weaknesses>",
        "screenshots_quality": "<detailed explanation of score, mentioning specific strengths/weaknesses>"
    }}
}}

REMEMBER: Use integers only (1, 2, 3, 4, or 5). Be discriminating and use the full range!"""
        
        return prompt
    
    def evaluate(self, video_info: Dict, study_guide_content: Dict) -> EvaluationResult:
        """Evaluate study guide and return result"""
        prompt = self.create_evaluation_prompt(video_info, study_guide_content)
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Extract JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            result = json.loads(response_text.strip())
            
            # Calculate overall score
            overall = (result['summary_accuracy'] + result['screenshots_quality']) / 2.0
            
            return EvaluationResult(
                summary_accuracy=result['summary_accuracy'],
                screenshots_quality=result['screenshots_quality'],
                overall_score=overall,
                reasoning=json.dumps(result.get('reasoning', {}), indent=2)
            )
        
        except Exception as e:
            print(f"Error in evaluation: {e}")
            if 'response_text' in locals():
                print(f"Response: {response_text}")
            raise


class SnapScholarGoogleFormsEvaluation:
    """Complete evaluation pipeline for Google Forms data"""
    
    def __init__(self, gemini_api_key: str):
        self.judge = LLMJudgeAdapted(gemini_api_key)
    
    def load_study_guide(self, file_path: str) -> Dict:
        """Load study guide from JSON"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def evaluate_samples(self, 
                        study_guides_dir: str,
                        human_scores_df: pd.DataFrame,
                        video_metadata: Dict = None) -> List[Dict]:
        """
        Evaluate samples and match with human scores
        
        Args:
            study_guides_dir: Directory with study guide JSON files
            human_scores_df: DataFrame with human scores from Google Forms
            video_metadata: Optional dict with video info
        
        Returns:
            List of evaluation results with both LLM and human scores
        """
        results = []
        
        print("\n" + "="*70)
        print("EVALUATING SAMPLES")
        print("="*70)
        
        # Get list of videos that have human scores
        videos_with_scores = set(human_scores_df['video_id'].values)
        
        for video_id in videos_with_scores:
            print(f"\nEvaluating: {video_id}")
            
            # Find study guide file
            study_guide_path = os.path.join(study_guides_dir, f"{video_id}.json")
            if not os.path.exists(study_guide_path):
                print(f"  ⚠ Study guide not found, skipping")
                continue
            
            # Load study guide
            study_guide_content = self.load_study_guide(study_guide_path)
            
            # Get video info
            if video_metadata and video_id in video_metadata:
                video_info = video_metadata[video_id]
            else:
                video_info = {
                    "title": video_id.replace('_', ' ').title(),
                    "description": "N/A"
                }
            
            # Get human scores for this video
            human_row = human_scores_df[human_scores_df['video_id'] == video_id].iloc[0]
            
            # LLM evaluation
            try:
                llm_result = self.judge.evaluate(video_info, study_guide_content)
                
                result = {
                    "video_id": video_id,
                    "human_scores": {
                        "summary_accuracy": int(human_row['summary_accuracy']),
                        "screenshots_quality": int(human_row['screenshots_quality']),
                        "overall": float(human_row['overall'])
                    },
                    "llm_scores": {
                        "summary_accuracy": llm_result.summary_accuracy,
                        "screenshots_quality": llm_result.screenshots_quality,
                        "overall": llm_result.overall_score
                    },
                    "llm_reasoning": llm_result.reasoning
                }
                
                results.append(result)
                
                print(f"  ✓ Human: {human_row['overall']:.2f}, LLM: {llm_result.overall_score:.2f}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        print("\n" + "="*70)
        print(f"✓ Evaluated {len(results)} samples")
        print("="*70)
        
        return results
    
    def compute_correlation(self, results: List[Dict]) -> Dict:
        """Compute correlation between human and LLM scores"""
        
        human_overall = [r['human_scores']['overall'] for r in results]
        llm_overall = [r['llm_scores']['overall'] for r in results]
        
        human_summary = [r['human_scores']['summary_accuracy'] for r in results]
        llm_summary = [r['llm_scores']['summary_accuracy'] for r in results]
        
        human_screenshots = [r['human_scores']['screenshots_quality'] for r in results]
        llm_screenshots = [r['llm_scores']['screenshots_quality'] for r in results]
        
        correlations = {
            "overall": {
                "pearson_r": pearsonr(human_overall, llm_overall)[0],
                "pearson_p": pearsonr(human_overall, llm_overall)[1],
                "spearman_r": spearmanr(human_overall, llm_overall)[0],
                "spearman_p": spearmanr(human_overall, llm_overall)[1],
                "mae": np.mean(np.abs(np.array(human_overall) - np.array(llm_overall))),
                "rmse": np.sqrt(np.mean((np.array(human_overall) - np.array(llm_overall))**2))
            },
            "summary_accuracy": {
                "pearson_r": pearsonr(human_summary, llm_summary)[0],
                "pearson_p": pearsonr(human_summary, llm_summary)[1],
            },
            "screenshots_quality": {
                "pearson_r": pearsonr(human_screenshots, llm_screenshots)[0],
                "pearson_p": pearsonr(human_screenshots, llm_screenshots)[1],
            }
        }
        
        return correlations
    
    def create_visualizations(self, results: List[Dict], correlations: Dict, output_dir: str):
        """Create correlation visualizations"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        human_overall = [r['human_scores']['overall'] for r in results]
        llm_overall = [r['llm_scores']['overall'] for r in results]
        
        human_summary = [r['human_scores']['summary_accuracy'] for r in results]
        llm_summary = [r['llm_scores']['summary_accuracy'] for r in results]
        
        human_screenshots = [r['human_scores']['screenshots_quality'] for r in results]
        llm_screenshots = [r['llm_scores']['screenshots_quality'] for r in results]
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('SnapScholar: LLM vs Human Evaluation (1-5 Scale)', 
                     fontsize=16, fontweight='bold')
        
        # 1. Overall scores scatter plot
        ax1 = axes[0, 0]
        ax1.scatter(human_overall, llm_overall, alpha=0.6, s=100, edgecolors='black')
        z = np.polyfit(human_overall, llm_overall, 1)
        p = np.poly1d(z)
        x_line = np.linspace(1, 5, 100)
        ax1.plot(x_line, p(x_line), "r--", linewidth=2, label='Best fit')
        ax1.plot([1, 5], [1, 5], 'k--', alpha=0.3, linewidth=1, label='Perfect correlation')
        ax1.set_xlabel('Human Scores (Overall)', fontsize=12)
        ax1.set_ylabel('LLM Scores (Overall)', fontsize=12)
        ax1.set_title(f'Overall Correlation (r = {correlations["overall"]["pearson_r"]:.3f})', 
                     fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0.5, 5.5)
        ax1.set_ylim(0.5, 5.5)
        
        # 2. Summary accuracy
        ax2 = axes[0, 1]
        ax2.scatter(human_summary, llm_summary, alpha=0.6, s=100, 
                   edgecolors='black', color='green')
        z = np.polyfit(human_summary, llm_summary, 1)
        p = np.poly1d(z)
        ax2.plot(x_line, p(x_line), "r--", linewidth=2)
        ax2.plot([1, 5], [1, 5], 'k--', alpha=0.3, linewidth=1)
        ax2.set_xlabel('Human Scores (Summary)', fontsize=12)
        ax2.set_ylabel('LLM Scores (Summary)', fontsize=12)
        ax2.set_title(f'Summary Accuracy (r = {correlations["summary_accuracy"]["pearson_r"]:.3f})', 
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0.5, 5.5)
        ax2.set_ylim(0.5, 5.5)
        
        # 3. Screenshots quality
        ax3 = axes[1, 0]
        ax3.scatter(human_screenshots, llm_screenshots, alpha=0.6, s=100, 
                   edgecolors='black', color='orange')
        z = np.polyfit(human_screenshots, llm_screenshots, 1)
        p = np.poly1d(z)
        ax3.plot(x_line, p(x_line), "r--", linewidth=2)
        ax3.plot([1, 5], [1, 5], 'k--', alpha=0.3, linewidth=1)
        ax3.set_xlabel('Human Scores (Screenshots)', fontsize=12)
        ax3.set_ylabel('LLM Scores (Screenshots)', fontsize=12)
        ax3.set_title(f'Screenshots Quality (r = {correlations["screenshots_quality"]["pearson_r"]:.3f})', 
                     fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0.5, 5.5)
        ax3.set_ylim(0.5, 5.5)
        
        # 4. Sample-by-sample comparison
        ax4 = axes[1, 1]
        sample_indices = np.arange(len(human_overall))
        width = 0.35
        ax4.bar(sample_indices - width/2, human_overall, width, label='Human', 
               color='blue', alpha=0.7, edgecolor='black')
        ax4.bar(sample_indices + width/2, llm_overall, width, label='LLM', 
               color='red', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Sample Index', fontsize=12)
        ax4.set_ylabel('Overall Score', fontsize=12)
        ax4.set_title('Sample-by-Sample Comparison', fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, 6)
        
        plt.tight_layout()
        
        plot_file = os.path.join(output_dir, "correlation_analysis.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to {plot_file}")
        plt.close()
    
    def run_complete_evaluation(self,
                               google_forms_csv: str,
                               study_guides_dir: str,
                               video_metadata_file: str = None,
                               output_dir: str = "evaluation_results"):
        """
        Run complete evaluation pipeline with Google Forms data
        
        Args:
            google_forms_csv: Path to Google Forms CSV export
            study_guides_dir: Directory with study guide JSON files
            video_metadata_file: Optional JSON with video metadata
            output_dir: Output directory for results
        """
        print("\n" + "="*70)
        print("SNAPSCHOLAR GOOGLE FORMS EVALUATION PIPELINE")
        print("="*70)
        
        # 1. Load Google Forms data
        print("\n1. Loading Google Forms data...")
        forms_loader = GoogleFormsDataLoader(google_forms_csv)
        human_scores_df = forms_loader.get_human_scores()
        
        if human_scores_df.empty:
            print("❌ Failed to load human scores")
            return
        
        print(f"✓ Loaded scores for {len(human_scores_df)} videos")
        
        # 2. Load video metadata if provided
        video_metadata = None
        if video_metadata_file and os.path.exists(video_metadata_file):
            with open(video_metadata_file, 'r') as f:
                video_metadata = json.load(f)
            print(f"✓ Loaded metadata for {len(video_metadata)} videos")
        
        # 3. Evaluate with LLM
        print("\n2. Running LLM evaluation...")
        results = self.evaluate_samples(study_guides_dir, human_scores_df, video_metadata)
        
        if not results:
            print("❌ No results generated")
            return
        
        # 4. Compute correlations
        print("\n3. Computing correlations...")
        correlations = self.compute_correlation(results)
        
        print("\n" + "="*70)
        print("CORRELATION RESULTS")
        print("="*70)
        print(f"\nOVERALL SCORES:")
        print(f"  Pearson r:  {correlations['overall']['pearson_r']:.3f} (p = {correlations['overall']['pearson_p']:.4f})")
        print(f"  Spearman ρ: {correlations['overall']['spearman_r']:.3f} (p = {correlations['overall']['spearman_p']:.4f})")
        print(f"  MAE:        {correlations['overall']['mae']:.3f}")
        print(f"  RMSE:       {correlations['overall']['rmse']:.3f}")
        
        print(f"\nSUMMARY ACCURACY:")
        print(f"  Pearson r:  {correlations['summary_accuracy']['pearson_r']:.3f} (p = {correlations['summary_accuracy']['pearson_p']:.4f})")
        
        print(f"\nSCREENSHOTS QUALITY:")
        print(f"  Pearson r:  {correlations['screenshots_quality']['pearson_r']:.3f} (p = {correlations['screenshots_quality']['pearson_p']:.4f})")
        print("="*70)
        
        # 5. Save results
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "evaluation_results.json"), 'w') as f:
            json.dump({
                "results": results,
                "correlations": correlations,
                "summary": {
                    "total_samples": len(results),
                    "mean_human_score": np.mean([r['human_scores']['overall'] for r in results]),
                    "mean_llm_score": np.mean([r['llm_scores']['overall'] for r in results])
                }
            }, f, indent=2)
        
        print(f"\n✓ Results saved to {output_dir}/evaluation_results.json")
        
        # 6. Create visualizations
        print("\n4. Creating visualizations...")
        self.create_visualizations(results, correlations, output_dir)
        
        print("\n" + "="*70)
        print("✓ EVALUATION COMPLETE!")
        print(f"✓ All results saved to: {output_dir}/")
        print("="*70)


def main():
    """Example usage"""
    
    # Set your API key
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not set")
        print("  export GEMINI_API_KEY='your_key_here'")
        return
    
    # Initialize pipeline
    pipeline = SnapScholarGoogleFormsEvaluation(GEMINI_API_KEY)
    
    # Run evaluation
    pipeline.run_complete_evaluation(
        google_forms_csv="google_forms_responses.csv",  # Your CSV export from Google Forms
        study_guides_dir="study_guides/",  # Your study guides directory
        video_metadata_file="video_metadata.json",  # Optional
        output_dir="evaluation_results/"
    )


if __name__ == "__main__":
    main()