"""
Analyze synthetic TTS data quality and diversity.

This module checks:
1. Audio diversity - Are samples actually different?
2. Spectral diversity - Do they have different frequency content?
3. Temporal diversity - Do they have different timing/duration patterns?
4. Similarity clustering - Are there duplicate-like samples?
"""

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict
import logging
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import librosa
import pandas as pd

logger = logging.getLogger(__name__)


class SyntheticDataQualityAnalyzer:
    """Analyzes quality and diversity of synthetic TTS data"""
    
    def __init__(self, synthetic_dataset_path: str):
        """
        Initialize analyzer.
        
        Args:
            synthetic_dataset_path: Path to synthetic dataset directory
        """
        self.dataset_path = Path(synthetic_dataset_path)
        
        # Load dataset
        from synthetic_data_loader import SyntheticDatasetLoader
        self.loader = SyntheticDatasetLoader(str(self.dataset_path))
        
        logger.info(f"Loaded dataset from {synthetic_dataset_path}")
        logger.info(f"Total samples: {self.loader.info['total_samples']}")
    
    def sample_random_audios(self, keyword: str, n_samples: int = 10,
                            random_state: int = 42) -> List[torch.Tensor]:
        """Sample random audio files for a keyword"""
        audio_samples, _ = self.loader.sample_keyword_data(
            keyword, n_samples, random_state
        )
        return audio_samples
    
    def calculate_audio_similarity(self, audio1: torch.Tensor, 
                                  audio2: torch.Tensor) -> float:
        """
        Calculate similarity between two audio samples.
        
        Returns:
            Pearson correlation coefficient (0=different, 1=identical)
        """
        # Flatten to 1D
        a1 = audio1.flatten().numpy()
        a2 = audio2.flatten().numpy()
        
        # Pearson correlation
        corr, _ = pearsonr(a1, a2)
        return corr
    
    def calculate_spectral_features(self, audio: torch.Tensor) -> Dict:
        """
        Extract spectral features from audio.
        
        Returns:
            Dictionary with spectral features
        """
        audio_np = audio.squeeze().numpy()
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_np, sr=16000, n_mels=128
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_np, sr=16000)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_np, sr=16000)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_np)
        
        # RMS energy
        rms = librosa.feature.rms(y=audio_np)
        
        return {
            'mel_spec': mel_spec_db,
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_centroid_std': np.std(spectral_centroid),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'spectral_rolloff_std': np.std(spectral_rolloff),
            'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
            'rms_mean': np.mean(rms),
            'rms_std': np.std(rms)
        }
    
    def analyze_keyword_diversity(self, keyword: str, n_samples: int = 100,
                                 random_state: int = 42) -> Dict:
        """
        Analyze diversity within a keyword.
        
        Args:
            keyword: Keyword to analyze
            n_samples: Number of samples to analyze
            random_state: Random seed
        
        Returns:
            Dictionary with diversity metrics
        """
        logger.info(f"Analyzing diversity for '{keyword}' ({n_samples} samples)")
        
        # Sample audios
        audios = self.sample_random_audios(keyword, n_samples, random_state)
        
        # Calculate pairwise similarities
        n = len(audios)
        similarities = []
        
        logger.info("Calculating pairwise similarities...")
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.calculate_audio_similarity(audios[i], audios[j])
                similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # Extract spectral features
        logger.info("Extracting spectral features...")
        features = []
        for audio in audios:
            feat = self.calculate_spectral_features(audio)
            features.append([
                feat['spectral_centroid_mean'],
                feat['spectral_rolloff_mean'],
                feat['zero_crossing_rate_mean'],
                feat['rms_mean']
            ])
        
        features = np.array(features)
        
        # Calculate feature diversity
        feature_std = np.std(features, axis=0)
        feature_range = np.ptp(features, axis=0)  # Peak-to-peak (max - min)
        
        results = {
            'keyword': keyword,
            'n_samples': n_samples,
            'similarity_mean': np.mean(similarities),
            'similarity_std': np.std(similarities),
            'similarity_min': np.min(similarities),
            'similarity_max': np.max(similarities),
            'similarity_median': np.median(similarities),
            'high_similarity_pairs': np.sum(similarities > 0.9),  # Very similar
            'low_similarity_pairs': np.sum(similarities < 0.5),   # Quite different
            'feature_std': feature_std,
            'feature_range': feature_range,
            'spectral_centroid_std': np.std(features[:, 0]),
            'spectral_rolloff_std': np.std(features[:, 1]),
            'zero_crossing_std': np.std(features[:, 2]),
            'rms_std': np.std(features[:, 3])
        }
        
        logger.info(f"Diversity analysis complete for '{keyword}'")
        logger.info(f"  Mean similarity: {results['similarity_mean']:.3f}")
        logger.info(f"  Similarity range: [{results['similarity_min']:.3f}, {results['similarity_max']:.3f}]")
        logger.info(f"  High similarity pairs (>0.9): {results['high_similarity_pairs']} / {len(similarities)}")
        
        return results
    
    def plot_diversity_analysis(self, keyword: str, n_samples: int = 100,
                               save_path: str = None):
        """
        Create visualization of diversity analysis.
        
        Args:
            keyword: Keyword to analyze
            n_samples: Number of samples
            save_path: Optional path to save plot
        """
        # Get samples
        audios = self.sample_random_audios(keyword, n_samples)
        
        # Calculate similarities
        n = len(audios)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                elif i < j:
                    sim = self.calculate_audio_similarity(audios[i], audios[j])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        
        # Extract features for PCA
        features = []
        for audio in audios:
            feat = self.calculate_spectral_features(audio)
            features.append([
                feat['spectral_centroid_mean'],
                feat['spectral_rolloff_mean'],
                feat['zero_crossing_rate_mean'],
                feat['rms_mean']
            ])
        
        features = np.array(features)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Diversity Analysis: "{keyword}" ({n_samples} samples)', 
                    fontsize=16, fontweight='bold')
        
        # 1. Similarity heatmap
        ax1 = axes[0, 0]
        sns.heatmap(similarity_matrix, cmap='RdYlGn_r', vmin=0, vmax=1, 
                   ax=ax1, cbar_kws={'label': 'Similarity'})
        ax1.set_title('Pairwise Similarity Matrix')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Sample Index')
        
        # 2. Similarity distribution
        ax2 = axes[0, 1]
        upper_tri = similarity_matrix[np.triu_indices(n, k=1)]
        ax2.hist(upper_tri, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(upper_tri), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(upper_tri):.3f}')
        ax2.axvline(0.9, color='orange', linestyle='--', 
                   label='High similarity threshold')
        ax2.set_xlabel('Pairwise Similarity')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Similarities')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. PCA visualization
        ax3 = axes[1, 0]
        if n >= 2:
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features)
            
            scatter = ax3.scatter(features_2d[:, 0], features_2d[:, 1], 
                                alpha=0.6, s=50, c=range(n), cmap='viridis')
            ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
            ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
            ax3.set_title('Feature Space (PCA)')
            ax3.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax3, label='Sample Index')
        
        # 4. Feature distributions
        ax4 = axes[1, 1]
        feature_names = ['Spectral\nCentroid', 'Spectral\nRolloff', 
                        'Zero\nCrossing', 'RMS\nEnergy']
        
        # Normalize features for comparison
        features_norm = (features - features.mean(axis=0)) / features.std(axis=0)
        
        bp = ax4.boxplot(features_norm, labels=feature_names, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('skyblue')
        ax4.set_ylabel('Normalized Feature Value')
        ax4.set_title('Feature Distributions (Normalized)')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved diversity plot to {save_path}")
        
        return fig
    
    def generate_diversity_report(self, n_samples_per_keyword: int = 100) -> pd.DataFrame:
        """
        Generate diversity report for all keywords.
        
        Args:
            n_samples_per_keyword: Number of samples to analyze per keyword
        
        Returns:
            DataFrame with diversity metrics for each keyword
        """
        keywords = self.loader.info['keywords']
        results = []
        
        print("\n" + "="*70)
        print("SYNTHETIC DATA DIVERSITY REPORT")
        print("="*70)
        
        for keyword in keywords:
            print(f"\nAnalyzing '{keyword}'...")
            analysis = self.analyze_keyword_diversity(keyword, n_samples_per_keyword)
            results.append(analysis)
        
        df = pd.DataFrame(results)
        
        # Print summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"\nMean similarity across all keywords: {df['similarity_mean'].mean():.3f}")
        print(f"  • Range: [{df['similarity_mean'].min():.3f}, {df['similarity_mean'].max():.3f}]")
        
        print(f"\nHigh similarity pairs (>0.9):")
        for _, row in df.iterrows():
            total_pairs = row['n_samples'] * (row['n_samples'] - 1) / 2
            pct = (row['high_similarity_pairs'] / total_pairs) * 100
            print(f"  • {row['keyword']}: {row['high_similarity_pairs']:.0f} / {total_pairs:.0f} ({pct:.1f}%)")
        
        print(f"\nSpectral diversity (std of spectral centroids):")
        for _, row in df.iterrows():
            print(f"  • {row['keyword']}: {row['spectral_centroid_std']:.2f}")
        
        # Interpretation
        avg_sim = df['similarity_mean'].mean()
        print("\n" + "="*70)
        print("INTERPRETATION")
        print("="*70)
        
        if avg_sim > 0.95:
            print("⚠️  WARNING: Very high similarity!")
            print("   Samples are nearly identical - limited diversity")
            print("   Recommendation: Increase prosodic variations in TTS generation")
        elif avg_sim > 0.85:
            print("⚠️  MODERATE: Similarity is moderately high")
            print("   Some diversity exists but could be improved")
            print("   Recommendation: Consider more acoustic variations")
        elif avg_sim > 0.7:
            print("✓ GOOD: Reasonable diversity")
            print("   Samples have meaningful differences")
        else:
            print("✅ EXCELLENT: High diversity")
            print("   Samples are quite different from each other")
        
        print("="*70 + "\n")
        
        return df


def quick_diversity_check(synthetic_path: str, keyword: str = None, 
                         n_samples: int = 50):
    """
    Quick diversity check for a synthetic dataset.
    
    Args:
        synthetic_path: Path to synthetic dataset
        keyword: Specific keyword to check (or None for first available)
        n_samples: Number of samples to analyze
    """
    from synthetic_data_loader import SyntheticDatasetLoader
    
    loader = SyntheticDatasetLoader(synthetic_path)
    
    if keyword is None:
        keyword = loader.info['keywords'][0]
    
    print(f"\n🔍 Quick Diversity Check: '{keyword}' ({n_samples} samples)")
    print("="*60)
    
    analyzer = SyntheticDataQualityAnalyzer(synthetic_path)
    results = analyzer.analyze_keyword_diversity(keyword, n_samples)
    
    # Simple verdict
    mean_sim = results['similarity_mean']
    high_sim_pct = (results['high_similarity_pairs'] / (n_samples * (n_samples - 1) / 2)) * 100
    
    print(f"\n📊 Results:")
    print(f"  Mean similarity: {mean_sim:.3f}")
    print(f"  Very similar pairs: {high_sim_pct:.1f}%")
    
    if mean_sim > 0.95 or high_sim_pct > 50:
        print(f"\n⚠️  CONCERN: Samples may be too similar!")
        print(f"     This could limit augmentation effectiveness")
    elif mean_sim > 0.85 or high_sim_pct > 30:
        print(f"\n⚠️  MODERATE: Some diversity, but could be better")
    else:
        print(f"\n✅ GOOD: Sufficient diversity for augmentation")
    
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze synthetic data quality')
    parser.add_argument('--dataset', type=str, 
                       default='./synthetic_datasets/gsc_synthetic_comprehensive',
                       help='Path to synthetic dataset')
    parser.add_argument('--keyword', type=str, default=None,
                       help='Keyword to analyze (default: first available)')
    parser.add_argument('--n-samples', type=int, default=100,
                       help='Number of samples to analyze')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--full-report', action='store_true',
                       help='Generate full diversity report for all keywords')
    
    args = parser.parse_args()
    
    if args.full_report:
        analyzer = SyntheticDataQualityAnalyzer(args.dataset)
        df = analyzer.generate_diversity_report(args.n_samples)
        
        # Save report
        output_dir = Path(args.dataset) / 'quality_analysis'
        output_dir.mkdir(exist_ok=True)
        df.to_csv(output_dir / 'diversity_report.csv', index=False)
        print(f"✓ Report saved to {output_dir / 'diversity_report.csv'}")
        
        # Generate plots for each keyword
        if args.plot:
            for keyword in df['keyword']:
                fig = analyzer.plot_diversity_analysis(keyword, args.n_samples)
                plot_path = output_dir / f'diversity_{keyword}.png'
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"✓ Plot saved: {plot_path}")
    
    else:
        # Quick check
        quick_diversity_check(args.dataset, args.keyword, args.n_samples)
        
        if args.plot:
            analyzer = SyntheticDataQualityAnalyzer(args.dataset)
            from synthetic_data_loader import SyntheticDatasetLoader
            loader = SyntheticDatasetLoader(args.dataset)
            keyword = args.keyword or loader.info['keywords'][0]
            
            fig = analyzer.plot_diversity_analysis(keyword, args.n_samples)
            output_path = Path(args.dataset) / f'diversity_{keyword}.png'
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Visualization saved to {output_path}")
            plt.show()
