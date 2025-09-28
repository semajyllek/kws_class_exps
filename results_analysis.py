"""
Results analysis and visualization for class imbalance experiments.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class ExperimentAnalyzer:
    """Analyzes experimental results and generates publication-quality plots"""
    
    def __init__(self, results_file: str):
        self.results_file = Path(results_file)
        self.results_df = self._load_and_validate_results()
        self.summary_df = self._prepare_summary_statistics()
        
    def _load_and_validate_results(self) -> pd.DataFrame:
        """Load and validate experimental results"""
        try:
            df = pd.read_csv(self.results_file)
            logger.info(f"Loaded {len(df)} experimental results")
            
            # Validate required columns
            required_cols = ['dataset_size', 'imbalance_ratio', 'augmentation_method', 
                           'trial', 'f1_keyword']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            raise
    
    def _prepare_summary_statistics(self) -> pd.DataFrame:
        """Prepare summary statistics across trials"""
        
        # Group by experimental conditions and calculate statistics
        grouping_cols = ['dataset_size', 'imbalance_ratio', 'augmentation_method']
        
        summary = self.results_df.groupby(grouping_cols).agg({
            'f1_keyword': ['mean', 'std', 'count'],
            'f1_non_keyword': ['mean', 'std'],
            'balanced_accuracy': ['mean', 'std'],
            'precision_keyword': ['mean', 'std'],
            'recall_keyword': ['mean', 'std'],
            'auc_roc': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        summary.columns = [
            '_'.join(col).strip() if col[1] else col[0] 
            for col in summary.columns.values
        ]
        
        logger.info(f"Prepared summary statistics for {len(summary)} conditions")
        return summary
    
    def calculate_improvement_metrics(self, baseline_method: str = 'none') -> pd.DataFrame:
        """Calculate improvement over baseline method"""
        
        baseline_data = self.summary_df[
            self.summary_df['augmentation_method'] == baseline_method
        ].copy()
        
        improvement_records = []
        
        for _, row in self.summary_df.iterrows():
            if row['augmentation_method'] == baseline_method:
                continue
            
            # Find matching baseline condition
            baseline_match = baseline_data[
                (baseline_data['dataset_size'] == row['dataset_size']) &
                (baseline_data['imbalance_ratio'] == row['imbalance_ratio'])
            ]
            
            if len(baseline_match) == 0:
                logger.warning(f"No baseline found for condition: "
                             f"{row['dataset_size']}, {row['imbalance_ratio']}")
                continue
            
            baseline_f1 = baseline_match['f1_keyword_mean'].iloc[0]
            current_f1 = row['f1_keyword_mean']
            
            absolute_improvement = current_f1 - baseline_f1
            relative_improvement = (absolute_improvement / baseline_f1 * 100) if baseline_f1 > 0 else 0
            
            improvement_records.append({
                'dataset_size': row['dataset_size'],
                'imbalance_ratio': row['imbalance_ratio'],
                'augmentation_method': row['augmentation_method'],
                'baseline_f1': baseline_f1,
                'augmented_f1': current_f1,
                'absolute_improvement': absolute_improvement,
                'relative_improvement': relative_improvement,
                'effect_size': absolute_improvement / row['f1_keyword_std'] if row['f1_keyword_std'] > 0 else 0
            })
        
        improvement_df = pd.DataFrame(improvement_records)
        logger.info(f"Calculated improvements for {len(improvement_df)} conditions")
        
        return improvement_df
    
    def create_performance_heatmap(self, metric: str = 'f1_keyword', 
                                 save_path: Optional[str] = None) -> plt.Figure:
        """Create heatmap showing performance across all conditions"""
        
        n_methods = len(self.summary_df['augmentation_method'].unique())
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        fig.suptitle(f'{metric.replace("_", " ").title()} Performance Across Conditions', 
                    fontsize=16, fontweight='bold')
        
        for i, method in enumerate(self.summary_df['augmentation_method'].unique()):
            if i >= 4:  # Limit to 4 subplots
                break
                
            ax = axes[i]
            
            # Filter data for this method
            method_data = self.summary_df[
                self.summary_df['augmentation_method'] == method
            ]
            
            # Create pivot table
            heatmap_data = method_data.pivot_table(
                values=f'{metric}_mean',
                index='imbalance_ratio',
                columns='dataset_size',
                aggfunc='first'
            )
            
            # Create heatmap
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', ax=ax, 
                       cmap='viridis', cbar_kws={'label': f'{metric}_mean'})
            
            ax.set_title(f'{method.title()} Method', fontweight='bold')
            ax.set_xlabel('Dataset Size')
            ax.set_ylabel('Imbalance Ratio (minority:majority)')
        
        # Remove unused subplots
        for j in range(i + 1, 4):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved performance heatmap to {save_path}")
        
        return fig
    
    def create_improvement_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive improvement analysis plots"""
        
        improvement_df = self.calculate_improvement_metrics()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Improvement Analysis Over Baseline', fontsize=16, fontweight='bold')
        
        # Plot 1: Improvement by imbalance ratio
        ax1 = axes[0, 0]
        sns.boxplot(data=improvement_df, x='imbalance_ratio', y='relative_improvement',
                   hue='augmentation_method', ax=ax1)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax1.set_title('Relative Improvement vs Imbalance Ratio')
        ax1.set_xlabel('Imbalance Ratio')
        ax1.set_ylabel('Relative Improvement (%)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Improvement by dataset size
        ax2 = axes[0, 1]
        sns.boxplot(data=improvement_df, x='dataset_size', y='relative_improvement',
                   hue='augmentation_method', ax=ax2)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax2.set_title('Relative Improvement vs Dataset Size')
        ax2.set_xlabel('Dataset Size')
        ax2.set_ylabel('Relative Improvement (%)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 3: Distribution of improvements
        ax3 = axes[1, 0]
        methods = improvement_df['augmentation_method'].unique()
        for method in methods:
            method_data = improvement_df[improvement_df['augmentation_method'] == method]
            ax3.hist(method_data['relative_improvement'], alpha=0.6, label=method, bins=15)
        
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Relative Improvement (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Improvements')
        ax3.legend()
        
        # Plot 4: Method effectiveness heatmap
        ax4 = axes[1, 1]
        effectiveness_data = improvement_df.pivot_table(
            values='relative_improvement',
            index='imbalance_ratio',
            columns='augmentation_method',
            aggfunc='mean'
        )
        
        sns.heatmap(effectiveness_data, annot=True, fmt='.1f', ax=ax4,
                   cmap='RdBu_r', center=0, cbar_kws={'label': 'Avg Improvement (%)'})
        ax4.set_title('Average Improvement by Method and Ratio')
        ax4.set_xlabel('Augmentation Method')
        ax4.set_ylabel('Imbalance Ratio')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved improvement analysis to {save_path}")
        
        return fig
    
    def create_method_comparison(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create detailed method comparison plots"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Method Comparison Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: F1 keyword performance
        ax1 = axes[0, 0]
        sns.lineplot(data=self.summary_df, x='imbalance_ratio', y='f1_keyword_mean',
                    hue='augmentation_method', marker='o', ax=ax1)
        ax1.set_title('F1 Keyword Score vs Imbalance')
        ax1.set_xlabel('Imbalance Ratio')
        ax1.set_ylabel('F1 Keyword Score')
        
        # Plot 2: Balanced accuracy
        ax2 = axes[0, 1]
        sns.lineplot(data=self.summary_df, x='imbalance_ratio', y='balanced_accuracy_mean',
                    hue='augmentation_method', marker='s', ax=ax2)
        ax2.set_title('Balanced Accuracy vs Imbalance')
        ax2.set_xlabel('Imbalance Ratio')
        ax2.set_ylabel('Balanced Accuracy')
        
        # Plot 3: AUC-ROC
        ax3 = axes[0, 2]
        sns.lineplot(data=self.summary_df, x='imbalance_ratio', y='auc_roc_mean',
                    hue='augmentation_method', marker='^', ax=ax3)
        ax3.set_title('AUC-ROC vs Imbalance')
        ax3.set_xlabel('Imbalance Ratio')
        ax3.set_ylabel('AUC-ROC')
        
        # Plot 4: Performance by dataset size
        ax4 = axes[1, 0]
        size_order = ['small', 'medium', 'large', 'full']
        available_sizes = [s for s in size_order if s in self.summary_df['dataset_size'].unique()]
        
        sns.boxplot(data=self.summary_df, x='dataset_size', y='f1_keyword_mean',
                   hue='augmentation_method', ax=ax4, order=available_sizes)
        ax4.set_title('F1 Keyword by Dataset Size')
        ax4.set_xlabel('Dataset Size')
        ax4.set_ylabel('F1 Keyword Score')
        
        # Plot 5: Precision vs Recall trade-off
        ax5 = axes[1, 1]
        for method in self.summary_df['augmentation_method'].unique():
            method_data = self.summary_df[self.summary_df['augmentation_method'] == method]
            ax5.scatter(method_data['recall_keyword_mean'], method_data['precision_keyword_mean'],
                       label=method, alpha=0.7, s=50)
        
        ax5.set_xlabel('Recall (Keyword)')
        ax5.set_ylabel('Precision (Keyword)')
        ax5.set_title('Precision-Recall Trade-off')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Best method by condition
        ax6 = axes[1, 2]
        best_methods = self._find_best_methods_per_condition()
        method_counts = Counter(best_methods.values())
        
        methods = list(method_counts.keys())
        counts = list(method_counts.values())
        colors = sns.color_palette("Set2", len(methods))
        
        wedges, texts, autotexts = ax6.pie(counts, labels=methods, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax6.set_title('Best Method by Condition')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved method comparison to {save_path}")
        
        return fig
    
    def _find_best_methods_per_condition(self, metric: str = 'f1_keyword_mean') -> Dict[str, str]:
        """Find best performing method for each experimental condition"""
        best_methods = {}
        
        conditions = self.summary_df[['dataset_size', 'imbalance_ratio']].drop_duplicates()
        
        for _, condition in conditions.iterrows():
            condition_data = self.summary_df[
                (self.summary_df['dataset_size'] == condition['dataset_size']) &
                (self.summary_df['imbalance_ratio'] == condition['imbalance_ratio'])
            ]
            
            if len(condition_data) > 0:
                best_idx = condition_data[metric].idxmax()
                best_method = condition_data.loc[best_idx, 'augmentation_method']
                
                condition_key = f"{condition['dataset_size']}_ratio_{condition['imbalance_ratio']}"
                best_methods[condition_key] = best_method
        
        return best_methods
    
    def create_statistical_significance_plot(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create plot showing statistical significance of improvements"""
        from scipy import stats
        
        improvement_df = self.calculate_improvement_metrics()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Statistical Significance Analysis', fontsize=16, fontweight='bold')
        
        methods = [m for m in improvement_df['augmentation_method'].unique() if m != 'none']
        
        for i, method in enumerate(methods):
            if i >= 3:
                break
                
            ax = axes[i]
            method_data = improvement_df[improvement_df['augmentation_method'] == method]
            
            # Group by imbalance ratio
            significance_results = []
            
            for ratio in sorted(method_data['imbalance_ratio'].unique()):
                ratio_data = method_data[method_data['imbalance_ratio'] == ratio]
                improvements = ratio_data['relative_improvement'].values
                
                # One-sample t-test against zero improvement
                if len(improvements) > 1:
                    t_stat, p_value = stats.ttest_1samp(improvements, 0)
                    significance_results.append({
                        'ratio': ratio,
                        'mean_improvement': np.mean(improvements),
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    })
            
            if significance_results:
                sig_df = pd.DataFrame(significance_results)
                
                # Plot mean improvement with significance markers
                colors = ['green' if sig else 'red' for sig in sig_df['significant']]
                ax.bar(range(len(sig_df)), sig_df['mean_improvement'], color=colors, alpha=0.7)
                ax.set_xticks(range(len(sig_df)))
                ax.set_xticklabels([f"{r:.1f}" for r in sig_df['ratio']])
                ax.set_xlabel('Imbalance Ratio')
                ax.set_ylabel('Mean Improvement (%)')
                ax.set_title(f'{method.title()} Method\n(Green = p < 0.05)')
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved significance analysis to {save_path}")
        
        return fig
    
    def generate_publication_summary(self) -> str:
        """Generate summary suitable for paper results section"""
        
        improvement_df = self.calculate_improvement_metrics()
        
        # Key findings
        summary_stats = {}
        
        for method in improvement_df['augmentation_method'].unique():
            method_data = improvement_df[improvement_df['augmentation_method'] == method]
            
            summary_stats[method] = {
                'mean_improvement': method_data['relative_improvement'].mean(),
                'max_improvement': method_data['relative_improvement'].max(),
                'min_improvement': method_data['relative_improvement'].min(),
                'positive_cases': (method_data['relative_improvement'] > 0).sum(),
                'total_cases': len(method_data),
                'best_ratio': method_data.loc[
                    method_data['relative_improvement'].idxmax(), 'imbalance_ratio'
                ] if len(method_data) > 0 else None
            }
        
        # Generate report text
        report_lines = [
            "EXPERIMENTAL RESULTS SUMMARY",
            "=" * 50,
            f"Total experimental conditions: {len(improvement_df)}",
            f"Dataset versions: {self.results_df['dataset_size'].unique()}",
            f"Imbalance ratios tested: {sorted(self.results_df['imbalance_ratio'].unique())}",
            "",
            "KEY FINDINGS:",
        ]
        
        for method, stats in summary_stats.items():
            positive_pct = (stats['positive_cases'] / stats['total_cases']) * 100
            report_lines.extend([
                f"",
                f"{method.upper()} METHOD:",
                f"  - Average improvement: {stats['mean_improvement']:+.1f}%",
                f"  - Best case improvement: {stats['max_improvement']:+.1f}%", 
                f"  - Worst case: {stats['min_improvement']:+.1f}%",
                f"  - Helpful in {stats['positive_cases']}/{stats['total_cases']} conditions ({positive_pct:.1f}%)",
                f"  - Most effective at ratio: {stats['best_ratio']:.1f}" if stats['best_ratio'] else ""
            ])
        
        return "\n".join(report_lines)
    
    def save_all_analyses(self, output_dir: Optional[str] = None):
        """Generate and save all analysis plots and summaries"""
        
        if output_dir is None:
            output_dir = self.results_file.parent / 'analysis'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Generate all plots
        logger.info("Generating analysis plots...")
        
        heatmap_fig = self.create_performance_heatmap()
        heatmap_fig.savefig(output_dir / 'performance_heatmap.png', 
                          dpi=300, bbox_inches='tight')
        plt.close(heatmap_fig)
        
        improvement_fig = self.create_improvement_analysis()
        improvement_fig.savefig(output_dir / 'improvement_analysis.png',
                              dpi=300, bbox_inches='tight')
        plt.close(improvement_fig)
        
        comparison_fig = self.create_method_comparison()
        comparison_fig.savefig(output_dir / 'method_comparison.png',
                             dpi=300, bbox_inches='tight')
        plt.close(comparison_fig)
        
        significance_fig = self.create_statistical_significance_plot()
        significance_fig.savefig(output_dir / 'statistical_significance.png',
                                dpi=300, bbox_inches='tight')
        plt.close(significance_fig)
        
        # Save summary report
        summary_report = self.generate_publication_summary()
        with open(output_dir / 'summary_report.txt', 'w') as f:
            f.write(summary_report)
        
        # Save processed data
        improvement_df = self.calculate_improvement_metrics()
        improvement_df.to_csv(output_dir / 'improvement_metrics.csv', index=False)
        self.summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False)
        
        logger.info(f"All analyses saved to {output_dir}")
        
        return output_dir
