"""
Tools for inspecting and visualizing synthetic datasets.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
from IPython.display import Audio, display

from synthetic_data_generator import SyntheticDatasetLoader

def get_dataset_statistics(dataset_path: str) -> Dict:
    """
    Get comprehensive statistics about a synthetic dataset.
    
    Args:
        dataset_path: Path to synthetic dataset
        
    Returns:
        Dictionary of statistics
    """
    loader = SyntheticDatasetLoader(dataset_path)
    stats = loader.get_dataset_statistics()
    
    return stats

def print_dataset_summary(dataset_path: str):
    """
    Print a formatted summary of dataset statistics.
    
    Args:
        dataset_path: Path to synthetic dataset
    """
    loader = SyntheticDatasetLoader(dataset_path)
    stats = loader.get_dataset_statistics()
    
    print("SYNTHETIC DATASET STATISTICS")
    print("="*50)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Samples per keyword: {stats['samples_per_keyword']}")
    print(f"Unique text variants: {stats['unique_text_variants']}")
    print(f"Base samples: {stats['base_samples']}")
    print(f"Variation samples: {stats['variation_samples']}")
    print(f"\nAudio Quality:")
    print(f"  Mean energy: {stats['audio_quality']['mean_energy']:.4f}")
    print(f"  Mean max amplitude: {stats['audio_quality']['mean_max_amplitude']:.4f}")

def show_metadata_sample(dataset_path: str, n_rows: int = 10):
    """
    Display sample metadata from the dataset.
    
    Args:
        dataset_path: Path to synthetic dataset
        n_rows: Number of rows to display
    """
    loader = SyntheticDatasetLoader(dataset_path)
    
    print("\nMETADATA SAMPLE")
    print("="*50)
    print(loader.metadata_df.head(n_rows))

def play_audio_samples(dataset_path: str, keywords: List[str], 
                       samples_per_keyword: int = 3):
    """
    Play audio samples for inspection in Jupyter/Colab.
    
    Args:
        dataset_path: Path to synthetic dataset
        keywords: List of keywords to play samples for
        samples_per_keyword: Number of samples to play per keyword
    """
    loader = SyntheticDatasetLoader(dataset_path)
    
    print("\nAUDIO SAMPLES (Click to play)")
    print("="*50)
    
    for keyword in keywords:
        print(f"\n--- Keyword: '{keyword}' ---")
        
        # Get random samples for this keyword
        keyword_meta = loader.metadata_df[loader.metadata_df['keyword'] == keyword]
        
        if len(keyword_meta) == 0:
            print(f"  No samples found for '{keyword}'")
            continue
            
        sample_indices = keyword_meta.sample(
            min(samples_per_keyword, len(keyword_meta))
        ).index.tolist()
        
        for i, idx in enumerate(sample_indices):
            row = loader.metadata_df.loc[idx]
            audio_sample = loader.audio_tensor[idx]
            
            print(f"\nSample {i+1}:")
            print(f"  Text variant: '{row['text_variant']}'")
            print(f"  Is base: {row['is_base_sample']}")
            print(f"  Energy: {row['audio_energy']:.4f}")
            
            # Convert to numpy for playback
            audio_np = audio_sample.squeeze().numpy()
            display(Audio(audio_np, rate=16000))

def visualize_waveforms(dataset_path: str, keywords: List[str], 
                       figsize: Tuple[int, int] = (12, 8)):
    """
    Visualize waveforms for sample audio from each keyword.
    
    Args:
        dataset_path: Path to synthetic dataset
        keywords: List of keywords to visualize
        figsize: Figure size (width, height)
    """
    loader = SyntheticDatasetLoader(dataset_path)
    
    n_keywords = len(keywords)
    rows = (n_keywords + 1) // 2
    cols = 2
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle('Synthetic Sample Waveforms', fontsize=14)
    
    # Flatten axes for easier iteration
    if n_keywords == 1:
        axes = [axes]
    elif n_keywords <= 2:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    for i, keyword in enumerate(keywords):
        ax = axes[i]
        
        # Get one sample for this keyword
        keyword_meta = loader.metadata_df[loader.metadata_df['keyword'] == keyword]
        
        if len(keyword_meta) == 0:
            ax.text(0.5, 0.5, f"No samples for '{keyword}'", 
                   ha='center', va='center')
            ax.set_title(f"Keyword: '{keyword}'")
            continue
            
        idx = keyword_meta.sample(1).index[0]
        
        audio_sample = loader.audio_tensor[idx]
        audio_np = audio_sample.squeeze().numpy()
        
        ax.plot(audio_np)
        ax.set_title(f"Keyword: '{keyword}'")
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_keywords, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def show_text_variant_distribution(dataset_path: str, top_n: int = 20):
    """
    Show distribution of text variants in the dataset.
    
    Args:
        dataset_path: Path to synthetic dataset
        top_n: Number of top variants to show
    """
    loader = SyntheticDatasetLoader(dataset_path)
    
    print("\nTEXT VARIANT DISTRIBUTION")
    print("="*50)
    variant_counts = loader.metadata_df['text_variant'].value_counts()
    print(variant_counts.head(top_n))

def inspect_dataset_quality(dataset_path: str, keywords: List[str]):
    """
    Comprehensive quality inspection of a synthetic dataset.
    
    Args:
        dataset_path: Path to synthetic dataset
        keywords: Keywords to inspect
    """
    # Print statistics
    print_dataset_summary(dataset_path)
    
    # Show metadata
    show_metadata_sample(dataset_path, n_rows=5)
    
    # Show text variants
    show_text_variant_distribution(dataset_path, top_n=15)
    
    # Visualize waveforms
    print("\nGenerating waveform visualizations...")
    fig = visualize_waveforms(dataset_path, keywords)
    plt.show()
    
    print("\nQuality inspection complete!")
