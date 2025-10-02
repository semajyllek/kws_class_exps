"""
Loader for pre-generated synthetic datasets.
"""
import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SyntheticDatasetLoader:
    """Loads pre-generated synthetic datasets."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize loader for synthetic dataset.
        
        Args:
            dataset_path: Path to synthetic dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self._validate_files()
        
        with open(self.dataset_path / 'dataset_info.json', 'r') as f:
            self.info = json.load(f)
        
        self.audio_tensor = torch.load(self.dataset_path / 'synthetic_audio.pt')
        self.metadata_df = pd.read_csv(self.dataset_path / 'synthetic_metadata.csv')
        
        logger.info(f"Loaded dataset: {self.info['total_samples']} samples")
    
    def _validate_files(self):
        """Validate required files exist."""
        required = ['dataset_info.json', 'synthetic_audio.pt', 'synthetic_metadata.csv']
        for file_name in required:
            if not (self.dataset_path / file_name).exists():
                raise FileNotFoundError(f"Missing: {self.dataset_path / file_name}")
    
    def sample_keyword_data(self, keyword: str, n_samples: int, 
                           random_state: Optional[int] = None) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Sample synthetic data for a keyword.
        
        Args:
            keyword: Target keyword to sample
            n_samples: Number of samples to draw
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of (audio_samples, labels)
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        keyword_meta = self.metadata_df[self.metadata_df['keyword'] == keyword]
        
        if len(keyword_meta) == 0:
            logger.error(f"No samples for keyword: {keyword}")
            return [], []
        
        n_samples = min(n_samples, len(keyword_meta))
        sampled_indices = np.random.choice(keyword_meta.index, n_samples, replace=False)
        
        sampled_audio = [self.audio_tensor[idx] for idx in sampled_indices]
        sampled_labels = ['keyword'] * len(sampled_audio)
        
        return sampled_audio, sampled_labels
    
    def get_balanced_samples(self, keywords: List[str], samples_per_keyword: int,
                           random_state: Optional[int] = None) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Get balanced samples across keywords.
        
        Args:
            keywords: List of keywords to sample
            samples_per_keyword: Number of samples per keyword
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of (audio_samples, labels)
        """
        all_audio = []
        all_labels = []
        
        for keyword in keywords:
            audio, labels = self.sample_keyword_data(keyword, samples_per_keyword, random_state)
            all_audio.extend(audio)
            all_labels.extend(labels)
        
        if random_state is not None:
            np.random.seed(random_state)
        
        indices = list(range(len(all_audio)))
        np.random.shuffle(indices)
        
        return [all_audio[i] for i in indices], [all_labels[i] for i in indices]
    
    def get_dataset_statistics(self) -> Dict:
        """
        Get statistics about the loaded synthetic dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        return {
            'total_samples': len(self.metadata_df),
            'samples_per_keyword': self.metadata_df['keyword'].value_counts().to_dict(),
            'unique_text_variants': self.metadata_df['text_variant'].nunique(),
            'base_samples': len(self.metadata_df[self.metadata_df['is_base_sample'] == True]),
            'audio_quality': {
                'mean_energy': self.metadata_df['audio_energy'].mean(),
                'mean_max_amplitude': self.metadata_df['audio_max_amplitude'].mean()
            }
        }


def check_dataset_exists(dataset_path: str) -> Dict:
    """
    Check if dataset exists and return info.
    
    Args:
        dataset_path: Path to dataset directory
    
    Returns:
        Dictionary with 'exists' bool and 'info' (dataset info dict or None)
    """
    info_file = Path(dataset_path) / 'dataset_info.json'
    
    if info_file.exists():
        with open(info_file, 'r') as f:
            return {'exists': True, 'info': json.load(f)}
    return {'exists': False, 'info': None}
