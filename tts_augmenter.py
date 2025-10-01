"""
TTS (Text-to-Speech) augmentation using pre-generated synthetic datasets.
"""
import torch
import numpy as np
import json
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TTSAugmenter:
    """Manages TTS-based data augmentation using pre-generated synthetic datasets"""
    
    def __init__(self, synthetic_dataset_path: Optional[str] = None):
        """
        Initialize TTS augmenter.
        
        Args:
            synthetic_dataset_path: Path to pre-generated synthetic dataset directory
        """
        self.synthetic_dataset_path = synthetic_dataset_path
        self.dataset_info = None
        self.audio_tensor = None
        self.metadata_df = None
        
        if synthetic_dataset_path:
            self._load_synthetic_dataset(synthetic_dataset_path)
    
    def _load_synthetic_dataset(self, dataset_path: str):
        """Load pre-generated synthetic dataset"""
        dataset_path = Path(dataset_path)
        
        try:
            # Validate required files exist
            required_files = ['dataset_info.json', 'synthetic_audio.pt', 'synthetic_metadata.csv']
            for file_name in required_files:
                file_path = dataset_path / file_name
                if not file_path.exists():
                    raise FileNotFoundError(f"Missing required file: {file_path}")
            
            # Load dataset components
            with open(dataset_path / 'dataset_info.json', 'r') as f:
                self.dataset_info = json.load(f)
            
            self.audio_tensor = torch.load(dataset_path / 'synthetic_audio.pt')
            self.metadata_df = pd.read_csv(dataset_path / 'synthetic_metadata.csv')
            
            logger.info(f"Loaded synthetic dataset: {self.dataset_info['total_samples']} samples")
            logger.info(f"Available keywords: {self.dataset_info['keywords']}")
            
        except Exception as e:
            logger.error(f"Failed to load synthetic dataset from {dataset_path}: {e}")
            raise RuntimeError(
                f"Could not load synthetic dataset. Please ensure you have run "
                f"'python synthetic_data_generator.py' to generate the dataset first. "
                f"Error: {e}"
            )
    
    def is_ready(self) -> bool:
        """Check if augmenter is ready to generate samples"""
        return all([
            self.dataset_info is not None,
            self.audio_tensor is not None,
            self.metadata_df is not None
        ])
    
    def validate_ready(self):
        """Raise error if augmenter is not ready"""
        if not self.is_ready():
            raise RuntimeError(
                "TTS augmenter not initialized with synthetic dataset. "
                "Please provide synthetic_dataset_path or call _load_synthetic_dataset()."
            )
    
    def get_available_keywords(self) -> List[str]:
        """Get list of keywords available in the synthetic dataset"""
        self.validate_ready()
        return self.dataset_info['keywords']
    
    def get_keyword_sample_count(self, keyword: str) -> int:
        """Get number of available samples for a keyword"""
        self.validate_ready()
        return len(self.metadata_df[self.metadata_df['keyword'] == keyword])
    
    def sample_keyword_data(self, keyword: str, n_samples: int,
                          random_state: Optional[int] = None) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Sample synthetic data for a specific keyword.
        
        Args:
            keyword: Target keyword to sample
            n_samples: Number of samples to generate
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of (audio_samples, labels)
        """
        self.validate_ready()
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Filter metadata for target keyword
        keyword_metadata = self.metadata_df[self.metadata_df['keyword'] == keyword]
        
        if len(keyword_metadata) == 0:
            logger.error(f"No synthetic samples found for keyword: {keyword}")
            logger.info(f"Available keywords: {self.get_available_keywords()}")
            return [], []
        
        # Determine number of samples to draw
        available_samples = len(keyword_metadata)
        n_samples_to_draw = min(n_samples, available_samples)
        
        if n_samples_to_draw < n_samples:
            logger.warning(
                f"Requested {n_samples} samples for '{keyword}', but only "
                f"{available_samples} available. Using {n_samples_to_draw} samples."
            )
        
        # Randomly sample indices
        sampled_indices = np.random.choice(
            keyword_metadata.index,
            n_samples_to_draw,
            replace=False
        )
        
        # Extract audio samples
        sampled_audio = [self.audio_tensor[idx] for idx in sampled_indices]
        sampled_labels = ['keyword'] * len(sampled_audio)
        
        logger.info(f"Sampled {len(sampled_audio)} TTS samples for keyword '{keyword}'")
        
        return sampled_audio, sampled_labels
    
    def get_balanced_synthetic_samples(self, keywords: List[str],
                                     samples_per_keyword: int,
                                     random_state: Optional[int] = None) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Get balanced synthetic samples across multiple keywords.
        
        Args:
            keywords: List of target keywords
            samples_per_keyword: Number of samples per keyword
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of (audio_samples, labels)
        """
        self.validate_ready()
        
        if random_state is not None:
            np.random.seed(random_state)
        
        all_audio = []
        all_labels = []
        
        # Sample each keyword
        for keyword in keywords:
            audio, labels = self.sample_keyword_data(
                keyword,
                samples_per_keyword,
                random_state
            )
            all_audio.extend(audio)
            all_labels.extend(labels)
        
        if len(all_audio) == 0:
            logger.error("Failed to generate any synthetic samples")
            return [], []
        
        # Shuffle combined dataset
        indices = list(range(len(all_audio)))
        np.random.shuffle(indices)
        
        shuffled_audio = [all_audio[i] for i in indices]
        shuffled_labels = [all_labels[i] for i in indices]
        
        logger.info(f"Generated {len(shuffled_audio)} balanced TTS samples across {len(keywords)} keywords")
        
        return shuffled_audio, shuffled_labels
    
    def generate_tts_samples(self, keywords: List[str], n_samples: int,
                           random_state: Optional[int] = None) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Generate TTS samples for augmentation (main interface).
        
        Args:
            keywords: List of target keywords
            n_samples: Total number of samples to generate
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of (audio_samples, labels)
        """
        self.validate_ready()
        
        if n_samples <= 0:
            return [], []
        
        # Distribute samples evenly across keywords
        samples_per_keyword = max(1, n_samples // len(keywords))
        
        logger.info(f"TTS augmentation: generating {samples_per_keyword} samples per keyword")
        
        return self.get_balanced_synthetic_samples(
            keywords,
            samples_per_keyword,
            random_state
        )
    
    def get_dataset_statistics(self) -> Dict:
        """Get statistics about the loaded synthetic dataset"""
        self.validate_ready()
        
        stats = {
            'total_samples': len(self.metadata_df),
            'samples_per_keyword': self.metadata_df['keyword'].value_counts().to_dict(),
            'unique_text_variants': self.metadata_df['text_variant'].nunique(),
            'unique_keywords': len(self.dataset_info['keywords']),
            'audio_quality': {
                'mean_energy': self.metadata_df['audio_energy'].mean(),
                'std_energy': self.metadata_df['audio_energy'].std(),
                'mean_max_amplitude': self.metadata_df['audio_max_amplitude'].mean(),
            }
        }
        
        return stats
    
    def validate_synthetic_samples(self, audio_samples: List[torch.Tensor],
                                  labels: List[str]) -> bool:
        """
        Validate that synthetic samples are properly formatted.
        
        Args:
            audio_samples: List of audio tensors
            labels: List of corresponding labels
        
        Returns:
            True if validation passes, False otherwise
        """
        if len(audio_samples) != len(labels):
            logger.error("Mismatch between audio samples and labels")
            return False
        
        # Check audio validity
        invalid_count = 0
        for i, audio in enumerate(audio_samples):
            if not isinstance(audio, torch.Tensor):
                invalid_count += 1
                logger.warning(f"Sample {i}: not a torch.Tensor")
            elif audio.dim() != 2 or audio.shape[0] != 1:
                invalid_count += 1
                logger.warning(f"Sample {i}: invalid shape {audio.shape}")
            elif torch.isnan(audio).any() or torch.isinf(audio).any():
                invalid_count += 1
                logger.warning(f"Sample {i}: contains NaN or Inf values")
        
        if invalid_count > 0:
            logger.error(f"Found {invalid_count} invalid audio samples")
            return False
        
        # Check label validity
        valid_labels = {'keyword', 'non_keyword'}
        invalid_labels = [label for label in labels if label not in valid_labels]
        
        if invalid_labels:
            logger.error(f"Found invalid labels: {set(invalid_labels)}")
            return False
        
        logger.info("Synthetic samples validation passed")
        return True
