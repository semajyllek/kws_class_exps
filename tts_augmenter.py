"""
TTS (Text-to-Speech) augmentation using pre-generated synthetic datasets.
"""
import torch
from typing import List, Tuple, Dict, Optional
import logging

from synthetic_data_loader import SyntheticDatasetLoader

logger = logging.getLogger(__name__)


class TTSAugmenter:
    """
    Manages TTS-based data augmentation using pre-generated synthetic datasets.
    
    This is a thin wrapper around SyntheticDatasetLoader that provides
    an interface matching AdversarialAugmenter for consistency.
    """
    
    def __init__(self, synthetic_dataset_path: Optional[str] = None):
        """
        Initialize TTS augmenter.
        
        Args:
            synthetic_dataset_path: Path to pre-generated synthetic dataset directory
        """
        self.synthetic_dataset_path = synthetic_dataset_path
        self.loader = None
        
        if synthetic_dataset_path:
            self._load_synthetic_dataset(synthetic_dataset_path)
    
    def _load_synthetic_dataset(self, dataset_path: str):
        """Load pre-generated synthetic dataset"""
        try:
            self.loader = SyntheticDatasetLoader(dataset_path)
            logger.info(f"Loaded synthetic dataset: {self.loader.info['total_samples']} samples")
            logger.info(f"Available keywords: {self.loader.info['keywords']}")
        except Exception as e:
            logger.error(f"Failed to load synthetic dataset from {dataset_path}: {e}")
            raise RuntimeError(
                f"Could not load synthetic dataset. Please ensure you have run "
                f"'python synthetic_data_generator.py' to generate the dataset first. "
                f"Error: {e}"
            )
    
    def is_ready(self) -> bool:
        """Check if augmenter is ready to generate samples"""
        return self.loader is not None
    
    def validate_ready(self):
        """Raise error if augmenter is not ready"""
        if not self.is_ready():
            raise RuntimeError(
                "TTS augmenter not initialized with synthetic dataset. "
                "Please provide synthetic_dataset_path or call _load_synthetic_dataset()."
            )
    
    def generate_tts_samples(self, keywords: List[str], n_samples: int,
                           random_state: Optional[int] = None) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Generate TTS samples for augmentation (main interface matching AdversarialAugmenter).
        
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
        
        return self.loader.get_balanced_samples(
            keywords,
            samples_per_keyword,
            random_state
        )
    
    def get_dataset_statistics(self) -> Dict:
        """Get statistics about the loaded synthetic dataset"""
        self.validate_ready()
        return self.loader.get_dataset_statistics()
    
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
