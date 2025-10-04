"""
Unified augmentation management coordinating TTS and adversarial methods.
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
import logging
import numpy as np

from synthetic_data_generator import SyntheticDatasetLoader
from adversarial_augmenter import AdversarialAugmenter

logger = logging.getLogger(__name__)


class AugmentationManager:
    """
    Orchestrates all augmentation methods for class imbalance experiments.
    
    Manages:
    - TTS (synthetic) augmentation using pre-generated datasets
    - Adversarial (FGSM) augmentation
    - Combined augmentation strategies
    """
    
    def __init__(self, fgsm_epsilon: float = 0.01,
                 synthetic_dataset_path: Optional[str] = None):
        """
        Initialize augmentation manager.
        
        Args:
            fgsm_epsilon: Epsilon parameter for FGSM adversarial generation
            synthetic_dataset_path: Path to pre-generated synthetic TTS dataset
        """
        self.fgsm_epsilon = fgsm_epsilon
        
        # Initialize augmentation methods
        self.synthetic_loader = None
        if synthetic_dataset_path:
            try:
                self.synthetic_loader = SyntheticDatasetLoader(synthetic_dataset_path)
                logger.info(f"Loaded synthetic dataset: {self.synthetic_loader.info['total_samples']} samples")
            except Exception as e:
                logger.error(f"Failed to load synthetic dataset: {e}")
                raise
        
        self.adversarial_augmenter = AdversarialAugmenter(fgsm_epsilon)
        
        logger.info("AugmentationManager initialized")
        logger.info(f"  - Synthetic dataset: {'ready' if self.synthetic_loader else 'not loaded'}")
        logger.info(f"  - Adversarial augmenter: epsilon={fgsm_epsilon}")
    
    def set_adversarial_model(self, model: nn.Module):
        """
        Set target model for adversarial generation.
        
        Args:
            model: Trained PyTorch model
        """
        self.adversarial_augmenter.set_target_model(model)
        logger.info("Set adversarial target model")
    
    def calculate_augmentation_requirements(self, labels: List[str],
                                          target_ratio: float) -> Dict[str, int]:
        """
        Calculate number of samples needed to balance the training set.
        
        Note: target_ratio is the IMBALANCE that was created in the dataset,
        not the target after augmentation. Augmentation aims to achieve 1:1 balance.
        
        Args:
            labels: List of current labels
            target_ratio: Current imbalance ratio (minority:majority)
        
        Returns:
            Dictionary with augmentation requirements
        """
        n_positive = sum(1 for label in labels if label == 'keyword')
        n_negative = sum(1 for label in labels if label == 'non_keyword')
        
        # For augmentation: balance the training set to 1:1 ratio
        target_positive = n_negative
        samples_needed = max(0, target_positive - n_positive)
        
        current_ratio = n_positive / n_negative if n_negative > 0 else float('inf')
        
        return {
            'current_positive': n_positive,
            'current_negative': n_negative,
            'target_positive': target_positive,
            'samples_needed': samples_needed,
            'current_ratio': current_ratio
        }
    
    
    def apply_tts_augmentation(self, keywords, n_samples, random_state):
        if n_samples <= 0:
            return [], []
    
        # Distribute samples across keywords, handling remainder
        base_samples = n_samples // len(keywords)
        remainder = n_samples % len(keywords)
    
        all_audio = []
        all_labels = []
    
        for i, keyword in enumerate(keywords):
            # Give remainder samples to first few keywords
            samples_this_keyword = base_samples + (1 if i < remainder else 0)
        
            audio, labels = self.synthetic_loader.sample_keyword_data(
                keyword, 
                samples_this_keyword, 
                random_state
            )
            all_audio.extend(audio)
            all_labels.extend(labels)
    
        logger.info(f"TTS augmentation: generated {len(all_audio)} samples (requested {n_samples})")
    
        return all_audio, all_labels


    def apply_adversarial_augmentation(self, audio_files: List[torch.Tensor],
                                     labels: List[str], n_samples: int
                                     ) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Apply adversarial (FGSM) augmentation.
        
        Args:
            audio_files: Original audio samples
            labels: Original labels
            n_samples: Number of adversarial samples to generate
        
        Returns:
            Tuple of (audio_samples, labels)
        """
        if n_samples <= 0:
            return [], []
        
        if not self.adversarial_augmenter.is_ready():
            raise RuntimeError(
                "Adversarial augmenter not ready. No target model set. "
                "Call set_adversarial_model() first."
            )
        
        logger.info(f"Applying adversarial augmentation: {n_samples} samples")
        
        return self.adversarial_augmenter.generate_adversarial_samples(
            audio_files,
            labels,
            n_samples
        )
    
    def apply_combined_augmentation(self, audio_files: List[torch.Tensor],
                                  labels: List[str], keywords: List[str],
                                  n_samples: int,
                                  random_state: Optional[int] = None,
                                  tts_ratio: float = 0.5
                                  ) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Apply combined TTS + adversarial augmentation.
        
        Args:
            audio_files: Original audio samples
            labels: Original labels
            keywords: List of target keywords
            n_samples: Total number of samples to generate
            random_state: Random seed for reproducibility
            tts_ratio: Proportion of samples to generate with TTS (default 0.5)
        
        Returns:
            Tuple of (audio_samples, labels)
        """
        if n_samples <= 0:
            return [], []
        
        # Split samples between methods
        n_tts = int(n_samples * tts_ratio)
        n_adversarial = n_samples - n_tts
        
        logger.info(
            f"Applying combined augmentation: {n_tts} TTS + "
            f"{n_adversarial} adversarial = {n_samples} total"
        )
        
        # Generate TTS samples
        tts_audio, tts_labels = self.apply_tts_augmentation(
            keywords,
            n_tts,
            random_state
        )
        
        # Generate adversarial samples
        adv_audio, adv_labels = self.apply_adversarial_augmentation(
            audio_files,
            labels,
            n_adversarial
        )
        
        # Combine samples
        combined_audio = tts_audio + adv_audio
        combined_labels = tts_labels + adv_labels
        
        logger.info(
            f"Combined augmentation generated {len(combined_audio)} samples "
            f"({len(tts_audio)} TTS + {len(adv_audio)} adversarial)"
        )
        
        return combined_audio, combined_labels
    
    def apply_augmentation(self, audio_files: List[torch.Tensor],
                         labels: List[str], method: str,
                         keywords: List[str], target_ratio: float,
                         random_state: Optional[int] = None
                         ) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Apply specified augmentation method (main interface).
        
        Args:
            audio_files: Original audio samples
            labels: Original labels
            method: Augmentation method ('none', 'synthetic', 'tts', 'adversarial', 'combined')
            keywords: List of target keywords
            target_ratio: Current imbalance ratio
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of (augmented_audio, augmented_labels)
        """
        # Calculate augmentation requirements
        requirements = self.calculate_augmentation_requirements(labels, target_ratio)
        samples_needed = requirements['samples_needed']
        
        logger.info(
            f"Augmentation requirements for method '{method}': "
            f"{samples_needed} samples needed "
            f"(current: {requirements['current_positive']} pos, "
            f"{requirements['current_negative']} neg, "
            f"ratio: {requirements['current_ratio']:.3f})"
        )
        
        # No augmentation needed if already balanced
        if samples_needed <= 0:
            logger.info("Dataset already balanced, no augmentation needed")
            return audio_files, labels
        
        # Apply appropriate augmentation method
        if method == 'none':
            logger.info("Method 'none': skipping augmentation")
            return audio_files, labels
        
        elif method in ['synthetic', 'tts']:  # Accept both names
            aug_audio, aug_labels = self.apply_tts_augmentation(
                keywords,
                samples_needed,
                random_state
            )
        
        elif method == 'adversarial':
            aug_audio, aug_labels = self.apply_adversarial_augmentation(
                audio_files,
                labels,
                samples_needed
            )
        
        elif method == 'combined':
            aug_audio, aug_labels = self.apply_combined_augmentation(
                audio_files,
                labels,
                keywords,
                samples_needed,
                random_state
            )
        
        else:
            raise ValueError(f"Unknown augmentation method: {method}")
        
        # Combine with original data
        combined_audio = audio_files + aug_audio
        combined_labels = labels + aug_labels
        
        # Log final statistics
        final_pos = sum(1 for label in combined_labels if label == 'keyword')
        final_neg = sum(1 for label in combined_labels if label == 'non_keyword')
        final_ratio = final_pos / final_neg if final_neg > 0 else float('inf')
        
        logger.info(f"Augmentation complete:")
        logger.info(f"  Original: {len(audio_files)} samples")
        logger.info(f"  Added: {len(aug_audio)} augmented samples")
        logger.info(f"  Final: {len(combined_audio)} samples "
                   f"({final_pos} pos, {final_neg} neg, ratio: {final_ratio:.3f})")
        
        return combined_audio, combined_labels
    
    def validate_augmented_dataset(self, audio_files: List[torch.Tensor],
                                 labels: List[str]) -> bool:
        """
        Validate that augmented dataset is properly formatted.
        
        Args:
            audio_files: List of audio tensors
            labels: List of corresponding labels
        
        Returns:
            True if validation passes, False otherwise
        """
        if len(audio_files) != len(labels):
            logger.error(f"Mismatch between audio files ({len(audio_files)}) and labels ({len(labels)})")
            return False
        
        # Check audio validity
        invalid_count = 0
        for i, audio in enumerate(audio_files):
            if not isinstance(audio, torch.Tensor):
                invalid_count += 1
                if invalid_count <= 5:  # Log first 5 errors
                    logger.warning(f"Sample {i}: not a torch.Tensor")
            elif audio.dim() != 2 or audio.shape[0] != 1:
                invalid_count += 1
                if invalid_count <= 5:
                    logger.warning(f"Sample {i}: invalid shape {audio.shape}")
            elif torch.isnan(audio).any() or torch.isinf(audio).any():
                invalid_count += 1
                if invalid_count <= 5:
                    logger.warning(f"Sample {i}: contains NaN or Inf values")
        
        if invalid_count > 0:
            logger.error(f"Found {invalid_count} invalid audio samples")
            return False
        
        # Check label validity
        valid_labels = {'keyword', 'non_keyword'}
        invalid_labels = [label for label in labels if label not in valid_labels]
        
        if invalid_labels:
            logger.error(f"Found {len(invalid_labels)} invalid labels: {set(invalid_labels)}")
            return False
        
        logger.info("Augmented dataset validation passed")
        return True
    
    def get_augmentation_statistics(self) -> Dict:
        """
        Get statistics about available augmentation methods.
        
        Returns:
            Dictionary with augmentation statistics
        """
        stats = {
            'synthetic_augmentation': {
                'ready': self.synthetic_loader is not None,
                'total_samples': None,
                'samples_per_keyword': None
            },
            'adversarial_augmentation': {
                'ready': self.adversarial_augmenter.is_ready(),
                'epsilon': self.fgsm_epsilon
            }
        }
        
        # Add synthetic dataset statistics if available
        if self.synthetic_loader:
            try:
                loader_stats = self.synthetic_loader.get_dataset_statistics()
                stats['synthetic_augmentation']['total_samples'] = loader_stats.get('total_samples')
                stats['synthetic_augmentation']['samples_per_keyword'] = loader_stats.get('samples_per_keyword')
            except Exception as e:
                logger.warning(f"Could not get synthetic dataset statistics: {e}")
        
        return stats
