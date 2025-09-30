"""
Data augmentation management using pre-generated synthetic datasets.
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
import logging
import numpy as np

from adversarial_augmenter import AdversarialAugmenter

logger = logging.getLogger(__name__)

class SyntheticDatasetLoader:
    """Simple loader for pre-generated synthetic data"""
    
    def __init__(self, dataset_path: str):
        import json
        import pandas as pd
        from pathlib import Path
        
        self.dataset_path = Path(dataset_path)
        
        # Load dataset components
        with open(self.dataset_path / 'dataset_info.json', 'r') as f:
            self.info = json.load(f)
        
        self.audio_tensor = torch.load(self.dataset_path / 'synthetic_audio.pt')
        self.metadata_df = pd.read_csv(self.dataset_path / 'synthetic_metadata.csv')
        
        logger.info(f"Loaded synthetic dataset: {self.info['total_samples']} samples")
    
    def get_balanced_synthetic_samples(self, keywords: List[str], 
                                     samples_per_keyword: int,
                                     random_state: Optional[int] = None) -> Tuple[List[torch.Tensor], List[str]]:
        """Get balanced synthetic samples across keywords"""
        
        if random_state is not None:
            np.random.seed(random_state)
        
        all_audio = []
        all_labels = []
        
        for keyword in keywords:
            # Get samples for this keyword
            keyword_metadata = self.metadata_df[self.metadata_df['keyword'] == keyword]
            
            if len(keyword_metadata) == 0:
                logger.error(f"No synthetic samples found for keyword: {keyword}")
                continue
            
            # Sample indices
            available_samples = len(keyword_metadata)
            n_samples = min(samples_per_keyword, available_samples)
            
            sampled_indices = np.random.choice(
                keyword_metadata.index, n_samples, replace=False
            )
            
            # Extract audio samples
            sampled_audio = [self.audio_tensor[idx] for idx in sampled_indices]
            sampled_labels = ['keyword'] * len(sampled_audio)
            
            all_audio.extend(sampled_audio)
            all_labels.extend(sampled_labels)
        
        # Shuffle combined dataset
        indices = list(range(len(all_audio)))
        np.random.shuffle(indices)
        
        shuffled_audio = [all_audio[i] for i in indices]
        shuffled_labels = [all_labels[i] for i in indices]
        
        return shuffled_audio, shuffled_labels

class AugmentationManager:
    """Orchestrates augmentation using pre-generated synthetic data and adversarial samples"""
    
    def __init__(self, fgsm_epsilon: float, synthetic_dataset_path: Optional[str] = None):
        self.adversarial_augmenter = AdversarialAugmenter(fgsm_epsilon)
        
        # Load pre-generated synthetic dataset if available
        self.synthetic_loader = None
        if synthetic_dataset_path:
            self._load_synthetic_dataset(synthetic_dataset_path)
    
    def _load_synthetic_dataset(self, dataset_path: str):
        """Load pre-generated synthetic dataset"""
        try:
            self.synthetic_loader = SyntheticDatasetLoader(dataset_path)
            logger.info("Successfully loaded synthetic dataset")
        except Exception as e:
            logger.error(f"Failed to load synthetic dataset: {e}")
            self.synthetic_loader = None
    
    def set_adversarial_model(self, model: nn.Module):
        """Set target model for adversarial generation"""
        self.adversarial_augmenter.set_target_model(model)
    
   
    def calculate_augmentation_requirements(self, labels: List[str], 
                                      target_ratio: float) -> Dict[str, int]:
        """
        Calculate samples needed to balance training set.
        Note: target_ratio is the IMBALANCE that was created, not the target after augmentation.
        """
        n_positive = sum(1 for label in labels if label == 'keyword')
        n_negative = sum(1 for label in labels if label == 'non_keyword')
    
        # For augmentation methods: balance the training set (bring to 1:1)
        target_positive = n_negative
        samples_needed = max(0, target_positive - n_positive)
    
        return {
            'current_positive': n_positive,
            'current_negative': n_negative,
            'target_positive': target_positive,
            'samples_needed': samples_needed,
            'current_ratio': n_positive / n_negative if n_negative > 0 else float('inf')
        }

 
    def apply_synthetic_augmentation(self, keywords: List[str], 
                                   n_samples: int, 
                                   random_state: Optional[int] = None) -> Tuple[List[torch.Tensor], List[str]]:
        """Apply synthetic (TTS) augmentation using pre-generated data"""
        
        if self.synthetic_loader is None:
            raise RuntimeError("No synthetic dataset loaded. Generate one first using synthetic_data_generator.py")
        
        if n_samples <= 0:
            return [], []
        
        samples_per_keyword = max(1, n_samples // len(keywords))
        logger.info(f"Synthetic augmentation: {samples_per_keyword} samples per keyword")
        
        return self.synthetic_loader.get_balanced_synthetic_samples(
            keywords, samples_per_keyword, random_state
        )
    
    def apply_adversarial_augmentation(self, audio_files: List[torch.Tensor],
                                     labels: List[str], 
                                     n_samples: int) -> Tuple[List[torch.Tensor], List[str]]:
        """Apply adversarial augmentation"""
        if n_samples <= 0:
            return [], []
        
        logger.info(f"Adversarial augmentation: {n_samples} samples")
        return self.adversarial_augmenter.generate_adversarial_samples(
            audio_files, labels, n_samples
        )
    
    def apply_combined_augmentation(self, audio_files: List[torch.Tensor],
                                  labels: List[str], keywords: List[str],
                                  n_samples: int, 
                                  random_state: Optional[int] = None) -> Tuple[List[torch.Tensor], List[str]]:
        """Apply combined synthetic + adversarial augmentation"""
        if n_samples <= 0:
            return [], []
        
        # Split samples between methods (50/50)
        n_adversarial = n_samples // 2
        n_synthetic = n_samples - n_adversarial
        
        logger.info(f"Combined augmentation: {n_adversarial} adversarial + {n_synthetic} synthetic")
        
        # Generate adversarial samples
        adv_audio, adv_labels = self.apply_adversarial_augmentation(
            audio_files, labels, n_adversarial
        )
        
        # Generate synthetic samples
        synth_audio, synth_labels = self.apply_synthetic_augmentation(
            keywords, n_synthetic, random_state
        )
        
        # Combine
        combined_audio = adv_audio + synth_audio
        combined_labels = adv_labels + synth_labels
        
        return combined_audio, combined_labels
    
    def apply_augmentation(self, audio_files: List[torch.Tensor], 
                         labels: List[str], method: str,
                         keywords: List[str], target_ratio: float,
                         random_state: Optional[int] = None) -> Tuple[List[torch.Tensor], List[str]]:
        """Apply specified augmentation method"""
        
        # Calculate requirements
        requirements = self.calculate_augmentation_requirements(labels, target_ratio)
        samples_needed = requirements['samples_needed']
        
        logger.info(f"Augmentation requirements: {requirements}")
        
        if samples_needed <= 0:
            logger.info("Dataset already balanced, no augmentation needed")
            return audio_files, labels
        
        # Apply appropriate method
        if method == 'none':
            return audio_files, labels
            
        elif method == 'synthetic':
            aug_audio, aug_labels = self.apply_synthetic_augmentation(
                keywords, samples_needed, random_state
            )
            
        elif method == 'adversarial':
            aug_audio, aug_labels = self.apply_adversarial_augmentation(
                audio_files, labels, samples_needed
            )
            
        elif method == 'combined':
            aug_audio, aug_labels = self.apply_combined_augmentation(
                audio_files, labels, keywords, samples_needed, random_state
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
        
        logger.info(f"Final dataset after augmentation:")
        logger.info(f"  {final_pos} positive, {final_neg} negative (ratio: {final_ratio:.3f})")
        logger.info(f"  Added {len(aug_audio)} augmented samples")
        
        return combined_audio, combined_labels
    
    def validate_augmented_dataset(self, audio_files: List[torch.Tensor], 
                                 labels: List[str]) -> bool:
        """Validate that augmented dataset is properly formatted"""
        
        if len(audio_files) != len(labels):
            logger.error("Mismatch between audio files and labels")
            return False
        
        # Check audio validity
        invalid_count = 0
        for i, audio in enumerate(audio_files):
            if not isinstance(audio, torch.Tensor):
                invalid_count += 1
            elif audio.dim() != 2 or audio.shape[0] != 1:
                invalid_count += 1
            elif torch.isnan(audio).any() or torch.isinf(audio).any():
                invalid_count += 1
        
        if invalid_count > 0:
            logger.error(f"Found {invalid_count} invalid audio samples")
            return False
        
        # Check label validity
        valid_labels = {'keyword', 'non_keyword'}
        invalid_labels = [label for label in labels if label not in valid_labels]
        
        if invalid_labels:
            logger.error(f"Found invalid labels: {set(invalid_labels)}")
            return False
        
        logger.info("Augmented dataset validation passed")
        return True
