"""
Dataset management for Google Speech Commands experiments.
"""
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter
import logging

from audio_processing import AudioProcessor

logger = logging.getLogger(__name__)

class GSCDatasetManager:
    """Manages Google Speech Commands dataset loading and manipulation"""
    
    def __init__(self, root_dir: str, version: str = 'v2'):
        self.root_dir = Path(root_dir)
        self.version = version
        self.audio_processor = AudioProcessor()
        
        # Define dataset size limits
        self.dataset_size_limits = {
            'small': 1000,
            'medium': 5000,
            'large': 20000,
            'full': None
        }
        
    def load_raw_dataset(self, subset: str = "training") -> torchaudio.datasets.SPEECHCOMMANDS:
        """Load raw Google Speech Commands dataset"""
        logger.info(f"Loading GSC {self.version} dataset from {self.root_dir}")
        
        return torchaudio.datasets.SPEECHCOMMANDS(
            root=str(self.root_dir),
            download=True,
            subset=subset if self.version == 'v2' else None
        )
    
    def extract_audio_labels(self, dataset, max_samples: int = None) -> Tuple[List[torch.Tensor], List[str]]:
        """Extract and preprocess audio samples with labels"""
        audio_files = []
        labels = []
          
        logger.info(f"Extracting samples (max: {max_samples or 'unlimited'})")
        
        for i, (waveform, sample_rate, label, speaker_id, utterance_number) in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            
            # Preprocess audio
            processed_audio = self.audio_processor.preprocess_audio(waveform, sample_rate)
            
            # Validate processed audio
            if self.audio_processor.validate_audio(processed_audio):
                audio_files.append(processed_audio)
                labels.append(label)
            else:
                logger.warning(f"Skipping invalid audio sample {i}")
        

        # TODO: delete later
        from collections import Counter
        label_counts = Counter(labels)
        print(f"Available labels: {dict(label_counts.most_common(10))}")
        

        logger.info(f"Extracted {len(audio_files)} valid samples")
        return audio_files, labels
    
    def load_dataset(self, size: str = 'full') -> Tuple[List[torch.Tensor], List[str]]:
        """Load dataset with specified size constraint"""
        dataset = self.load_raw_dataset()
        max_samples = self.dataset_size_limits[size]
        return self.extract_audio_labels(dataset, max_samples)
    
    def convert_to_binary_labels(self, labels: List[str], 
                               target_keywords: List[str]) -> List[str]:
        """Convert multi-class labels to binary classification"""
        binary_labels = []
        for label in labels:
            if label in target_keywords:
                binary_labels.append('keyword')
            else:
                binary_labels.append('non_keyword')
        return binary_labels
    
    def create_imbalanced_split(self, audio_files: List[torch.Tensor], 
                              labels: List[str], target_keywords: List[str], 
                              imbalance_ratio: float) -> Tuple[List[torch.Tensor], List[str]]:
        """Create dataset with specified imbalance ratio"""
        
        # Convert to binary labels
        binary_labels = self.convert_to_binary_labels(labels, target_keywords)
        
        # Get class indices
        positive_indices = [i for i, label in enumerate(binary_labels) if label == 'keyword']
        negative_indices = [i for i, label in enumerate(binary_labels) if label == 'non_keyword']
        
        n_positive = len(positive_indices)
        n_negative = len(negative_indices)
        
        logger.info(f"Original distribution: {n_positive} positive, {n_negative} negative")
        
        # Calculate target numbers based on imbalance ratio
        if imbalance_ratio >= 1.0:
            target_negative = n_positive
        else:
            target_negative = int(n_positive / imbalance_ratio)
        
        # Subsample negative class if needed
        if len(negative_indices) > target_negative:
            negative_indices = np.random.choice(
                negative_indices, target_negative, replace=False
            ).tolist()
        
        # Combine selected indices
        selected_indices = positive_indices + negative_indices
        np.random.shuffle(selected_indices)
        
        # Create new dataset
        new_audio = [audio_files[i] for i in selected_indices]
        new_labels = [binary_labels[i] for i in selected_indices]
        
        # Log final distribution
        final_pos = sum(1 for label in new_labels if label == 'keyword')
        final_neg = sum(1 for label in new_labels if label == 'non_keyword')
        actual_ratio = final_pos / final_neg if final_neg > 0 else float('inf')
        
        logger.info(f"Created imbalanced dataset: {final_pos} positive, {final_neg} negative")
        logger.info(f"Target ratio: {imbalance_ratio:.3f}, Actual ratio: {actual_ratio:.3f}")
        
        return new_audio, new_labels
    
    def split_train_test(self, audio_files: List[torch.Tensor], 
                        labels: List[str], test_ratio: float = 0.2,
                        random_state: int = None) -> Tuple[List, List, List, List]:
        """Split dataset into train and test sets with stratification"""
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Get indices for each class
        positive_indices = [i for i, label in enumerate(labels) if label == 'keyword']
        negative_indices = [i for i, label in enumerate(labels) if label == 'non_keyword']
        
        # Split each class separately (stratified split)
        n_pos_test = max(1, int(len(positive_indices) * test_ratio))
        n_neg_test = max(1, int(len(negative_indices) * test_ratio))
        
        # Randomly sample test indices
        test_pos_indices = np.random.choice(positive_indices, n_pos_test, replace=False).tolist()
        test_neg_indices = np.random.choice(negative_indices, n_neg_test, replace=False).tolist()
        
        # Remaining indices for training
        train_pos_indices = [i for i in positive_indices if i not in test_pos_indices]
        train_neg_indices = [i for i in negative_indices if i not in test_neg_indices]
        
        # Combine and shuffle
        train_indices = train_pos_indices + train_neg_indices
        test_indices = test_pos_indices + test_neg_indices
        
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
        
        # Create splits
        train_audio = [audio_files[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        test_audio = [audio_files[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]
        
        logger.info(f"Train split: {len(train_audio)} samples")
        logger.info(f"Test split: {len(test_audio)} samples")
        
        return train_audio, train_labels, test_audio, test_labels
