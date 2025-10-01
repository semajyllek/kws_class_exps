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
from tqdm import tqdm

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
            'large': 10000,
            'full': None
        }
    
    def _check_dataset_exists(self) -> bool:
        """Check if dataset is already downloaded and extracted"""
        # Check for the extracted dataset directory structure
        dataset_dir = self.root_dir / 'SpeechCommands' / 'speech_commands_v0.02'
        return dataset_dir.exists() and len(list(dataset_dir.glob('*'))) > 0
    
    def load_raw_dataset(self, subset: str = "training") -> torchaudio.datasets.SPEECHCOMMANDS:
        """Load raw Google Speech Commands dataset"""
        logger.info(f"Loading GSC {self.version} dataset from {self.root_dir}")
        
        # Check if dataset needs to be downloaded/extracted
        dataset_exists = self._check_dataset_exists()
        
        if not dataset_exists:
            print("\n" + "="*60)
            print("📦 FIRST-TIME SETUP: Google Speech Commands Dataset")
            print("="*60)
            print("This will download and extract ~2.3GB (one-time only)")
            print("\nSteps:")
            print("  1. ⬇️  Download (shows progress bar)")
            print("  2. 📂 Extract archive (5-10 min, no progress)")
            print("  3. 🔄 Load samples (shows progress bar)")
            print("\n⏳ Starting download...")
            print("="*60 + "\n")
        
        dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(self.root_dir),
            download=True,
            subset=subset if self.version == 'v2' else None
        )
        
        if not dataset_exists:
            print("\n" + "="*60)
            print("✅ Dataset ready! Moving to sample loading...")
            print("="*60 + "\n")
        
        return dataset

    
    def extract_audio_labels(self, dataset, max_samples: int = None) -> Tuple[List[torch.Tensor], List[str]]:
        """Extract and preprocess audio samples with labels"""
        audio_files = []
        labels = []
        
        # Get dataset length for progress bar
        try:
            total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
        except:
            total = max_samples
        
        logger.info(f"Extracting samples (max: {max_samples or 'unlimited'})")
        print(f"\n📂 Loading audio samples from Google Speech Commands...")
        
        # Add progress bar
        with tqdm(total=total, desc="Processing audio", unit="samples") as pbar:
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
                
                # Update progress bar
                pbar.update(1)
                
                # Update description every 1000 samples
                if i % 1000 == 0 and i > 0:
                    pbar.set_postfix({'valid': len(audio_files), 'invalid': i - len(audio_files)})
        
        # Log label distribution
        label_counts = Counter(labels)
        print(f"\n✓ Loaded {len(audio_files)} valid samples")
        print(f"📊 Top labels: {dict(label_counts.most_common(10))}")
        
        logger.info(f"Extracted {len(audio_files)} valid samples")
        return audio_files, labels
    
    def load_dataset(self, size: str = 'full') -> Tuple[List[torch.Tensor], List[str]]:
        """Load dataset with specified size constraint"""
        dataset = self.load_raw_dataset()
        max_samples = self.dataset_size_limits[size]
        
        if max_samples:
            print(f"📏 Loading {size} dataset (max {max_samples} samples)")
        else:
            print(f"📏 Loading full dataset (this may take 5-10 minutes)")
        
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
        """
        Create dataset with specified imbalance ratio.
        
        imbalance_ratio = n_positive / n_negative
        
        For example:
        - ratio=0.1 means 10% positive, 90% negative
        - ratio=0.5 means 33% positive, 67% negative  
        - ratio=1.0 means 50% positive, 50% negative
        """
        
        print(f"\n⚖️  Creating imbalanced split (ratio: {imbalance_ratio})")
        
        # Convert to binary labels
        binary_labels = self.convert_to_binary_labels(labels, target_keywords)
        
        # Get class indices
        positive_indices = [i for i, label in enumerate(binary_labels) if label == 'keyword']
        negative_indices = [i for i, label in enumerate(binary_labels) if label == 'non_keyword']
        
        n_positive_available = len(positive_indices)
        n_negative_available = len(negative_indices)
        
        logger.info(f"Available: {n_positive_available} positive, {n_negative_available} negative")
        
        # FIXED: Calculate target counts to achieve exact imbalance ratio
        # Strategy: Keep as many positives as possible, then calculate negatives needed
        # to achieve target ratio: ratio = n_pos / n_neg, so n_neg = n_pos / ratio
        
        if imbalance_ratio >= 1.0:
            # Balanced or positive-heavy: keep all positives, match with negatives
            n_positive = n_positive_available
            n_negative = n_positive  # 1:1 ratio
        else:
            # Imbalanced: keep positives, calculate negatives needed
            n_positive = n_positive_available
            n_negative = int(n_positive / imbalance_ratio)
            
            # Check if we have enough negatives
            if n_negative > n_negative_available:
                # Not enough negatives, reduce positives instead
                n_negative = n_negative_available
                n_positive = int(n_negative * imbalance_ratio)
                
                logger.warning(f"Not enough negatives. Reducing positives to {n_positive}")
        
        # Sample the required number of samples
        if n_positive < n_positive_available:
            selected_positive = np.random.choice(
                positive_indices, n_positive, replace=False
            ).tolist()
        else:
            selected_positive = positive_indices
        
        if n_negative < n_negative_available:
            selected_negative = np.random.choice(
                negative_indices, n_negative, replace=False
            ).tolist()
        else:
            selected_negative = negative_indices
        
        # Combine selected indices
        selected_indices = selected_positive + selected_negative
        np.random.shuffle(selected_indices)
        
        # Create new dataset
        new_audio = [audio_files[i] for i in selected_indices]
        new_labels = [binary_labels[i] for i in selected_indices]
        
        # Log final distribution
        final_pos = sum(1 for label in new_labels if label == 'keyword')
        final_neg = sum(1 for label in new_labels if label == 'non_keyword')
        actual_ratio = final_pos / final_neg if final_neg > 0 else float('inf')
        
        print(f"✓ Created: {final_pos} positive, {final_neg} negative (actual ratio: {actual_ratio:.3f})")
        
        logger.info(f"Created imbalanced dataset: {final_pos} positive, {final_neg} negative")
        logger.info(f"Target ratio: {imbalance_ratio:.3f}, Actual ratio: {actual_ratio:.3f}")
        
        return new_audio, new_labels
    
    def split_train_test(self, audio_files: List[torch.Tensor], 
                        labels: List[str], test_ratio: float = 0.2,
                        random_state: int = None) -> Tuple[List, List, List, List]:
        """Split dataset into train and test sets with stratification"""
        
        if random_state is not None:
            np.random.seed(random_state)
        
        print(f"\n✂️  Splitting into train/test (test ratio: {test_ratio})")
        
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
        
        # Log class distribution in splits
        train_pos = sum(1 for l in train_labels if l == 'keyword')
        train_neg = sum(1 for l in train_labels if l == 'non_keyword')
        test_pos = sum(1 for l in test_labels if l == 'keyword')
        test_neg = sum(1 for l in test_labels if l == 'non_keyword')
        
        print(f"✓ Train: {len(train_audio)} samples ({train_pos} pos, {train_neg} neg)")
        print(f"✓ Test: {len(test_audio)} samples ({test_pos} pos, {test_neg} neg)")
        
        logger.info(f"Train split: {len(train_audio)} samples ({train_pos} pos, {train_neg} neg)")
        logger.info(f"Test split: {len(test_audio)} samples ({test_pos} pos, {test_neg} neg)")
        
        return train_audio, train_labels, test_audio, test_labels
