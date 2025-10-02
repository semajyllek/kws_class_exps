"""
Dataset management for Google Speech Commands experiments.
WITH EXPLICIT VOCABULARY CONTROL
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
    
    def __init__(self, root_dir: str, version: str = 'v2', 
                 target_keywords: List[str] = None):
        self.root_dir = Path(root_dir)
        self.version = version
        self.audio_processor = AudioProcessor()
        
        # VOCABULARY CONTROL - only need to specify positive keywords
        # Negative class = everything else (realistic keyword spotting)
        self.target_keywords = target_keywords or ['forward', 'backward', 'left', 'right']
        
        # Define dataset size limits
        self.dataset_size_limits = {
            'small': 1000,
            'medium': 5000,
            'large': 10000,
            'full': None
        }
        
        logger.info(f"Dataset manager initialized")
        logger.info(f"  Positive keywords: {self.target_keywords}")
        logger.info(f"  Negative class: All other words (realistic keyword spotting)")
    
    def _check_dataset_exists(self) -> bool:
        """Check if dataset is already downloaded and extracted"""
        dataset_dir = self.root_dir / 'SpeechCommands' / 'speech_commands_v0.02'
        return dataset_dir.exists() and len(list(dataset_dir.glob('*'))) > 0
    
    def load_raw_dataset(self, subset: str = "training") -> torchaudio.datasets.SPEECHCOMMANDS:
        """Load raw Google Speech Commands dataset"""
        logger.info(f"Loading GSC {self.version} dataset from {self.root_dir}")
        
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
    
    def load_dataset(self, size: str = 'full') -> Tuple[List[torch.Tensor], List[str]]:
        """
        Load dataset with controlled vocabulary.
        
        CRITICAL: Only loads samples from target_keywords + negative_keywords.
        This ensures reproducible, well-defined experimental conditions.
        """
        dataset = self.load_raw_dataset()
        
        # Define allowed vocabulary
        allowed_vocabulary = set(self.target_keywords + self.negative_keywords)
        
        max_samples = self.dataset_size_limits[size]
        
        if max_samples:
            print(f"📏 Loading {size} dataset (max {max_samples} samples from controlled vocabulary)")
        else:
            print(f"📏 Loading full dataset from controlled vocabulary")
        
        print(f"   Positive: {self.target_keywords}")
        print(f"   Negative: {self.negative_keywords}")
        
        # Load samples organized by label
        print(f"\n📂 Loading audio samples from Google Speech Commands...")
        
        label_to_samples = {}
        total_processed = 0
        
        # First pass: collect all samples from allowed vocabulary
        for waveform, sample_rate, label, speaker_id, utterance_number in tqdm(dataset, desc="Scanning dataset"):
            # Skip if not in our controlled vocabulary
            if label not in allowed_vocabulary:
                continue
            
            # Preprocess audio
            processed_audio = self.audio_processor.preprocess_audio(waveform, sample_rate)
            
            # Validate processed audio
            if self.audio_processor.validate_audio(processed_audio):
                if label not in label_to_samples:
                    label_to_samples[label] = []
                label_to_samples[label].append(processed_audio)
                total_processed += 1
        
        print(f"\n✓ Loaded {total_processed} samples from {len(label_to_samples)} labels")
        
        # Show distribution
        label_counts = {label: len(samples) for label, samples in label_to_samples.items()}
        print(f"📊 Distribution: {label_counts}")
        
        # Check we have all required keywords
        missing_positive = [kw for kw in self.target_keywords if kw not in label_to_samples]
        missing_negative = [kw for kw in self.negative_keywords if kw not in label_to_samples]
        
        if missing_positive:
            logger.warning(f"Missing positive keywords: {missing_positive}")
            print(f"⚠️  WARNING: Missing positive keywords: {missing_positive}")
        
        if missing_negative:
            logger.warning(f"Missing negative keywords: {missing_negative}")
            print(f"⚠️  WARNING: Missing negative keywords: {missing_negative}")
        
        # Combine all samples
        audio_files = []
        labels = []
        
        for label, samples in label_to_samples.items():
            audio_files.extend(samples)
            labels.extend([label] * len(samples))
        
        # Shuffle to mix positive and negative classes
        combined = list(zip(audio_files, labels))
        np.random.shuffle(combined)
        
        # Limit to max_samples if specified
        if max_samples and len(combined) > max_samples:
            combined = combined[:max_samples]
        
        audio_files, labels = zip(*combined) if combined else ([], [])
        audio_files = list(audio_files)
        labels = list(labels)
        
        # Final distribution
        final_counts = Counter(labels)
        print(f"\n✓ Final dataset: {len(audio_files)} samples")
        print(f"📊 Final distribution: {dict(final_counts)}")
        
        logger.info(f"Loaded {len(audio_files)} samples from controlled vocabulary")
        
        return audio_files, labels
    
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
        
        if n_positive_available == 0:
            raise ValueError(
                f"No positive samples found! Check that target_keywords {target_keywords} "
                f"are present in the loaded dataset."
            )
        
        if n_negative_available == 0:
            raise ValueError(
                f"No negative samples found! Check that negative_keywords are present."
            )
        
        # Calculate target counts to achieve exact imbalance ratio
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
