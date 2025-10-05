"""
FIXED: Adversarial augmentation for data augmentation (not adversarial training).

Key changes:
1. Generate from MINORITY class (keywords) - create harder positive examples
2. Use a BALANCED baseline model (not imbalanced)
3. Use smaller, targeted perturbations
4. Add quality validation
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FixedAdversarialAugmenter:
    """
    Generates adversarial examples for data augmentation.
    
    Strategy (CORRECTED):
    - Take minority class (keyword) samples
    - Add small perturbations to make them harder to classify
    - Keep them labeled as keywords
    - This creates challenging positive examples near decision boundary
    """
    
    def __init__(self, epsilon: float = 0.005):
        """
        Initialize adversarial augmenter.
        
        Args:
            epsilon: Perturbation magnitude (REDUCED from 0.01 to 0.005)
        """
        self.epsilon = epsilon
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initialized FixedAdversarialAugmenter (epsilon={epsilon})")
    
    def set_target_model(self, model: nn.Module):
        """Set the target model for adversarial generation."""
        self.model = model.to(self.device)
        self.model.eval()
        logger.info("Set target model for adversarial generation")
    
    def is_ready(self) -> bool:
        """Check if augmenter is ready"""
        return self.model is not None
    
    def generate_fgsm_perturbation(self, audio: torch.Tensor, 
                                   true_label: int) -> torch.Tensor:
        """
        Generate adversarial perturbation using FGSM.
        
        CORRECTED: We want to KEEP the same label but make it harder to classify.
        Strategy: Maximize loss for the TRUE class (pushes toward decision boundary).
        """
        if not self.is_ready():
            raise RuntimeError("Model not set")
        
        # Prepare input
        audio_input = audio.clone().detach().to(self.device)
        audio_input.requires_grad_(True)
        
        label_tensor = torch.tensor([true_label], dtype=torch.long, device=self.device)
        
        # Forward pass
        self.model.zero_grad()
        output = self.model(audio_input.unsqueeze(0))
        
        # Calculate loss - we want to MAXIMIZE loss for true class
        # This pushes the sample toward the decision boundary
        loss = nn.CrossEntropyLoss()(output, label_tensor)
        
        # Backward pass
        loss.backward()
        
        # Get gradient sign
        grad_sign = audio_input.grad.data.sign()
        
        # Create perturbation (positive gradient increases loss)
        perturbed = audio_input.detach() + self.epsilon * grad_sign
        
        # Clip to valid audio range
        perturbed = torch.clamp(perturbed, -1.0, 1.0)
        
        return perturbed.cpu()
    
    def select_source_samples(self, audio_files: List[torch.Tensor],
                            labels: List[str], n_samples: int
                            ) -> List[Tuple[torch.Tensor, int, int]]:
        """
        Select source samples for adversarial generation.
        
        CORRECTED: Use MINORITY class (keyword) samples.
        We want to create harder positive examples, not fake positives from negatives.
        
        Returns:
            List of (audio, true_label, index) tuples
        """
        # Get MINORITY class indices
        minority_indices = [
            i for i, label in enumerate(labels)
            if label == 'keyword'
        ]
        
        if len(minority_indices) == 0:
            logger.error("No minority class samples available")
            return []
        
        # Sample with replacement if needed (we can perturb same sample differently)
        n_available = len(minority_indices)
        replace = n_samples > n_available
        
        selected_indices = np.random.choice(
            minority_indices,
            size=min(n_samples, n_available * 3),  # Limit to 3x oversampling
            replace=replace
        ).tolist()
        
        # Return (audio, label=1 for keyword, index) tuples
        source_samples = [
            (audio_files[i], 1, i) for i in selected_indices
        ]
        
        logger.info(f"Selected {len(source_samples)} minority samples for adversarial generation")
        
        return source_samples
    
    def generate_adversarial_batch(self, source_samples: List[Tuple[torch.Tensor, int, int]]
                                 ) -> List[torch.Tensor]:
        """Generate adversarial samples in batch."""
        adversarial_samples = []
        
        for i, (audio, true_label, orig_idx) in enumerate(source_samples):
            try:
                # Generate perturbation
                adv_sample = self.generate_fgsm_perturbation(audio, true_label)
                
                # Validate quality
                if self._validate_adversarial(audio, adv_sample):
                    adversarial_samples.append(adv_sample)
                else:
                    logger.debug(f"Sample {i} failed validation, skipping")
                
                if (i + 1) % 100 == 0:
                    logger.debug(f"Generated {i + 1}/{len(source_samples)} adversarial samples")
                    
            except Exception as e:
                logger.error(f"Failed to generate adversarial sample {i}: {e}")
                continue
        
        logger.info(f"Generated {len(adversarial_samples)}/{len(source_samples)} valid adversarial samples")
        
        return adversarial_samples
    
    def _validate_adversarial(self, original: torch.Tensor, 
                            adversarial: torch.Tensor) -> bool:
        """
        Validate adversarial sample quality.
        
        Checks:
        1. Perturbation is within epsilon bounds
        2. No NaN/Inf values
        3. Stays in valid audio range
        4. Not too similar to original (some diversity)
        """
        # Check for invalid values
        if torch.isnan(adversarial).any() or torch.isinf(adversarial).any():
            return False
        
        # Check audio range
        if adversarial.abs().max() > 1.0:
            return False
        
        # Check perturbation magnitude
        perturbation = (adversarial - original).abs()
        max_pert = perturbation.max().item()
        
        if max_pert > self.epsilon * 1.2:  # Allow 20% tolerance
            logger.debug(f"Perturbation too large: {max_pert:.6f} > {self.epsilon * 1.2:.6f}")
            return False
        
        # Check that there IS some perturbation (not identical)
        if max_pert < self.epsilon * 0.1:  # At least 10% of epsilon
            logger.debug(f"Perturbation too small: {max_pert:.6f}")
            return False
        
        return True
    
    def generate_adversarial_samples(self, audio_files: List[torch.Tensor],
                                   labels: List[str], n_samples: int
                                   ) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Generate adversarial samples for dataset augmentation (main interface).
        
        CORRECTED: Generates hard positive examples from minority class.
        
        Args:
            audio_files: Original audio samples
            labels: Original labels
            n_samples: Number of adversarial samples to generate
        
        Returns:
            Tuple of (adversarial_audio, adversarial_labels)
        """
        if not self.is_ready():
            raise RuntimeError("Model not set. Call set_target_model() first.")
        
        if n_samples <= 0:
            return [], []
        
        logger.info(f"Generating {n_samples} adversarial samples (epsilon={self.epsilon})")
        
        # Select minority class samples
        source_samples = self.select_source_samples(audio_files, labels, n_samples)
        
        if not source_samples:
            logger.error("No source samples available")
            return [], []
        
        # Generate adversarial examples
        adversarial_audio = self.generate_adversarial_batch(source_samples)
        
        if not adversarial_audio:
            logger.error("Failed to generate any valid adversarial samples")
            return [], []
        
        # All adversarial samples keep their original label (keyword)
        adversarial_labels = ['keyword'] * len(adversarial_audio)
        
        logger.info(f"Successfully generated {len(adversarial_audio)} adversarial samples")
        
        return adversarial_audio, adversarial_labels
    
    def analyze_perturbations(self, original_samples: List[torch.Tensor],
                            adversarial_samples: List[torch.Tensor]) -> dict:
        """Analyze characteristics of generated perturbations."""
        if len(original_samples) != len(adversarial_samples):
            raise ValueError("Sample count mismatch")
        
        stats = {
            'l2_norm': [],
            'max_pert': [],
            'mean_pert': [],
            'snr_db': []
        }
        
        for orig, adv in zip(original_samples, adversarial_samples):
            pert = adv - orig
            
            stats['l2_norm'].append(torch.norm(pert).item())
            stats['max_pert'].append(pert.abs().max().item())
            stats['mean_pert'].append(pert.abs().mean().item())
            
            # SNR
            signal_power = torch.norm(orig) ** 2
            noise_power = torch.norm(pert) ** 2
            if noise_power > 1e-10:
                snr = 10 * torch.log10(signal_power / noise_power).item()
                stats['snr_db'].append(snr)
        
        # Summary statistics
        analysis = {}
        for key, values in stats.items():
            if values:
                analysis[f'{key}_mean'] = np.mean(values)
                analysis[f'{key}_std'] = np.std(values)
                analysis[f'{key}_min'] = np.min(values)
                analysis[f'{key}_max'] = np.max(values)
        
        logger.info(
            f"Perturbation analysis: "
            f"L2={analysis.get('l2_norm_mean', 0):.6f}, "
            f"SNR={analysis.get('snr_db_mean', 0):.1f}dB"
        )
        
        return analysis


def train_balanced_baseline_for_adversarial(dataset_manager, config, 
                                           dataset_size: str, keywords: List[str]):
    """
    Train a BALANCED baseline model for adversarial generation.
    
    CRITICAL FIX: The baseline model must be trained on balanced data,
    not the imbalanced experimental data.
    """
    from model_training import ModelTrainer
    import logging
    
    logger = logging.getLogger(__name__)
    
    logger.info("Training BALANCED baseline model for adversarial generation")
    
    # Load dataset
    audio_files, labels = dataset_manager.load_dataset(dataset_size)
    
    # Create BALANCED split (ratio=1.0)
    audio_files, labels = dataset_manager.create_imbalanced_split(
        audio_files, labels, keywords, imbalance_ratio=1.0  # ← BALANCED
    )
    
    # Train/test split
    train_audio, train_labels, test_audio, test_labels = dataset_manager.split_train_test(
        audio_files, labels, test_ratio=0.2, random_state=42
    )
    
    # Train model
    trainer = ModelTrainer()
    logger.info("Training balanced baseline (this may take a few minutes)...")
    
    metrics, baseline_model = trainer.full_training_pipeline(
        train_audio, train_labels, test_audio, test_labels, config
    )
    
    logger.info(f"Baseline model trained: F1={metrics['f1_keyword']:.3f}")
    
    return baseline_model
