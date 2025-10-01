"""
Adversarial augmentation using FGSM for keyword detection class imbalance.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AdversarialAugmenter:
    """Generates adversarial examples using Fast Gradient Sign Method (FGSM)"""
    
    def __init__(self, epsilon: float = 0.01):
        """
        Initialize adversarial augmenter.
        
        Args:
            epsilon: Perturbation magnitude for FGSM attack
        """
        self.epsilon = epsilon
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initialized AdversarialAugmenter (epsilon={epsilon}, device={self.device})")
    
    def set_target_model(self, model: nn.Module):
        """
        Set the target model for adversarial generation.
        
        Args:
            model: Trained PyTorch model to generate adversarial examples against
        """
        self.model = model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        logger.info(f"Set target model for adversarial generation (epsilon={self.epsilon})")
    
    def is_ready(self) -> bool:
        """Check if augmenter is ready to generate samples"""
        return self.model is not None
    
    def validate_ready(self):
        """Raise error if model is not set"""
        if not self.is_ready():
            raise RuntimeError(
                "No target model set for adversarial generation. "
                "Call set_target_model() with a trained model first."
            )
    
    def generate_fgsm_sample(self, audio: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        Generate single adversarial example using FGSM.
        
        Args:
            audio: Input audio tensor
            target_class: Target class label (0 or 1)
        
        Returns:
            Adversarial audio tensor
        """
        self.validate_ready()
        
        # Prepare input
        audio_input = audio.clone().detach().to(self.device)
        audio_input.requires_grad_(True)
        
        target_tensor = torch.tensor([target_class], dtype=torch.long, device=self.device)
        
        # Forward pass
        self.model.zero_grad()
        output = self.model(audio_input.unsqueeze(0))
        
        # Calculate loss (we want to maximize loss for target class)
        loss = nn.CrossEntropyLoss()(output, target_tensor)
        
        # Backward pass to get gradients
        loss.backward()
        
        # Generate adversarial perturbation using gradient sign
        audio_grad = audio_input.grad.data
        sign_grad = audio_grad.sign()
        
        # Apply perturbation
        adversarial_audio = audio_input.detach() + self.epsilon * sign_grad
        
        # Ensure audio remains in valid range [-1, 1]
        adversarial_audio = torch.clamp(adversarial_audio, -1.0, 1.0)
        
        return adversarial_audio.cpu()
    
    def select_source_samples(self, audio_files: List[torch.Tensor],
                            labels: List[str], n_samples: int) -> List[Tuple[torch.Tensor, int]]:
        """
        Select source samples for adversarial generation.
        
        Strategy: Use majority class (non-keyword) samples and flip them to minority class (keyword).
        
        Args:
            audio_files: List of audio tensors
            labels: List of corresponding labels
            n_samples: Number of samples to select
        
        Returns:
            List of (audio, target_label) tuples
        """
        # Get indices of majority class samples
        majority_class_indices = [
            i for i, label in enumerate(labels)
            if label == 'non_keyword'
        ]
        
        if len(majority_class_indices) == 0:
            logger.error("No majority class samples available for adversarial generation")
            return []
        
        if len(majority_class_indices) < n_samples:
            logger.warning(
                f"Only {len(majority_class_indices)} majority samples available, "
                f"requested {n_samples}. Using all available samples."
            )
            selected_indices = majority_class_indices
        else:
            # Randomly sample from majority class
            selected_indices = np.random.choice(
                majority_class_indices,
                n_samples,
                replace=False
            ).tolist()
        
        # Return (audio, target_label) pairs - flip to keyword class (label 1)
        source_samples = [(audio_files[i], 1) for i in selected_indices]
        
        logger.info(f"Selected {len(source_samples)} source samples for adversarial generation")
        
        return source_samples
    
    def generate_adversarial_batch(self, source_samples: List[Tuple[torch.Tensor, int]]
                                 ) -> List[torch.Tensor]:
        """
        Generate adversarial samples in batch.
        
        Args:
            source_samples: List of (audio, target_class) tuples
        
        Returns:
            List of adversarial audio tensors
        """
        adversarial_samples = []
        
        for i, (audio, target_class) in enumerate(source_samples):
            try:
                adv_sample = self.generate_fgsm_sample(audio, target_class)
                adversarial_samples.append(adv_sample)
                
                if (i + 1) % 100 == 0:
                    logger.debug(f"Generated {i + 1}/{len(source_samples)} adversarial samples")
                    
            except Exception as e:
                logger.error(f"Failed to generate adversarial sample {i}: {e}")
                # Skip this sample - don't use fallbacks in scientific experiments
                continue
        
        logger.info(f"Successfully generated {len(adversarial_samples)}/{len(source_samples)} adversarial samples")
        
        return adversarial_samples
    
    def validate_adversarial_samples(self, original_samples: List[torch.Tensor],
                                   adversarial_samples: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Validate that adversarial samples are properly perturbed.
        
        Args:
            original_samples: Original audio samples
            adversarial_samples: Generated adversarial samples
        
        Returns:
            Tuple of (valid_samples, invalid_indices)
        """
        valid_samples = []
        invalid_indices = []
        
        for i, (orig, adv) in enumerate(zip(original_samples, adversarial_samples)):
            # Check that perturbation is within bounds
            perturbation = (adv - orig).abs().max().item()
            
            # Allow small numerical error (10% tolerance)
            if perturbation > self.epsilon * 1.1:
                logger.warning(
                    f"Sample {i}: perturbation {perturbation:.6f} exceeds "
                    f"epsilon {self.epsilon} by {(perturbation/self.epsilon - 1)*100:.1f}%"
                )
            
            # Check for valid audio properties
            is_valid = True
            
            if torch.isnan(adv).any() or torch.isinf(adv).any():
                logger.warning(f"Sample {i}: contains NaN or Inf values")
                is_valid = False
            
            if adv.abs().max() > 1.0:
                logger.warning(f"Sample {i}: exceeds valid range [-1, 1]")
                is_valid = False
            
            if is_valid:
                valid_samples.append(adv)
            else:
                invalid_indices.append(i)
        
        logger.info(
            f"Validated {len(valid_samples)}/{len(adversarial_samples)} adversarial samples "
            f"({len(invalid_indices)} failed validation)"
        )
        
        return valid_samples, invalid_indices
    
    def generate_adversarial_samples(self, audio_files: List[torch.Tensor],
                                   labels: List[str], n_samples: int
                                   ) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Generate adversarial samples for dataset balancing (main interface).
        
        Args:
            audio_files: List of original audio samples
            labels: List of corresponding labels
            n_samples: Number of adversarial samples to generate
        
        Returns:
            Tuple of (adversarial_audio, adversarial_labels)
        """
        self.validate_ready()
        
        if n_samples <= 0:
            return [], []
        
        logger.info(f"Generating {n_samples} adversarial samples using FGSM (epsilon={self.epsilon})")
        
        # Select source samples from majority class
        source_samples = self.select_source_samples(audio_files, labels, n_samples)
        
        if not source_samples:
            logger.error("No source samples available for adversarial generation")
            return [], []
        
        # Generate adversarial samples
        adversarial_audio = self.generate_adversarial_batch(source_samples)
        
        if not adversarial_audio:
            logger.error("Failed to generate any adversarial samples")
            return [], []
        
        # Validate generated samples
        original_audio = [sample[0] for sample in source_samples[:len(adversarial_audio)]]
        adversarial_audio, invalid_indices = self.validate_adversarial_samples(
            original_audio,
            adversarial_audio
        )
        
        # Create labels (all adversarial samples become positive/keyword class)
        adversarial_labels = ['keyword'] * len(adversarial_audio)
        
        logger.info(f"Successfully generated {len(adversarial_audio)} valid adversarial samples")
        
        return adversarial_audio, adversarial_labels
    
    def analyze_perturbations(self, original_samples: List[torch.Tensor],
                            adversarial_samples: List[torch.Tensor]) -> dict:
        """
        Analyze characteristics of generated perturbations.
        
        Args:
            original_samples: Original audio samples
            adversarial_samples: Generated adversarial samples
        
        Returns:
            Dictionary with perturbation statistics
        """
        if len(original_samples) != len(adversarial_samples):
            raise ValueError("Mismatch in sample counts for perturbation analysis")
        
        perturbation_stats = {
            'l2_norm': [],
            'max_perturbation': [],
            'snr_db': []
        }
        
        for orig, adv in zip(original_samples, adversarial_samples):
            perturbation = adv - orig
            
            # L2 norm of perturbation
            l2_norm = torch.norm(perturbation).item()
            perturbation_stats['l2_norm'].append(l2_norm)
            
            # Maximum absolute perturbation
            max_pert = perturbation.abs().max().item()
            perturbation_stats['max_perturbation'].append(max_pert)
            
            # Signal-to-noise ratio
            signal_power = torch.norm(orig) ** 2
            noise_power = torch.norm(perturbation) ** 2
            
            if noise_power > 1e-10:
                snr_db = 10 * torch.log10(signal_power / noise_power).item()
                perturbation_stats['snr_db'].append(snr_db)
        
        # Calculate statistics
        analysis = {}
        for key, values in perturbation_stats.items():
            if values:
                analysis[f'{key}_mean'] = np.mean(values)
                analysis[f'{key}_std'] = np.std(values)
                analysis[f'{key}_min'] = np.min(values)
                analysis[f'{key}_max'] = np.max(values)
        
        logger.info(
            f"Perturbation analysis: "
            f"mean L2 norm = {analysis.get('l2_norm_mean', 0):.6f}, "
            f"mean SNR = {analysis.get('snr_db_mean', 0):.2f} dB"
        )
        
        return analysis
