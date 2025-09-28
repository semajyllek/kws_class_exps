"""
Adversarial augmentation using FGSM for keyword detection class imbalance.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class AdversarialAugmenter:
    """Generates adversarial examples using Fast Gradient Sign Method (FGSM)"""
    
    def __init__(self, epsilon: float = 0.01):
        self.epsilon = epsilon
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def set_target_model(self, model: nn.Module):
        """Set the target model for adversarial generation"""
        self.model = model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        logger.info(f"Set target model for adversarial generation (epsilon={self.epsilon})")
    
    def validate_model_ready(self):
        """Ensure model is set and ready"""
        if self.model is None:
            raise RuntimeError("No target model set. Call set_target_model() first.")
    
    def generate_fgsm_sample(self, audio: torch.Tensor, target_class: int) -> torch.Tensor:
        """Generate single adversarial example using FGSM"""
        self.validate_model_ready()
        
        # Prepare input
        audio_input = audio.clone().detach().to(self.device)
        audio_input.requires_grad_(True)
        
        target_tensor = torch.tensor([target_class], dtype=torch.long, device=self.device)
        
        # Forward pass
        self.model.zero_grad()
        output = self.model(audio_input.unsqueeze(0))
        
        # Calculate loss (we want to maximize loss for target class)
        loss = nn.CrossEntropyLoss()(output, target_tensor)
        
        # Backward pass
        loss.backward()
        
        # Generate adversarial perturbation
        audio_grad = audio_input.grad.data
        sign_grad = audio_grad.sign()
        
        # Apply perturbation
        adversarial_audio = audio_input.detach() + self.epsilon * sign_grad
        
        # Ensure audio remains in valid range [-1, 1]
        adversarial_audio = torch.clamp(adversarial_audio, -1.0, 1.0)
        
        return adversarial_audio.cpu()
    
    def select_source_samples(self, audio_files: List[torch.Tensor], 
                            labels: List[str], n_samples: int) -> List[Tuple[torch.Tensor, int]]:
        """Select source samples for adversarial generation"""
        
        # Strategy: Use majority class samples and flip them to minority class
        majority_class_indices = [i for i, label in enumerate(labels) if label == 'non_keyword']
        
        if len(majority_class_indices) < n_samples:
            logger.warning(f"Only {len(majority_class_indices)} majority samples available, "
                         f"requested {n_samples}")
            selected_indices = majority_class_indices
        else:
            # Randomly sample from majority class
            selected_indices = np.random.choice(
                majority_class_indices, n_samples, replace=False
            ).tolist()
        
        # Return (audio, target_label) pairs - flip to keyword class (label 1)
        source_samples = [(audio_files[i], 1) for i in selected_indices]
        
        logger.info(f"Selected {len(source_samples)} source samples for adversarial generation")
        return source_samples
    
    def generate_adversarial_batch(self, source_samples: List[Tuple[torch.Tensor, int]]
                                 ) -> List[torch.Tensor]:
        """Generate adversarial samples in batch for efficiency"""
        adversarial_samples = []
        
        for i, (audio, target_class) in enumerate(source_samples):
            try:
                adv_sample = self.generate_fgsm_sample(audio, target_class)
                adversarial_samples.append(adv_sample)
                
                if (i + 1) % 10 == 0:
                    logger.debug(f"Generated {i + 1}/{len(source_samples)} adversarial samples")
                    
            except Exception as e:
                logger.error(f"Failed to generate adversarial sample {i}: {e}")
                # Skip this sample - don't use fallbacks in scientific experiments
                continue
        
        return adversarial_samples
    
    def validate_adversarial_samples(self, original_samples: List[torch.Tensor],
                                   adversarial_samples: List[torch.Tensor]) -> List[torch.Tensor]:
        """Validate that adversarial samples are properly perturbed"""
        valid_samples = []
        
        for i, (orig, adv) in enumerate(zip(original_samples, adversarial_samples)):
            # Check that perturbation is within bounds
            perturbation = (adv - orig).abs().max().item()
            
            if perturbation > self.epsilon * 1.1:  # Allow small numerical error
                logger.warning(f"Sample {i}: perturbation {perturbation:.6f} exceeds epsilon {self.epsilon}")
            
            # Check for valid audio
            if self.audio_processor.validate_audio(adv):
                valid_samples.append(adv)
            else:
                logger.warning(f"Adversarial sample {i} failed validation")
        
        logger.info(f"Validated {len(valid_samples)}/{len(adversarial_samples)} adversarial samples")
        return valid_samples
    
    def generate_adversarial_samples(self, audio_files: List[torch.Tensor], 
                                   labels: List[str], n_samples: int
                                   ) -> Tuple[List[torch.Tensor], List[str]]:
        """Generate adversarial samples for dataset balancing"""
        
        self.validate_model_ready()
        
        logger.info(f"Generating {n_samples} adversarial samples using FGSM")
        
        # Select source samples
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
        adversarial_audio = self.validate_adversarial_samples(original_audio, adversarial_audio)
        
        # Create labels (all adversarial samples become positive class)
        adversarial_labels = ['keyword'] * len(adversarial_audio)
        
        logger.info(f"Successfully generated {len(adversarial_audio)} adversarial samples")
        return adversarial_audio, adversarial_labels
    
    def analyze_perturbations(self, original_samples: List[torch.Tensor],
                            adversarial_samples: List[torch.Tensor]) -> dict:
        """Analyze characteristics of generated perturbations"""
        if len(original_samples) != len(adversarial_samples):
            raise ValueError("Mismatch in sample counts for perturbation analysis")
        
        perturbation_stats = {
            'mean_l2_norm': [],
            'max_perturbation': [],
            'snr_db': []
        }
        
        for orig, adv in zip(original_samples, adversarial_samples):
            perturbation = adv - orig
            
            # L2 norm of perturbation
            l2_norm = torch.norm(perturbation).item()
            perturbation_stats['mean_l2_norm'].append(l2_norm)
            
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
        
        logger.info(f"Perturbation analysis: mean L2 norm = {analysis.get('mean_l2_norm_mean', 0):.6f}")
        return analysis
