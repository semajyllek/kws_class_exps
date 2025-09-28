"""
Audio processing utilities for keyword detection experiments.
"""
import torch
import torchaudio
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles consistent audio preprocessing across experiments"""
    
    def __init__(self, sample_rate: int = 16000, max_length: int = 16000):
        self.sample_rate = sample_rate
        self.max_length = max_length
        
    def normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio to [-1, 1] range with epsilon for stability"""
        max_val = waveform.abs().max()
        if max_val > 1e-8:
            return waveform / max_val
        else:
            return waveform
    
    def pad_or_truncate(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pad with zeros or truncate to fixed length"""
        current_length = waveform.shape[-1]
        
        if current_length > self.max_length:
            # Truncate from center to preserve onset and offset
            start_idx = (current_length - self.max_length) // 2
            waveform = waveform[:, start_idx:start_idx + self.max_length]
        elif current_length < self.max_length:
            # Pad with zeros
            padding = self.max_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        return waveform
    
    def resample_audio(self, waveform: torch.Tensor, 
                      original_sr: int) -> torch.Tensor:
        """Resample audio to target sample rate"""
        if original_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sr, 
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)
        return waveform
    
    def preprocess_audio(self, waveform: torch.Tensor, 
                        original_sr: int) -> torch.Tensor:
        """Complete preprocessing pipeline"""
        # Ensure 2D tensor (channels, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        waveform = self.resample_audio(waveform, original_sr)
        
        # Normalize and standardize length
        waveform = self.normalize_audio(waveform)
        waveform = self.pad_or_truncate(waveform)
        
        return waveform
    
    def validate_audio(self, waveform: torch.Tensor) -> bool:
        """Validate that audio meets requirements"""
        if not isinstance(waveform, torch.Tensor):
            return False
        
        if waveform.dim() != 2 or waveform.shape[0] != 1:
            return False
            
        if waveform.shape[1] != self.max_length:
            return False
            
        if torch.isnan(waveform).any() or torch.isinf(waveform).any():
            return False
            
        return True

class AudioVariationGenerator:
    """Generates acoustic variations for data augmentation"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def add_gaussian_noise(self, waveform: torch.Tensor, 
                          noise_level: float = 0.01) -> torch.Tensor:
        """Add Gaussian noise to audio"""
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    
    def scale_amplitude(self, waveform: torch.Tensor, 
                       scale_factor: float) -> torch.Tensor:
        """Scale audio amplitude"""
        return waveform * scale_factor
    
    def create_variation(self, waveform: torch.Tensor, 
                        variation_type: str = 'light') -> torch.Tensor:
        """Create single acoustic variation"""
        varied = waveform.clone()
        
        if variation_type == 'light':
            # Light variations for TTS samples
            noise_level = np.random.uniform(0.005, 0.015)
            amp_factor = np.random.uniform(0.8, 1.2)
            
            varied = self.add_gaussian_noise(varied, noise_level)
            varied = self.scale_amplitude(varied, amp_factor)
            
        # Ensure normalization
        return self._safe_normalize(varied)
    
    def _safe_normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        """Safely normalize audio, handling edge cases"""
        max_val = waveform.abs().max()
        if max_val > 1e-8:
            # Prevent clipping by scaling to 90% of range
            return waveform / max_val * 0.9
        else:
            return waveform
