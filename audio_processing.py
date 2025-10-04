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
    
    def normalize_to_target_rms(self, waveform: torch.Tensor, 
        target_rms: float = 0.1) -> torch.Tensor:
        """Normalize audio to target RMS energy"""
        current_rms = torch.sqrt(torch.mean(waveform ** 2))
    
        if current_rms > 1e-8:
            scale_factor = target_rms / current_rms
            normalized = waveform * scale_factor
        
            # Prevent clipping
            max_val = normalized.abs().max()
            if max_val > 1.0:
                normalized = normalized / max_val * 0.95
        
            return normalized
        else:
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
    """Generates acoustic variations for data augmentation - ENHANCED VERSION"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def add_gaussian_noise(self, waveform: torch.Tensor, 
                          noise_level: float = 0.01) -> torch.Tensor:
        """Add Gaussian noise to audio"""
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    
    def add_colored_noise(self, waveform: torch.Tensor,
                         noise_level: float = 0.01,
                         noise_color: str = 'white') -> torch.Tensor:
        """
        Add colored noise (white, pink, brown).
        
        Different noise colors have different frequency characteristics.
        """
        if noise_color == 'white':
            # White noise (equal power at all frequencies)
            noise = torch.randn_like(waveform) * noise_level
        elif noise_color == 'pink':
            # Pink noise (1/f power spectrum)
            # Approximate using cascaded filters
            white = torch.randn_like(waveform)
            # Simple pink noise approximation
            noise = torch.zeros_like(waveform)
            b = torch.zeros(7)
            for i in range(len(white[0])):
                white_val = white[0, i]
                b[0] = 0.99886 * b[0] + white_val * 0.0555179
                b[1] = 0.99332 * b[1] + white_val * 0.0750759
                b[2] = 0.96900 * b[2] + white_val * 0.1538520
                b[3] = 0.86650 * b[3] + white_val * 0.3104856
                b[4] = 0.55000 * b[4] + white_val * 0.5329522
                b[5] = -0.7616 * b[5] - white_val * 0.0168980
                noise[0, i] = (b.sum() + white_val * 0.5362) * noise_level
        elif noise_color == 'brown':
            # Brown noise (1/f^2 power spectrum)
            white = torch.randn_like(waveform)
            brown = torch.cumsum(white, dim=1)
            brown = brown / brown.abs().max()  # Normalize
            noise = brown * noise_level
        else:
            noise = torch.randn_like(waveform) * noise_level
        
        return waveform + noise
    
    def scale_amplitude(self, waveform: torch.Tensor, 
                       scale_factor: float) -> torch.Tensor:
        """Scale audio amplitude"""
        return waveform * scale_factor
    
    def time_stretch(self, waveform: torch.Tensor, 
                    rate: float) -> torch.Tensor:
        """
        Time stretch audio (speed up/slow down without changing pitch).
        
        Args:
            rate: Stretch factor (0.9 = slower, 1.1 = faster)
        """
        import torch.nn.functional as F
        
        if abs(rate - 1.0) < 0.01:
            return waveform
        
        # Simple time stretching using interpolation
        original_length = waveform.shape[1]
        new_length = int(original_length / rate)
        
        # Resample
        stretched = F.interpolate(
            waveform.unsqueeze(0), 
            size=new_length, 
            mode='linear', 
            align_corners=False
        ).squeeze(0)
        
        # Pad or truncate back to original length
        if stretched.shape[1] > original_length:
            stretched = stretched[:, :original_length]
        elif stretched.shape[1] < original_length:
            padding = original_length - stretched.shape[1]
            stretched = F.pad(stretched, (0, padding))
        
        return stretched
    
    def pitch_shift(self, waveform: torch.Tensor, 
                   n_steps: float) -> torch.Tensor:
        """
        Pitch shift audio (change pitch without changing speed).
        
        Args:
            n_steps: Semitones to shift (-2 to +2 recommended)
        """
        if abs(n_steps) < 0.1:
            return waveform
        
        # Pitch shifting = time stretch + resample
        # Rate for time stretching
        rate = 2 ** (n_steps / 12.0)
        
        # Stretch
        stretched = self.time_stretch(waveform, rate)
        
        # Resample back to original rate
        import torch.nn.functional as F
        original_length = waveform.shape[1]
        resampled = F.interpolate(
            stretched.unsqueeze(0),
            size=original_length,
            mode='linear',
            align_corners=False
        ).squeeze(0)
        
        return resampled
    
    def add_room_reverb(self, waveform: torch.Tensor,
                       reverb_amount: float = 0.3) -> torch.Tensor:
        """
        Add simple room reverb effect.
        
        Args:
            reverb_amount: Amount of reverb (0.0 to 1.0)
        """
        if reverb_amount < 0.01:
            return waveform
        
        # Simple reverb using delayed copies
        delay_samples = [int(0.03 * self.sample_rate),   # 30ms
                        int(0.05 * self.sample_rate),   # 50ms
                        int(0.08 * self.sample_rate)]   # 80ms
        
        reverbed = waveform.clone()
        
        for delay in delay_samples:
            if delay < waveform.shape[1]:
                # Create delayed copy
                delayed = torch.zeros_like(waveform)
                delayed[:, delay:] = waveform[:, :-delay]
                # Add with decay
                reverbed += delayed * reverb_amount * 0.3
        
        # Normalize
        return self._safe_normalize(reverbed)
    
    def apply_bandpass_filter(self, waveform: torch.Tensor,
                             low_freq: float = 300,
                             high_freq: float = 3400) -> torch.Tensor:
        """
        Apply bandpass filter (simulate telephone/radio quality).
        
        Args:
            low_freq: Low cutoff frequency (Hz)
            high_freq: High cutoff frequency (Hz)
        """
        # Simple frequency domain filtering
        # Convert to frequency domain
        fft = torch.fft.rfft(waveform)
        
        # Create frequency axis
        freqs = torch.fft.rfftfreq(waveform.shape[1], 1/self.sample_rate)
        
        # Create bandpass mask
        mask = ((freqs >= low_freq) & (freqs <= high_freq)).float()
        
        # Apply filter
        filtered_fft = fft * mask.unsqueeze(0)
        
        # Convert back to time domain
        filtered = torch.fft.irfft(filtered_fft, n=waveform.shape[1])
        
        return filtered
    
    def create_variation(self, waveform: torch.Tensor, 
                        variation_type: str = 'medium') -> torch.Tensor:
        """
        Create single acoustic variation - ENHANCED VERSION.
        
        Args:
            variation_type: 'light', 'medium', 'heavy', or 'extreme'
        """
        varied = waveform.clone()
        
        if variation_type == 'light':
            # Light variations (subtle changes)
            noise_level = np.random.uniform(0.005, 0.015)
            amp_factor = np.random.uniform(0.85, 1.15)
            
            varied = self.add_gaussian_noise(varied, noise_level)
            varied = self.scale_amplitude(varied, amp_factor)
            
        elif variation_type == 'medium':
            # Medium variations (noticeable but natural)
            noise_level = np.random.uniform(0.01, 0.03)
            amp_factor = np.random.uniform(0.7, 1.3)
            time_rate = np.random.uniform(0.95, 1.05)
            
            # Randomly choose noise color
            noise_color = np.random.choice(['white', 'pink', 'brown'])
            varied = self.add_colored_noise(varied, noise_level, noise_color)
            varied = self.scale_amplitude(varied, amp_factor)
            varied = self.time_stretch(varied, time_rate)
            
            # 30% chance of adding reverb
            if np.random.random() < 0.3:
                reverb = np.random.uniform(0.1, 0.3)
                varied = self.add_room_reverb(varied, reverb)
        
        elif variation_type == 'heavy':
            # Heavy variations (significant changes)
            noise_level = np.random.uniform(0.02, 0.05)
            amp_factor = np.random.uniform(0.6, 1.4)
            time_rate = np.random.uniform(0.9, 1.1)
            pitch_steps = np.random.uniform(-1.0, 1.0)
            
            noise_color = np.random.choice(['white', 'pink', 'brown'])
            varied = self.add_colored_noise(varied, noise_level, noise_color)
            varied = self.scale_amplitude(varied, amp_factor)
            varied = self.time_stretch(varied, time_rate)
            varied = self.pitch_shift(varied, pitch_steps)
            
            # 50% chance of reverb
            if np.random.random() < 0.5:
                reverb = np.random.uniform(0.2, 0.4)
                varied = self.add_room_reverb(varied, reverb)
            
            # 20% chance of bandpass filter (telephone quality)
            if np.random.random() < 0.2:
                varied = self.apply_bandpass_filter(varied)
        
        elif variation_type == 'extreme':
            # Extreme variations (maximum diversity)
            noise_level = np.random.uniform(0.03, 0.08)
            amp_factor = np.random.uniform(0.5, 1.5)
            time_rate = np.random.uniform(0.85, 1.15)
            pitch_steps = np.random.uniform(-2.0, 2.0)
            
            noise_color = np.random.choice(['white', 'pink', 'brown'])
            varied = self.add_colored_noise(varied, noise_level, noise_color)
            varied = self.scale_amplitude(varied, amp_factor)
            varied = self.time_stretch(varied, time_rate)
            varied = self.pitch_shift(varied, pitch_steps)
            varied = self.add_room_reverb(varied, np.random.uniform(0.1, 0.5))
            
            # 40% chance of bandpass filter
            if np.random.random() < 0.4:
                low_freq = np.random.uniform(200, 500)
                high_freq = np.random.uniform(3000, 4000)
                varied = self.apply_bandpass_filter(varied, low_freq, high_freq)
        
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
