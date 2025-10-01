"""
Pre-generate large synthetic datasets using gTTS or Bark for reuse across experiments.
"""

import torch
import torchaudio
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from gtts import gTTS
import logging
from dataclasses import dataclass
from tqdm import tqdm
import hashlib
import math

from audio_processing import AudioProcessor, AudioVariationGenerator
from measure_gsc_energy import load_gsc_energy_profile

logger = logging.getLogger(__name__)

# Try to import Bark
try:
    from bark import SAMPLE_RATE as BARK_SAMPLE_RATE, generate_audio, preload_models
    import scipy.io.wavfile
    BARK_AVAILABLE = True
except ImportError:
    BARK_AVAILABLE = False
    logger.warning("Bark not available. Install with: pip install git+https://github.com/suno-ai/bark.git")


DEFAULT_TARGET_RMS = 0.1

@dataclass
class SyntheticDatasetConfig:
    """Configuration for synthetic dataset generation"""
    target_keywords: List[str]
    samples_per_keyword: int = 500
    sample_rate: int = 16000
    max_audio_length: int = 16000
    output_dir: str = './synthetic_datasets'
    dataset_name: str = 'gsc_synthetic'
    save_audio_files: bool = True
    min_energy_threshold: float = 1e-6
    target_rms: Optional[float] = None
    energy_profile_path: str = './synthetic_datasets/gsc_energy_profile.json'
    tts_engine: str = 'gtts'  # 'gtts' or 'bark'


def calculate_variation_counts(target_samples: int) -> Tuple[int, int]:
    """Calculate text and acoustic variations needed for target sample count."""
    text_variations = max(5, math.ceil(math.sqrt(target_samples)))
    acoustic_variations = max(3, math.ceil(target_samples / text_variations))
    return text_variations, acoustic_variations


def generate_prosodic_variations(keyword: str, n_variations: int) -> List[str]:
    """Generate prosodic variations of a keyword."""
    variations = [
        keyword,           # Neutral
        keyword + "!",     # Excited
        keyword + "?",     # Questioning
        keyword + "...",   # Hesitant
        keyword.upper(),   # Loud
        keyword.lower(),   # Soft
    ]
    
    # Extend list if needed
    while len(variations) < n_variations:
        variations.append(keyword)
    
    return variations[:n_variations]


def generate_sample_id(keyword: str, text_variant: str, variation_idx: int) -> str:
    """Generate unique sample ID."""
    content = f"{keyword}_{text_variant}_{variation_idx}"
    hash_obj = hashlib.md5(content.encode())
    return f"{keyword}_{hash_obj.hexdigest()[:8]}"


def validate_audio_quality(audio: torch.Tensor, min_energy: float) -> bool:
    """Validate audio meets quality standards."""
    if torch.isnan(audio).any() or torch.isinf(audio).any():
        return False
    
    energy = torch.sum(audio ** 2).item()
    if energy < min_energy:
        return False
    
    if torch.all(audio.abs() < 1e-6):
        return False
    
    return True


class TTSEngine:
    """Manages different TTS engines"""
    
    def __init__(self, engine_type: str = 'gtts'):
        self.engine_type = engine_type
        self.bark_initialized = False
        
        if engine_type == 'bark':
            if not BARK_AVAILABLE:
                raise RuntimeError(
                    "Bark not installed. Install with:\n"
                    "pip install git+https://github.com/suno-ai/bark.git"
                )
            logger.info("Initializing Bark TTS (first run will download models ~2GB)")
            logger.info("This may take 2-5 minutes on first use...")
            
            # Preload models (downloads on first run)
            preload_models()
            self.bark_initialized = True
            
            logger.info("Bark TTS initialized successfully!")
        else:
            logger.info("Using gTTS engine")
    
    def synthesize(self, text: str, output_path: str):
        """Synthesize text to audio file"""
        if self.engine_type == 'gtts':
            self._synthesize_gtts(text, output_path)
        elif self.engine_type == 'bark':
            self._synthesize_bark(text, output_path)
        else:
            raise ValueError(f"Unknown TTS engine: {self.engine_type}")
    
    def _synthesize_gtts(self, text: str, output_path: str):
        """Synthesize using gTTS"""
        slow = '...' in text or len(text) > 8
        tts = gTTS(text=text, lang='en', slow=slow)
        tts.save(output_path)
    
    def _synthesize_bark(self, text: str, output_path: str):
        """Synthesize using Bark"""
        # Clean text - Bark handles punctuation naturally
        clean_text = text.strip()
        
        # Bark uses special markers for prosody
        if text.isupper():
            clean_text = f"[LOUD] {clean_text.lower()}"
        elif '...' in text:
            clean_text = clean_text.replace('...', '...')  # Bark handles pauses
        
        try:
            # Generate audio with Bark
            # Use voice preset for consistency (v2/en_speaker_6 is clear and neutral)
            audio_array = generate_audio(
                clean_text,
                history_prompt="v2/en_speaker_6"  # Consistent neutral English voice
            )
            
            # Bark outputs at 24kHz, save as wav
            scipy.io.wavfile.write(
                output_path,
                rate=BARK_SAMPLE_RATE,
                data=audio_array
            )
            
        except Exception as e:
            # Fallback: try without voice preset
            logger.warning(f"Bark synthesis failed with preset, trying default: {e}")
            audio_array = generate_audio(clean_text)
            scipy.io.wavfile.write(
                output_path,
                rate=BARK_SAMPLE_RATE,
                data=audio_array
            )


class SyntheticDatasetGenerator:
    """Generates large-scale synthetic datasets using TTS"""
    
    def __init__(self, config: SyntheticDatasetConfig):
        self.config = config
        self.output_dir = Path(config.output_dir) / config.dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
        self.audio_processor = AudioProcessor(config.sample_rate, config.max_audio_length)
        self.variation_generator = AudioVariationGenerator(config.sample_rate)
    
        # Initialize TTS engine
        logger.info(f"Initializing TTS engine: {config.tts_engine}")
        self.tts_engine = TTSEngine(config.tts_engine)
        
        # Calculate actual variations needed
        self.text_variations, self.acoustic_variations = calculate_variation_counts(
            config.samples_per_keyword
        )
    
        # Auto-load target RMS from GSC energy profile
        if config.target_rms is None:
            try:
                energy_profile = load_gsc_energy_profile(config.energy_profile_path)
                self.target_rms = energy_profile['recommended_target_rms']
                logger.info(f"Auto-loaded target_rms={self.target_rms:.6f} from GSC profile")
            except FileNotFoundError:
                logger.warning("GSC energy profile not found. Using default target_rms=0.1")
                logger.warning("Run 'python measure_gsc_energy.py' to generate profile")
                self.target_rms = DEFAULT_TARGET_RMS
        else:
            self.target_rms = config.target_rms
    
        self.metadata = {
            'config': config.__dict__, 
            'failed_samples': [], 
            'sample_metadata': [],
            'tts_engine': config.tts_engine
        }
    
        logger.info(f"Generator initialized with {config.tts_engine.upper()} engine")
        logger.info(f"  {self.text_variations} text × {self.acoustic_variations} acoustic = "
                   f"{self.text_variations * self.acoustic_variations} samples/keyword")
        logger.info(f"  Target RMS: {self.target_rms:.6f}")


    def synthesize_text_variant(self, text: str, sample_id: str) -> Optional[torch.Tensor]:
        """Synthesize single text variant to audio."""
        temp_file = self.output_dir / f"temp_{sample_id}.wav"
    
        try:
            self.tts_engine.synthesize(text, str(temp_file))
        
            if not temp_file.exists():
                raise RuntimeError(f"TTS did not generate file for: {text}")
        
            waveform, sr = torchaudio.load(temp_file)
            processed = self.audio_processor.preprocess_audio(waveform, sr)
        
            # Normalize to target RMS energy (auto-loaded from GSC profile)
            processed = self.audio_processor.normalize_to_target_rms(
                processed, target_rms=self.target_rms
            )
        
            if validate_audio_quality(processed, self.config.min_energy_threshold):
                return processed
            else:
                logger.warning(f"Quality check failed: {text}")
                return None
            
        except Exception as e:
            logger.error(f"Failed to synthesize '{text}': {e}")
            return None
        finally:
            if temp_file.exists():
                temp_file.unlink()
 
 
    def create_acoustic_variations(self, base_audio: torch.Tensor, 
                                  n_variations: int, sample_id: str) -> List[torch.Tensor]:
        """Generate acoustic variations from base sample."""
        variations = [base_audio]
        
        for i in range(n_variations - 1):
            try:
                variation = self.variation_generator.create_variation(base_audio, 'light')
                
                if validate_audio_quality(variation, self.config.min_energy_threshold):
                    variations.append(variation)
            except Exception as e:
                logger.error(f"Acoustic variation {i} failed for {sample_id}: {e}")
        
        return variations
    
    def generate_keyword_samples(self, keyword: str) -> Tuple[List[torch.Tensor], List[Dict]]:
        """Generate all samples for one keyword."""
        logger.info(f"Generating samples for '{keyword}'")
        
        all_samples = []
        sample_metadata = []
        
        text_variations = generate_prosodic_variations(keyword, self.text_variations)
        
        with tqdm(total=len(text_variations), desc=f"'{keyword}'") as pbar:
            for text_idx, text_variant in enumerate(text_variations):
                sample_id = generate_sample_id(keyword, text_variant, text_idx)
                base_sample = self.synthesize_text_variant(text_variant, sample_id)
                
                if base_sample is None:
                    self.metadata['failed_samples'].append({
                        'keyword': keyword, 'text': text_variant, 'type': 'synthesis_failed'
                    })
                    pbar.update(1)
                    continue
                
                acoustic_variations = self.create_acoustic_variations(
                    base_sample, self.acoustic_variations, sample_id
                )
                
                for var_idx, sample in enumerate(acoustic_variations):
                    variation_id = f"{sample_id}_var{var_idx}"
                    all_samples.append(sample)
                    
                    sample_metadata.append({
                        'sample_id': variation_id,
                        'keyword': keyword,
                        'text_variant': text_variant,
                        'text_variation_idx': text_idx,
                        'acoustic_variation_idx': var_idx,
                        'is_base_sample': var_idx == 0,
                        'audio_energy': torch.sum(sample ** 2).item(),
                        'audio_max_amplitude': sample.abs().max().item()
                    })
                    
                    if self.config.save_audio_files:
                        self._save_audio_file(sample, variation_id)
                
                pbar.update(1)
        
        logger.info(f"Generated {len(all_samples)} samples for '{keyword}'")
        return all_samples, sample_metadata
    
    def _save_audio_file(self, audio: torch.Tensor, sample_id: str):
        """Save individual audio sample."""
        audio_dir = self.output_dir / 'audio_files'
        audio_dir.mkdir(exist_ok=True)
        file_path = audio_dir / f"{sample_id}.wav"
        
        try:
            torchaudio.save(file_path, audio, self.config.sample_rate)
        except Exception as e:
            logger.error(f"Failed to save {sample_id}: {e}")
    
    def generate_complete_dataset(self) -> str:
        """Generate complete synthetic dataset for all keywords."""
        logger.info(f"Starting dataset generation: {self.config.dataset_name}")
        logger.info(f"Target: {self.config.samples_per_keyword} samples × "
                   f"{len(self.config.target_keywords)} keywords")
        
        all_audio = []
        all_metadata = []
        
        for keyword in self.config.target_keywords:
            try:
                samples, metadata = self.generate_keyword_samples(keyword)
                all_audio.extend(samples)
                all_metadata.extend(metadata)
            except Exception as e:
                logger.error(f"Failed for keyword '{keyword}': {e}")
                self.metadata['failed_samples'].append({
                    'keyword': keyword, 'type': 'keyword_failed', 'error': str(e)
                })
        
        dataset_path = self._save_dataset(all_audio, all_metadata)
        self._save_quality_report(all_metadata)
        
        logger.info(f"Dataset complete: {len(all_audio)} total samples at {dataset_path}")
        return str(dataset_path)
    
    def _save_dataset(self, audio_samples: List[torch.Tensor], 
                     metadata: List[Dict]) -> Path:
        """Save complete dataset."""
        # Save audio tensor
        audio_tensor = torch.stack(audio_samples)
        torch.save(audio_tensor, self.output_dir / 'synthetic_audio.pt')
        
        # Save metadata
        pd.DataFrame(metadata).to_csv(self.output_dir / 'synthetic_metadata.csv', index=False)
        
        # Save generation metadata
        with open(self.output_dir / 'generation_metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save dataset info
        info = {
            'dataset_name': self.config.dataset_name,
            'total_samples': len(audio_samples),
            'keywords': self.config.target_keywords,
            'tts_engine': self.config.tts_engine,
            'samples_per_keyword': {
                kw: len([m for m in metadata if m['keyword'] == kw])
                for kw in self.config.target_keywords
            },
            'audio_format': {
                'tensor_shape': list(audio_tensor.shape),
                'sample_rate': self.config.sample_rate,
                'duration_seconds': self.config.max_audio_length / self.config.sample_rate
            },
            'files': {
                'audio_tensor': 'synthetic_audio.pt',
                'metadata': 'synthetic_metadata.csv',
                'generation_log': 'generation_metadata.json'
            }
        }
        
        with open(self.output_dir / 'dataset_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        return self.output_dir
    
    def _save_quality_report(self, metadata: List[Dict]):
        """Generate quality report."""
        df = pd.DataFrame(metadata)
        
        report = {
            'total_generated': len(metadata),
            'total_requested': len(self.config.target_keywords) * self.config.samples_per_keyword,
            'success_rate': len(metadata) / (len(self.config.target_keywords) * self.config.samples_per_keyword),
            'samples_per_keyword': df['keyword'].value_counts().to_dict(),
            'failed_samples': len(self.metadata['failed_samples']),
            'energy_stats': df['audio_energy'].describe().to_dict()
        }
        
        with open(self.output_dir / 'quality_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Quality: {report['success_rate']:.1%} success, "
                   f"{report['failed_samples']} failed")


class SyntheticDatasetLoader:
    """Loads pre-generated synthetic datasets."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self._validate_files()
        
        with open(self.dataset_path / 'dataset_info.json', 'r') as f:
            self.info = json.load(f)
        
        self.audio_tensor = torch.load(self.dataset_path / 'synthetic_audio.pt')
        self.metadata_df = pd.read_csv(self.dataset_path / 'synthetic_metadata.csv')
        
        logger.info(f"Loaded dataset: {self.info['total_samples']} samples")
    
    def _validate_files(self):
        """Validate required files exist."""
        required = ['dataset_info.json', 'synthetic_audio.pt', 'synthetic_metadata.csv']
        for file_name in required:
            if not (self.dataset_path / file_name).exists():
                raise FileNotFoundError(f"Missing: {self.dataset_path / file_name}")
    
    def sample_keyword_data(self, keyword: str, n_samples: int, 
                           random_state: Optional[int] = None) -> Tuple[List[torch.Tensor], List[str]]:
        """Sample synthetic data for a keyword."""
        if random_state is not None:
            np.random.seed(random_state)
        
        keyword_meta = self.metadata_df[self.metadata_df['keyword'] == keyword]
        
        if len(keyword_meta) == 0:
            logger.error(f"No samples for keyword: {keyword}")
            return [], []
        
        n_samples = min(n_samples, len(keyword_meta))
        sampled_indices = np.random.choice(keyword_meta.index, n_samples, replace=False)
        
        sampled_audio = [self.audio_tensor[idx] for idx in sampled_indices]
        sampled_labels = ['keyword'] * len(sampled_audio)
        
        return sampled_audio, sampled_labels
    
    def get_balanced_samples(self, keywords: List[str], samples_per_keyword: int,
                           random_state: Optional[int] = None) -> Tuple[List[torch.Tensor], List[str]]:
        """Get balanced samples across keywords."""
        all_audio = []
        all_labels = []
        
        for keyword in keywords:
            audio, labels = self.sample_keyword_data(keyword, samples_per_keyword, random_state)
            all_audio.extend(audio)
            all_labels.extend(labels)
        
        if random_state is not None:
            np.random.seed(random_state)
        
        indices = list(range(len(all_audio)))
        np.random.shuffle(indices)
        
        return [all_audio[i] for i in indices], [all_labels[i] for i in indices]
    
    def get_dataset_statistics(self) -> Dict:
        """Get dataset statistics."""
        return {
            'total_samples': len(self.metadata_df),
            'samples_per_keyword': self.metadata_df['keyword'].value_counts().to_dict(),
            'unique_text_variants': self.metadata_df['text_variant'].nunique(),
            'base_samples': len(self.metadata_df[self.metadata_df['is_base_sample'] == True]),
            'audio_quality': {
                'mean_energy': self.metadata_df['audio_energy'].mean(),
                'mean_max_amplitude': self.metadata_df['audio_max_amplitude'].mean()
            }
        }


def check_dataset_exists(dataset_path: str) -> Dict:
    """Check if dataset exists and return info."""
    info_file = Path(dataset_path) / 'dataset_info.json'
    
    if info_file.exists():
        with open(info_file, 'r') as f:
            return {'exists': True, 'info': json.load(f)}
    return {'exists': False, 'info': None}


def generate_comprehensive_dataset(keywords: List[str], samples_per_keyword: int,
                                  output_dir: str, dataset_name: str = 'gsc_synthetic',
                                  tts_engine: str = 'gtts') -> str:
    """Generate comprehensive synthetic dataset."""
    config = SyntheticDatasetConfig(
        target_keywords=keywords,
        samples_per_keyword=samples_per_keyword,
        output_dir=output_dir,
        dataset_name=dataset_name,
        save_audio_files=True,
        tts_engine=tts_engine
    )
    
    generator = SyntheticDatasetGenerator(config)
    return generator.generate_complete_dataset()


def main():
    """CLI for dataset generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic TTS datasets')
    parser.add_argument('--keywords', nargs='+', default=['yes', 'no', 'up', 'down'])
    parser.add_argument('--samples', type=int, default=500,
                       help='Samples per keyword to generate')
    parser.add_argument('--output-dir', type=str, default='./synthetic_datasets')
    parser.add_argument('--name', type=str, default='gsc_synthetic')
    parser.add_argument('--engine', choices=['gtts', 'bark'], default='gtts',
                       help='TTS engine (gtts=fast/robotic, bark=slow/natural/Colab-compatible)')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    print(f"\n{'='*60}")
    print(f"Generating dataset with {args.engine.upper()} engine")
    if args.engine == 'bark':
        print("NOTE: Bark is MUCH slower (~10-15 sec per sample)")
        print("      First run will download ~2GB of models")
        print("      But it works great in Google Colab!")
        print(f"      Estimated time: ~{args.samples * len(args.keywords) * 15 / 3600:.1f} hours")
    else:
        print(f"      Estimated time: ~{args.samples * len(args.keywords) * 3 / 60:.1f} minutes")
    print(f"{'='*60}\n")
    
    dataset_path = generate_comprehensive_dataset(
        keywords=args.keywords,
        samples_per_keyword=args.samples,
        output_dir=args.output_dir,
        dataset_name=args.name,
        tts_engine=args.engine
    )
    
    loader = SyntheticDatasetLoader(dataset_path)
    stats = loader.get_dataset_statistics()
    
    print(f"\n{'='*60}")
    print(f"✓ Dataset generated with {args.engine.upper()}: {dataset_path}")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Samples per keyword: {stats['samples_per_keyword']}")
    print(f"  Mean energy: {stats['audio_quality']['mean_energy']:.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
