"""
Pre-generate large synthetic datasets using TTS for reuse across experiments.

This module creates comprehensive synthetic datasets upfront, allowing for:
1. Faster experiment iteration (no TTS during experiments)
2. Consistent synthetic samples across experiments
3. Analysis of synthetic vs real data quality
4. Separate evaluation of TTS quality
"""

import torch
import torchaudio
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from TTS.api import TTS
import logging
from dataclasses import dataclass
from tqdm import tqdm
import hashlib

from audio_processing import AudioProcessor, AudioVariationGenerator

logger = logging.getLogger(__name__)

@dataclass
class SyntheticDatasetConfig:
    """Configuration for synthetic dataset generation"""
    
    # Target words and variations
    target_keywords: List[str]
    samples_per_keyword: int = 1000  # Large pool for sampling
    
    # TTS configuration
    tts_model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"
    
    # Audio parameters
    sample_rate: int = 16000
    max_audio_length: int = 16000
    
    # Variation parameters
    text_variations_per_keyword: int = 4  # Different prosodic styles
    acoustic_variations_per_text: int = 3  # Acoustic modifications per text
    
    # Output configuration
    output_dir: str = './synthetic_datasets'
    dataset_name: str = 'gsc_synthetic'
    save_audio_files: bool = True  # Save individual wav files for inspection
    
    # Quality control
    validate_samples: bool = True
    min_energy_threshold: float = 1e-6  # Minimum audio energy
    
def create_large_synthetic_config() -> SyntheticDatasetConfig:
    """Create configuration for large synthetic dataset"""
    return SyntheticDatasetConfig(
        target_keywords=['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'],
        samples_per_keyword=1000,
        text_variations_per_keyword=6,
        acoustic_variations_per_text=4,
        dataset_name='gsc_synthetic_large'
    )

def create_quick_synthetic_config() -> SyntheticDatasetConfig:
    """Create configuration for quick testing"""
    return SyntheticDatasetConfig(
        target_keywords=['yes', 'no', 'up', 'down'],
        samples_per_keyword=100,
        text_variations_per_keyword=3,
        acoustic_variations_per_text=2,
        dataset_name='gsc_synthetic_quick'
    )

class TextVariationGenerator:
    """Generates diverse text variations for TTS input"""
    
    def __init__(self):
        self.prosodic_styles = [
            "",           # Neutral
            "!",          # Excited/emphatic
            "?",          # Questioning
            "...",        # Hesitant/trailing
            "!!!",        # Very excited
            ", please"    # Polite
        ]
        
        self.case_styles = [
            str.lower,    # lowercase
            str.upper,    # UPPERCASE  
            str.title,    # Title Case
        ]
    
    def generate_prosodic_variations(self, keyword: str, n_variations: int) -> List[str]:
        """Generate prosodic variations of a keyword"""
        variations = []
        
        # Base variations with prosodic markers
        for style in self.prosodic_styles[:n_variations]:
            if style == ", please":
                variation = keyword + style
            else:
                variation = keyword + style
            variations.append(variation)
        
        # Add case variations if we need more
        if len(variations) < n_variations:
            for case_fn in self.case_styles:
                if len(variations) >= n_variations:
                    break
                variations.append(case_fn(keyword))
        
        # Pad with base keyword if still needed
        while len(variations) < n_variations:
            variations.append(keyword)
        
        return variations[:n_variations]
    
    def create_training_phrases(self, keyword: str) -> List[str]:
        """Create phrases that include the keyword for more natural speech"""
        phrases = [
            keyword,                          # Isolated word
            f"Say {keyword}",                # Command context
            f"The answer is {keyword}",      # Sentence context
            f"{keyword}, exactly",           # Confirmation context
            f"I said {keyword}",             # Reported speech
        ]
        return phrases

class SyntheticDatasetGenerator:
    """Generates large-scale synthetic datasets using TTS"""
    
    def __init__(self, config: SyntheticDatasetConfig):
        self.config = config
        self.output_dir = Path(config.output_dir) / config.dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.audio_processor = AudioProcessor(config.sample_rate, config.max_audio_length)
        self.variation_generator = AudioVariationGenerator(config.sample_rate)
        self.text_generator = TextVariationGenerator()
        
        # Initialize TTS
        self._initialize_tts()
        
        # Storage for metadata
        self.generation_metadata = {
            'config': config.__dict__,
            'generation_stats': {},
            'failed_samples': [],
            'sample_metadata': []
        }
    
    def _initialize_tts(self):
        """Initialize TTS model"""
        logger.info(f"Initializing TTS model: {self.config.tts_model_name}")
        
        try:
            self.tts = TTS(model_name=self.config.tts_model_name)
            logger.info("TTS model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise RuntimeError(f"TTS initialization failed: {e}")
    
    def _generate_sample_id(self, keyword: str, text_variant: str, 
                           variation_idx: int) -> str:
        """Generate unique sample ID"""
        content = f"{keyword}_{text_variant}_{variation_idx}"
        hash_obj = hashlib.md5(content.encode())
        return f"{keyword}_{hash_obj.hexdigest()[:8]}"
    
    def synthesize_text_variant(self, text: str, sample_id: str) -> Optional[torch.Tensor]:
        """Synthesize single text variant to audio"""
        
        temp_file = self.output_dir / f"temp_{sample_id}.wav"
        
        try:
            # Synthesize with TTS
            self.tts.tts_to_file(text=text, file_path=str(temp_file))
            
            if not temp_file.exists():
                raise RuntimeError(f"TTS did not generate file for: {text}")
            
            # Load and preprocess
            waveform, sr = torchaudio.load(temp_file)
            processed = self.audio_processor.preprocess_audio(waveform, sr)
            
            # Validate quality
            if self._validate_audio_quality(processed, text):
                return processed
            else:
                logger.warning(f"Generated audio failed quality check: {text}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to synthesize '{text}': {e}")
            return None
            
        finally:
            # Cleanup temp file
            if temp_file.exists():
                temp_file.unlink()
    
    def _validate_audio_quality(self, audio: torch.Tensor, text: str) -> bool:
        """Validate synthesized audio meets quality standards"""
        
        # Basic tensor validation
        if not self.audio_processor.validate_audio(audio):
            return False
        
        # Check energy level
        energy = torch.sum(audio ** 2).item()
        if energy < self.config.min_energy_threshold:
            logger.debug(f"Low energy audio rejected for '{text}': {energy}")
            return False
        
        # Check for silence (all zeros)
        if torch.all(audio.abs() < 1e-6):
            logger.debug(f"Silent audio rejected for '{text}'")
            return False
        
        return True
    
    def generate_acoustic_variations(self, base_audio: torch.Tensor, 
                                   n_variations: int, sample_id: str) -> List[torch.Tensor]:
        """Generate acoustic variations from base sample"""
        variations = [base_audio]  # Include original
        
        for i in range(n_variations - 1):
            try:
                # Create variation with controlled randomness
                variation = self.variation_generator.create_variation(base_audio, 'light')
                
                if self._validate_audio_quality(variation, f"{sample_id}_var{i}"):
                    variations.append(variation)
                else:
                    logger.debug(f"Acoustic variation {i} failed validation for {sample_id}")
                    
            except Exception as e:
                logger.error(f"Failed to create acoustic variation {i} for {sample_id}: {e}")
                continue
        
        return variations
    
    def generate_keyword_dataset(self, keyword: str) -> Tuple[List[torch.Tensor], List[Dict]]:
        """Generate complete synthetic dataset for one keyword"""
        
        logger.info(f"Generating synthetic samples for keyword: '{keyword}'")
        
        all_samples = []
        sample_metadata = []
        
        # Generate text variations
        text_variations = self.text_generator.generate_prosodic_variations(
            keyword, self.config.text_variations_per_keyword
        )
        
        samples_per_text = self.config.samples_per_keyword // len(text_variations)
        
        with tqdm(total=len(text_variations), desc=f"Processing '{keyword}'") as pbar:
            
            for text_idx, text_variant in enumerate(text_variations):
                
                # Generate base sample from text
                sample_id = self._generate_sample_id(keyword, text_variant, text_idx)
                base_sample = self.synthesize_text_variant(text_variant, sample_id)
                
                if base_sample is None:
                    logger.warning(f"Failed to generate base sample for '{text_variant}'")
                    self.generation_metadata['failed_samples'].append({
                        'keyword': keyword,
                        'text': text_variant,
                        'type': 'synthesis_failed'
                    })
                    pbar.update(1)
                    continue
                
                # Generate acoustic variations
                acoustic_variations = self.generate_acoustic_variations(
                    base_sample, self.config.acoustic_variations_per_text, sample_id
                )
                
                # Store samples and metadata
                for var_idx, sample in enumerate(acoustic_variations):
                    variation_id = f"{sample_id}_var{var_idx}"
                    
                    all_samples.append(sample)
                    
                    metadata = {
                        'sample_id': variation_id,
                        'keyword': keyword,
                        'text_variant': text_variant,
                        'text_variation_idx': text_idx,
                        'acoustic_variation_idx': var_idx,
                        'is_base_sample': var_idx == 0,
                        'audio_energy': torch.sum(sample ** 2).item(),
                        'audio_max_amplitude': sample.abs().max().item()
                    }
                    
                    sample_metadata.append(metadata)
                    
                    # Save individual audio file if requested
                    if self.config.save_audio_files:
                        self._save_audio_sample(sample, variation_id)
                
                pbar.update(1)
        
        logger.info(f"Generated {len(all_samples)} samples for keyword '{keyword}'")
        return all_samples, sample_metadata
    
    def _save_audio_sample(self, audio: torch.Tensor, sample_id: str):
        """Save individual audio sample to file"""
        audio_dir = self.output_dir / 'audio_files'
        audio_dir.mkdir(exist_ok=True)
        
        file_path = audio_dir / f"{sample_id}.wav"
        
        try:
            torchaudio.save(file_path, audio, self.config.sample_rate)
        except Exception as e:
            logger.error(f"Failed to save audio file {sample_id}: {e}")
    
    def generate_complete_dataset(self) -> str:
        """Generate complete synthetic dataset for all keywords"""
        
        logger.info(f"Starting synthetic dataset generation: {self.config.dataset_name}")
        logger.info(f"Target: {self.config.samples_per_keyword} samples × {len(self.config.target_keywords)} keywords")
        
        all_audio_samples = []
        all_metadata = []
        
        # Generate for each keyword
        for keyword in self.config.target_keywords:
            try:
                keyword_samples, keyword_metadata = self.generate_keyword_dataset(keyword)
                
                all_audio_samples.extend(keyword_samples)
                all_metadata.extend(keyword_metadata)
                
                self.generation_metadata['generation_stats'][keyword] = {
                    'target_samples': self.config.samples_per_keyword,
                    'generated_samples': len(keyword_samples),
                    'success_rate': len(keyword_samples) / self.config.samples_per_keyword
                }
                
            except Exception as e:
                logger.error(f"Failed to generate samples for keyword '{keyword}': {e}")
                self.generation_metadata['failed_samples'].append({
                    'keyword': keyword,
                    'type': 'keyword_generation_failed',
                    'error': str(e)
                })
        
        # Save complete dataset
        dataset_path = self._save_dataset(all_audio_samples, all_metadata)
        
        # Generate quality report
        self._generate_quality_report(all_metadata)
        
        logger.info(f"Synthetic dataset generation completed: {len(all_audio_samples)} total samples")
        logger.info(f"Dataset saved to: {dataset_path}")
        
        return str(dataset_path)
    
    def _save_dataset(self, audio_samples: List[torch.Tensor], 
                     metadata: List[Dict]) -> Path:
        """Save complete synthetic dataset"""
        
        # Save audio tensors
        audio_tensor = torch.stack(audio_samples)
        audio_file = self.output_dir / 'synthetic_audio.pt'
        torch.save(audio_tensor, audio_file)
        
        # Save metadata
        metadata_df = pd.DataFrame(metadata)
        metadata_file = self.output_dir / 'synthetic_metadata.csv'
        metadata_df.to_csv(metadata_file, index=False)
        
        # Save generation metadata
        generation_file = self.output_dir / 'generation_metadata.json'
        with open(generation_file, 'w') as f:
            json.dump(self.generation_metadata, f, indent=2)
        
        # Save config
        config_file = self.output_dir / 'dataset_config.json'
        with open(config_file, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Create dataset info file
        info = {
            'dataset_name': self.config.dataset_name,
            'total_samples': len(audio_samples),
            'keywords': self.config.target_keywords,
            'samples_per_keyword': {
                keyword: len([m for m in metadata if m['keyword'] == keyword])
                for keyword in self.config.target_keywords
            },
            'audio_format': {
                'tensor_shape': list(audio_tensor.shape),
                'sample_rate': self.config.sample_rate,
                'duration_seconds': self.config.max_audio_length / self.config.sample_rate
            },
            'files': {
                'audio_tensor': 'synthetic_audio.pt',
                'metadata': 'synthetic_metadata.csv',
                'generation_log': 'generation_metadata.json',
                'config': 'dataset_config.json'
            }
        }
        
        info_file = self.output_dir / 'dataset_info.json'
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        return self.output_dir
    
    def _generate_quality_report(self, metadata: List[Dict]):
        """Generate quality assessment report"""
        
        metadata_df = pd.DataFrame(metadata)
        
        # Energy statistics
        energy_stats = metadata_df['audio_energy'].describe()
        
        # Amplitude statistics  
        amplitude_stats = metadata_df['audio_max_amplitude'].describe()
        
        # Success rates per keyword
        success_rates = {}
        for keyword in self.config.target_keywords:
            keyword_meta = metadata_df[metadata_df['keyword'] == keyword]
            success_rates[keyword] = len(keyword_meta) / self.config.samples_per_keyword
        
        quality_report = {
            'generation_summary': {
                'total_requested': len(self.config.target_keywords) * self.config.samples_per_keyword,
                'total_generated': len(metadata),
                'overall_success_rate': len(metadata) / (len(self.config.target_keywords) * self.config.samples_per_keyword)
            },
            'success_rates_per_keyword': success_rates,
            'audio_quality_stats': {
                'energy_statistics': energy_stats.to_dict(),
                'amplitude_statistics': amplitude_stats.to_dict(),
                'failed_samples': len(self.generation_metadata['failed_samples'])
            }
        }
        
        # Save quality report
        quality_file = self.output_dir / 'quality_report.json'
        with open(quality_file, 'w') as f:
            json.dump(quality_report, f, indent=2)
        
        # Log summary
        logger.info(f"Quality Report Summary:")
        logger.info(f"  Success rate: {quality_report['generation_summary']['overall_success_rate']:.1%}")
        logger.info(f"  Total samples: {quality_report['generation_summary']['total_generated']}")
        logger.info(f"  Failed samples: {quality_report['audio_quality_stats']['failed_samples']}")

class SyntheticDatasetLoader:
    """Loads pre-generated synthetic datasets for experiments"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self._validate_dataset_path()
        
        # Load dataset components
        self.info = self._load_dataset_info()
        self.audio_tensor = self._load_audio_tensor()
        self.metadata_df = self._load_metadata()
        
        logger.info(f"Loaded synthetic dataset: {self.info['total_samples']} samples")
    
    def _validate_dataset_path(self):
        """Validate that dataset path contains required files"""
        required_files = ['dataset_info.json', 'synthetic_audio.pt', 'synthetic_metadata.csv']
        
        for file_name in required_files:
            file_path = self.dataset_path / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Required dataset file missing: {file_path}")
    
    def _load_dataset_info(self) -> Dict:
        """Load dataset information"""
        with open(self.dataset_path / 'dataset_info.json', 'r') as f:
            return json.load(f)
    
    def _load_audio_tensor(self) -> torch.Tensor:
        """Load pre-generated audio tensor"""
        return torch.load(self.dataset_path / 'synthetic_audio.pt')
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load sample metadata"""
        return pd.read_csv(self.dataset_path / 'synthetic_metadata.csv')
    
    def sample_synthetic_data(self, keyword: str, n_samples: int, 
                            random_state: Optional[int] = None) -> Tuple[List[torch.Tensor], List[str]]:
        """Sample synthetic data for a specific keyword"""
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Get samples for this keyword
        keyword_metadata = self.metadata_df[self.metadata_df['keyword'] == keyword]
        
        if len(keyword_metadata) == 0:
            logger.error(f"No synthetic samples found for keyword: {keyword}")
            return [], []
        
        # Sample indices
        available_samples = len(keyword_metadata)
        if n_samples > available_samples:
            logger.warning(f"Requested {n_samples} samples for '{keyword}', "
                         f"only {available_samples} available")
            n_samples = available_samples
        
        sampled_indices = np.random.choice(
            keyword_metadata.index, n_samples, replace=False
        )
        
        # Extract audio samples
        sampled_audio = [self.audio_tensor[idx] for idx in sampled_indices]
        sampled_labels = ['keyword'] * len(sampled_audio)
        
        logger.info(f"Sampled {len(sampled_audio)} synthetic samples for '{keyword}'")
        return sampled_audio, sampled_labels
    
    def get_balanced_synthetic_samples(self, keywords: List[str], 
                                     samples_per_keyword: int,
                                     random_state: Optional[int] = None) -> Tuple[List[torch.Tensor], List[str]]:
        """Get balanced synthetic samples across keywords"""
        
        all_audio = []
        all_labels = []
        
        for keyword in keywords:
            keyword_audio, keyword_labels = self.sample_synthetic_data(
                keyword, samples_per_keyword, random_state
            )
            all_audio.extend(keyword_audio)
            all_labels.extend(keyword_labels)
        
        # Shuffle combined dataset
        if random_state is not None:
            np.random.seed(random_state)
        
        indices = list(range(len(all_audio)))
        np.random.shuffle(indices)
        
        shuffled_audio = [all_audio[i] for i in indices]
        shuffled_labels = [all_labels[i] for i in indices]
        
        return shuffled_audio, shuffled_labels
    
    def get_dataset_statistics(self) -> Dict:
        """Get comprehensive dataset statistics"""
        
        stats = {
            'total_samples': len(self.metadata_df),
            'samples_per_keyword': self.metadata_df['keyword'].value_counts().to_dict(),
            'unique_text_variants': self.metadata_df['text_variant'].nunique(),
            'unique_keywords': self.metadata_df['keyword'].nunique(),
            'base_samples': len(self.metadata_df[self.metadata_df['is_base_sample'] == True]),
            'variation_samples': len(self.metadata_df[self.metadata_df['is_base_sample'] == False])
        }
        
        # Audio quality statistics
        stats['audio_quality'] = {
            'mean_energy': self.metadata_df['audio_energy'].mean(),
            'energy_std': self.metadata_df['audio_energy'].std(),
            'mean_max_amplitude': self.metadata_df['audio_max_amplitude'].mean(),
            'amplitude_std': self.metadata_df['audio_max_amplitude'].std()
        }
        
        return stats

# Command line interface for dataset generation
def main():
    """Main function for synthetic dataset generation"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic TTS datasets')
    parser.add_argument('--config', choices=['quick', 'large', 'custom'],
                       default='quick', help='Dataset configuration')
    parser.add_argument('--keywords', nargs='+', 
                       help='Custom keywords (space-separated)')
    parser.add_argument('--samples-per-keyword', type=int,
                       help='Number of samples per keyword')
    parser.add_argument('--output-dir', type=str, default='./synthetic_datasets',
                       help='Output directory for datasets')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    if args.config == 'quick':
        config = create_quick_synthetic_config()
    elif args.config == 'large':
        config = create_large_synthetic_config()
    elif args.config == 'custom':
        config = SyntheticDatasetConfig(
            target_keywords=args.keywords or ['yes', 'no'],
            samples_per_keyword=args.samples_per_keyword or 500,
            output_dir=args.output_dir
        )
    
    # Override with command line arguments
    if args.keywords:
        config.target_keywords = args.keywords
    if args.samples_per_keyword:
        config.samples_per_keyword = args.samples_per_keyword
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Generate dataset
    generator = SyntheticDatasetGenerator(config)
    dataset_path = generator.generate_complete_dataset()
    
    # Load and print statistics
    loader = SyntheticDatasetLoader(dataset_path)
    stats = loader.get_dataset_statistics()
    
    print(f"\nDataset generation completed!")
    print(f"Location: {dataset_path}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Keywords: {list(stats['samples_per_keyword'].keys())}")
    print(f"Samples per keyword: {list(stats['samples_per_keyword'].values())}")

if __name__ == "__main__":
    main()
