"""
Experimental configuration for class imbalance augmentation study.
"""
from dataclasses import dataclass
from typing import List

@dataclass
class ExperimentConfig:
    """Configuration for systematic class imbalance experiments"""
    
    # Dataset configuration
    dataset_version: str = 'v2'
    dataset_root: str = field(default_factory=lambda: get_colab_path('./data', 'COLAB_DATA_DIR'))
    target_keywords: List[str] = None
    
    # Experimental grid
    dataset_sizes: List[str] = None
    imbalance_ratios: List[float] = None
    augmentation_methods: List[str] = None
    
    # Training configuration
    n_trials: int = 3
    n_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    
    # Audio configuration
    sample_rate: int = 16000
    max_audio_length: int = 16000
    
    # TTS configuration
    tts_model_name: str = "gtts"
    
    # Adversarial configuration
    fgsm_epsilon: float = 0.01
    
    # Output configuration
    save_dir: str = field(default_factory=lambda: get_colab_path('./results', 'COLAB_RESULTS_DIR'))
    save_intermediate: bool = True
    
    synthetic_dataset_path: str = field(default_factory=lambda: os.path.join(
        get_colab_path('./synthetic_datasets', 'COLAB_SYNTHETIC_DIR'),
        'gsc_synthetic_large'
    )) 
    
    def __post_init__(self):
        """Set default values"""
        if self.target_keywords is None:
            self.target_keywords = ['yes', 'no', 'up', 'down']
            
        if self.dataset_sizes is None:
            self.dataset_sizes = ['small', 'medium', 'large']
            
        if self.imbalance_ratios is None:
            self.imbalance_ratios = [0.1, 0.2, 0.5, 1.0]
            
        if self.augmentation_methods is None:
            self.augmentation_methods = ['none', 'adversarial', 'tts', 'combined']


def create_quick_config() -> ExperimentConfig:
    """Create configuration for quick testing"""
    return ExperimentConfig(
        dataset_version='v2',
        target_keywords=['yes', 'no'],
        dataset_sizes=['small', 'medium'],
        imbalance_ratios=[0.1, 0.5, 1.0],
        augmentation_methods=['none', 'synthetic', 'adversarial'],
        n_trials=2,
        n_epochs=10,
        synthetic_dataset_path=os.path.join(
            get_colab_path('./synthetic_datasets', 'COLAB_SYNTHETIC_DIR'),
            'gsc_synthetic_quick'  # Note: 'quick' not 'large' for the quick config
        )
    )

def create_full_config() -> ExperimentConfig:
    """Create configuration for comprehensive study"""
    return ExperimentConfig(
        dataset_version='v2',
        target_keywords=['yes', 'no', 'up', 'down'],
        dataset_sizes=['small', 'medium', 'large', 'full'],
        imbalance_ratios=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
        augmentation_methods=['none', 'adversarial', 'tts', 'combined'],
        n_trials=5,
        n_epochs=25
    )


def create_ablation_config() -> ExperimentConfig:
    """Create configuration for ablation studies"""
    return ExperimentConfig(
        dataset_version='v2',
        target_keywords=['yes', 'no'],
        dataset_sizes=['medium'],
        imbalance_ratios=[0.1, 0.2],
        augmentation_methods=['none', 'tts', 'adversarial', 'combined'],
        n_trials=5,
        n_epochs=30,
        save_dir='./ablation_results'
    )
