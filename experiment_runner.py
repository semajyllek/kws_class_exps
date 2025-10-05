"""
Fixed experiment runner with corrected adversarial training baseline.
"""
import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging

from config import ExperimentConfig
from dataset_manager import GSCDatasetManager
from model_training import ModelTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train_balanced_baseline_for_adversarial(dataset_manager: GSCDatasetManager,
                                           config: ExperimentConfig,
                                           dataset_size: str) -> torch.nn.Module:
    """
    Train a BALANCED baseline model for adversarial generation.
    
    CRITICAL FIX: The baseline must be trained on BALANCED data so it actually
    learns to distinguish keywords from non-keywords.
    
    Args:
        dataset_manager: Dataset manager instance
        config: Experiment configuration
        dataset_size: Size of dataset to use ('small', 'medium', 'large')
    
    Returns:
        Trained baseline model
    """
    logger.info(f"Training BALANCED baseline model for adversarial generation ({dataset_size})")
    
    try:
        # Load dataset
        audio_files, labels = dataset_manager.load_dataset(dataset_size)
        
        # Create BALANCED split (ratio=1.0 means 50/50 split)
        audio_files, labels = dataset_manager.create_imbalanced_split(
            audio_files, labels, 
            config.target_keywords, 
            imbalance_ratio=1.0  # ← BALANCED (50% positive, 50% negative)
        )
        
        # Train/test split
        train_audio, train_labels, test_audio, test_labels = dataset_manager.split_train_test(
            audio_files, labels, test_ratio=0.2, random_state=42
        )
        
        # Train model with same config as experiments
        trainer = ModelTrainer()
        
        logger.info(f"Training baseline: {len(train_audio)} train, {len(test_audio)} test samples")
        
        metrics, baseline_model = trainer.full_training_pipeline(
            train_audio, train_labels, test_audio, test_labels, config
        )
        
        logger.info(f"✓ Baseline trained successfully (F1={metrics['f1_keyword']:.3f})")
        
        return baseline_model
        
    except Exception as e:
        logger.error(f"Failed to train baseline model: {e}")
        return None


class ExperimentRunner:
    """Orchestrates systematic experimental evaluation with FIXED adversarial augmentation"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure dataset root directory exists
        Path(config.dataset_root).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.dataset_manager = GSCDatasetManager(
            root_dir=config.dataset_root,
            version=config.dataset_version,
            target_keywords=config.target_keywords
        )
        
        # Initialize augmentation manager
        # Import the appropriate manager based on config
        if hasattr(config, 'use_improved_augmentation') and config.use_improved_augmentation:
            logger.info("Using ImprovedAugmentationManager with mixed strategies")
            from improved_augmentation_manager import ImprovedAugmentationManager
            self.augmentation_manager = ImprovedAugmentationManager(
                synthetic_dataset_path=config.synthetic_dataset_path
            )
        else:
            logger.info("Using standard AugmentationManager")
            from augmentation_manager import AugmentationManager
            self.augmentation_manager = AugmentationManager(
                fgsm_epsilon=config.fgsm_epsilon,
                synthetic_dataset_path=config.synthetic_dataset_path
            )
        
        self.model_trainer = ModelTrainer()
        
        # Storage for results
        self.results = []
        self.experiment_metadata = {}
        
        # CACHE: Pre-load datasets once
        self.dataset_cache = {}
        
        # Setup logging
        self._setup_experiment_logging()
    
    def _setup_experiment_logging(self):
        """Setup detailed logging for experiments"""
        log_file = self.save_dir / 'experiment.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.info("Experiment logging initialized")
        logger.info(f"Vocabulary - Positive: {self.config.target_keywords}")
        logger.info(f"Vocabulary - Negative: All other words")
    
    def set_random_seed(self, seed: int):
        """Set random seed for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _load_base_dataset_cached(self, dataset_size: str) -> Tuple[List[torch.Tensor], List[str]]:
        """Load base dataset with caching"""
        if dataset_size not in self.dataset_cache:
            logger.info(f"Loading {dataset_size} dataset (first time - will be cached)")
            print(f"\n🔄 Loading {dataset_size} dataset for the first time...")
            print("    (This will be cached and reused for all experiments)")
            
            audio_files, labels = self.dataset_manager.load_dataset(dataset_size)
            self.dataset_cache[dataset_size] = (audio_files, labels)
            
            logger.info(f"Cached {dataset_size} dataset: {len(audio_files)} samples")
            print(f"✓ Cached {len(audio_files)} samples for reuse\n")
        else:
            logger.info(f"Using cached {dataset_size} dataset")
            audio_files, labels = self.dataset_cache[dataset_size]
        
        # Return a copy
        return audio_files.copy(), labels.copy()
    
    def run_single_experiment(self, dataset_size: str, imbalance_ratio: float,
                            augmentation_method: str, trial: int) -> Dict:
        """Execute single experimental configuration"""
        
        experiment_id = f"{dataset_size}_{imbalance_ratio}_{augmentation_method}_trial{trial}"
        logger.info(f"Starting experiment: {experiment_id}")
        
        # Set reproducible seed
        seed = 42 + trial * 1000 + hash(f"{dataset_size}_{imbalance_ratio}_{augmentation_method}") % 1000
        self.set_random_seed(seed)
        
        try:
            # Load base dataset (with caching!)
            audio_files, labels = self._load_base_dataset_cached(dataset_size)
            logger.info(f"Using cached dataset: {len(audio_files)} samples")
            
            # Create imbalanced dataset
            audio_files, labels = self.dataset_manager.create_imbalanced_split(
                audio_files, labels, self.config.target_keywords, imbalance_ratio
            )
            
            # Split into train/test before augmentation
            train_audio, train_labels, test_audio, test_labels = self.dataset_manager.split_train_test(
                audio_files, labels, test_ratio=0.2, random_state=seed
            )
            
            # Apply augmentation to training set only
            if augmentation_method != 'none':
                # Get max_synthetic_ratio from config if available
                max_synthetic_ratio = getattr(self.config, 'max_synthetic_ratio', 0.3)
                
                # Call augmentation with proper parameters
                if hasattr(self.augmentation_manager, 'apply_augmentation'):
                    # Check if manager accepts max_synthetic_ratio
                    import inspect
                    sig = inspect.signature(self.augmentation_manager.apply_augmentation)
                    
                    if 'max_synthetic_ratio' in sig.parameters:
                        train_audio, train_labels = self.augmentation_manager.apply_augmentation(
                            train_audio, train_labels, augmentation_method,
                            self.config.target_keywords, imbalance_ratio,
                            random_state=seed,
                            max_synthetic_ratio=max_synthetic_ratio
                        )
                    else:
                        # Old interface
                        train_audio, train_labels = self.augmentation_manager.apply_augmentation(
                            train_audio, train_labels, augmentation_method,
                            self.config.target_keywords, imbalance_ratio,
                            random_state=seed
                        )
                    
                    # Validate augmented dataset
                    if hasattr(self.augmentation_manager, 'validate_augmented_dataset'):
                        if not self.augmentation_manager.validate_augmented_dataset(train_audio, train_labels):
                            raise RuntimeError("Augmented dataset validation failed")
            
            # Train and evaluate model
            metrics, trained_model = self.model_trainer.full_training_pipeline(
                train_audio, train_labels, test_audio, test_labels, self.config
            )
            
            # Compile results
            result = {
                'experiment_id': experiment_id,
                'dataset_size': dataset_size,
                'imbalance_ratio': imbalance_ratio,
                'augmentation_method': augmentation_method,
                'trial': trial,
                'seed': seed,
                'n_train_samples': len(train_audio),
                'n_test_samples': len(test_audio),
                **metrics
            }
            
            logger.info(f"Experiment {experiment_id} completed. F1 (keyword): {metrics['f1_keyword']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {str(e)}")
            
            return {
                'experiment_id': experiment_id,
                'dataset_size': dataset_size,
                'imbalance_ratio': imbalance_ratio,
                'augmentation_method': augmentation_method,
                'trial': trial,
                'seed': seed,
                'status': 'failed',
                'error': str(e),
                **{metric: 0.0 for metric in [
                    'accuracy', 'balanced_accuracy', 'f1_keyword', 'f1_non_keyword', 
                    'precision_keyword', 'recall_keyword', 'auc_roc'
                ]}
            }
    
    def calculate_total_experiments(self) -> int:
        """Calculate total number of experiments"""
        return (len(self.config.dataset_sizes) * 
                len(self.config.imbalance_ratios) * 
                len(self.config.augmentation_methods) * 
                self.config.n_trials)
    
    def run_experimental_grid(self):
        """
        Execute complete experimental grid with FIXED adversarial augmentation.
        
        Key fix: Train baseline models on BALANCED data for adversarial generation.
        """
        
        total_experiments = self.calculate_total_experiments()
        logger.info(f"Starting experimental grid: {total_experiments} total experiments")
        
        # Log experimental setup
        self.experiment_metadata = {
            'config': self.config.__dict__,
            'total_experiments': total_experiments,
            'completed_experiments': 0,
            'failed_experiments': 0
        }
        
        experiment_count = 0
        
        # PRE-LOAD: Load all dataset sizes once
        print("\n" + "="*60)
        print("📦 PRE-LOADING DATASETS (one-time, will be cached)")
        print("="*60)
        for dataset_size in self.config.dataset_sizes:
            self._load_base_dataset_cached(dataset_size)
        print("="*60)
        print("✓ All datasets cached and ready!")
        print("="*60 + "\n")
        
        # Main experimental loop
        for dataset_size in self.config.dataset_sizes:
            for imbalance_ratio in self.config.imbalance_ratios:
                
                # =====================================================================
                # CRITICAL FIX: Train BALANCED baseline for adversarial generation
                # =====================================================================
                baseline_model = None
                needs_adversarial = (
                    'adversarial' in self.config.augmentation_methods or 
                    'combined' in self.config.augmentation_methods
                )
                
                if needs_adversarial:
                    print(f"\n{'='*60}")
                    print(f"🔧 Training BALANCED baseline for adversarial generation")
                    print(f"   Dataset: {dataset_size}, Imbalance ratio: {imbalance_ratio}")
                    print(f"{'='*60}")
                    
                    baseline_model = train_balanced_baseline_for_adversarial(
                        self.dataset_manager,
                        self.config,
                        dataset_size
                    )
                    
                    if baseline_model:
                        # Set the baseline model in augmentation manager
                        if hasattr(self.augmentation_manager, 'set_adversarial_model'):
                            self.augmentation_manager.set_adversarial_model(baseline_model)
                            logger.info("✓ Baseline model set for adversarial generation")
                        elif hasattr(self.augmentation_manager, 'adversarial_augmenter'):
                            # For improved augmentation manager
                            from fixed_adversarial_augmenter import FixedAdversarialAugmenter
                            
                            # Replace with fixed version
                            self.augmentation_manager.adversarial_augmenter = FixedAdversarialAugmenter(
                                epsilon=0.005  # REDUCED from 0.01
                            )
                            self.augmentation_manager.adversarial_augmenter.set_target_model(baseline_model)
                            logger.info("✓ Fixed adversarial augmenter configured")
                        
                        print("✓ Baseline model ready for adversarial generation\n")
                    else:
                        print("⚠️  Failed to train baseline model")
                        print("   Adversarial experiments will be skipped\n")
                
                # Run experiments for all augmentation methods
                for aug_method in self.config.augmentation_methods:
                    # Skip adversarial if baseline failed
                    if aug_method in ['adversarial', 'combined'] and baseline_model is None:
                        logger.warning(f"Skipping {aug_method} (no baseline model)")
                        continue
                    
                    for trial in range(self.config.n_trials):
                        
                        result = self.run_single_experiment(
                            dataset_size, imbalance_ratio, aug_method, trial
                        )
                        
                        self.results.append(result)
                        experiment_count += 1
                        
                        # Update metadata
                        if result.get('status') == 'failed':
                            self.experiment_metadata['failed_experiments'] += 1
                        else:
                            self.experiment_metadata['completed_experiments'] += 1
                        
                        # Progress logging
                        progress = (experiment_count / total_experiments) * 100
                        logger.info(f"Progress: {experiment_count}/{total_experiments} ({progress:.1f}%)")
                        
                        print(f"\n{'='*60}")
                        print(f"✓ Completed: {aug_method} | ratio={imbalance_ratio} | trial={trial}")
                        print(f"   F1 (keyword): {result.get('f1_keyword', 0):.3f}")
                        print(f"   Progress: {experiment_count}/{total_experiments} ({progress:.1f}%)")
                        print(f"{'='*60}\n")
                        
                        # Save intermediate results
                        if self.config.save_intermediate and experiment_count % 10 == 0:
                            self._save_intermediate_results(experiment_count)
        
        # Save final results
        self._save_final_results()
        logger.info("Experimental grid completed!")
    
    def _save_intermediate_results(self, experiment_count: int):
        """Save intermediate results during long experiments"""
        if self.results:
            results_df = pd.DataFrame(self.results)
            intermediate_file = self.save_dir / f'intermediate_results_{experiment_count}.csv'
            results_df.to_csv(intermediate_file, index=False)
            logger.info(f"Saved intermediate results after {experiment_count} experiments")
    
    def _save_final_results(self):
        """Save final experimental results and metadata"""
        
        # Save results DataFrame
        results_df = pd.DataFrame(self.results)
        results_file = self.save_dir / 'experiment_results.csv'
        results_df.to_csv(results_file, index=False)
        
        # Save experiment metadata
        metadata_file = self.save_dir / 'experiment_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.experiment_metadata, f, indent=2)
        
        # Save configuration
        config_file = self.save_dir / 'experiment_config.json'
        with open(config_file, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Generate summary statistics
        self._generate_summary_statistics(results_df)
        
        logger.info(f"Final results saved to {self.save_dir}")
    
    def _generate_summary_statistics(self, results_df: pd.DataFrame):
        """Generate and save summary statistics"""
        
        # Check if 'status' column exists
        if 'status' in results_df.columns:
            successful_count = len(results_df[results_df['status'] != 'failed'])
            failed_count = len(results_df[results_df['status'] == 'failed'])
        else:
            successful_count = len(results_df)
            failed_count = 0
        
        summary_stats = {
            'total_experiments': len(results_df),
            'successful_experiments': successful_count,
            'failed_experiments': failed_count,
        }
        
        # Performance statistics by method
        if 'f1_keyword' in results_df.columns:
            method_performance = results_df.groupby('augmentation_method')['f1_keyword'].agg([
                'mean', 'std', 'min', 'max', 'count'
            ]).round(4)
            
            summary_stats['method_performance'] = method_performance.to_dict()
        
        # Save summary
        summary_file = self.save_dir / 'experiment_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        logger.info("Generated experiment summary statistics")


def run_experiment_from_config(config: ExperimentConfig):
    """Convenience function to run experiment from configuration"""
    runner = ExperimentRunner(config)
    runner.run_experimental_grid()
    return runner.save_dir
