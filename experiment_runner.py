"""
Main experiment runner for systematic class imbalance augmentation study.
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
from augmentation_manager import AugmentationManager
from model_training import ModelTrainer

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Orchestrates systematic experimental evaluation"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # ensure dataset root directory exists
        Path(config.dataset_root).mkdir(parents=True, exist_ok=True)        
  
        # Initialize components with vocabulary control
        self.dataset_manager = GSCDatasetManager(
            root_dir=config.dataset_root,
            version=config.dataset_version,
            target_keywords=config.target_keywords  # Only need positive keywords
        )
        
        self.augmentation_manager = AugmentationManager(
            fgsm_epsilon=config.fgsm_epsilon,
            synthetic_dataset_path=config.synthetic_dataset_path
        )
        
        self.model_trainer = ModelTrainer()
        
        # Storage for results
        self.results = []
        self.experiment_metadata = {}
        
        # CACHE: Pre-load datasets once to avoid reloading for every experiment
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
        logger.info(f"Vocabulary - Negative: All other words (realistic keyword spotting)")
    
    def set_random_seed(self, seed: int):
        """Set random seed for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _load_base_dataset_cached(self, dataset_size: str) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Load base dataset with caching to avoid reloading for every experiment.
        
        The dataset is loaded once per size and reused across all experiments.
        """
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
        
        # Return a copy to avoid modifying the cached version
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
                train_audio, train_labels = self.augmentation_manager.apply_augmentation(
                    train_audio, train_labels, augmentation_method,
                    self.config.target_keywords, imbalance_ratio
                )
                
                # Validate augmented dataset
                if not self.augmentation_manager.validate_augmented_dataset(train_audio, train_labels):
                    raise RuntimeError("Augmented dataset validation failed")
            
            # Train and evaluate model
            metrics, trained_model = self.model_trainer.full_training_pipeline(
                train_audio, train_labels, test_audio, test_labels, self.config
            )
            
            # Note: We don't update the adversarial model here anymore
            # It's set once per (dataset_size, imbalance_ratio) in run_experimental_grid()
            
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
            
            logger.info(f"Experiment {experiment_id} completed successfully. "
                       f"F1 (keyword): {metrics['f1_keyword']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {str(e)}")
            
            # Return failed experiment record
            return {
                'experiment_id': experiment_id,
                'dataset_size': dataset_size,
                'imbalance_ratio': imbalance_ratio,
                'augmentation_method': augmentation_method,
                'trial': trial,
                'seed': seed,
                'status': 'failed',
                'error': str(e),
                # Zero out metrics for failed experiments
                **{metric: 0.0 for metric in [
                    'accuracy', 'balanced_accuracy', 'f1_keyword', 'f1_non_keyword', 
                    'precision_keyword', 'recall_keyword', 'auc_roc'
                ]}
            }
    
    def calculate_total_experiments(self) -> int:
        """Calculate total number of experiments to run"""
        return (len(self.config.dataset_sizes) * 
                len(self.config.imbalance_ratios) * 
                len(self.config.augmentation_methods) * 
                self.config.n_trials)
    
    def run_experimental_grid(self):
        """Execute complete experimental grid"""
        
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
        
        # PRE-LOAD: Load all dataset sizes once at the beginning
        print("\n" + "="*60)
        print("📦 PRE-LOADING DATASETS (one-time, will be cached)")
        print("="*60)
        for dataset_size in self.config.dataset_sizes:
            self._load_base_dataset_cached(dataset_size)
        print("="*60)
        print("✓ All datasets cached and ready!")
        print("="*60 + "\n")
        
        # Nested loops for systematic grid search
        for dataset_size in self.config.dataset_sizes:
            for imbalance_ratio in self.config.imbalance_ratios:
                
                # TRAIN BASELINE MODEL ONCE per (dataset_size, imbalance_ratio)
                # This model is used for adversarial augmentation
                baseline_model = None
                if 'adversarial' in self.config.augmentation_methods or 'combined' in self.config.augmentation_methods:
                    logger.info(f"Training baseline model for adversarial generation ({dataset_size}, ratio={imbalance_ratio})")
                    baseline_model = self._train_baseline_for_adversarial(dataset_size, imbalance_ratio)
                    if baseline_model:
                        self.augmentation_manager.set_adversarial_model(baseline_model)
                        logger.info("Baseline model set for adversarial generation")
                
                for aug_method in self.config.augmentation_methods:
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
                        logger.info(f"Progress: {experiment_count}/{total_experiments} "
                                   f"({progress:.1f}%)")
                        
                        # Save intermediate results
                        if self.config.save_intermediate and experiment_count % 10 == 0:
                            self._save_intermediate_results(experiment_count)
        
        # Save final results
        self._save_final_results()
        logger.info("Experimental grid completed!")
    
    def _train_baseline_for_adversarial(self, dataset_size: str, imbalance_ratio: float):
        """
        Train a baseline model for adversarial generation.
        
        This model is trained once per (dataset_size, imbalance_ratio) combination
        and used to generate adversarial examples for all trials.
        """
        try:
            print(f"\n🔧 Training baseline model for adversarial generation...")
            print(f"   Dataset: {dataset_size}, Ratio: {imbalance_ratio}")
            
            # Use fixed seed for reproducibility
            seed = 42
            self.set_random_seed(seed)
            
            # Load and prepare data
            audio_files, labels = self._load_base_dataset_cached(dataset_size)
            audio_files, labels = self.dataset_manager.create_imbalanced_split(
                audio_files, labels, self.config.target_keywords, imbalance_ratio
            )
            train_audio, train_labels, _, _ = self.dataset_manager.split_train_test(
                audio_files, labels, test_ratio=0.2, random_state=seed
            )
            
            # Train model (no augmentation for baseline)
            print("   Training baseline model (this may take a few minutes)...")
            _, baseline_model = self.model_trainer.full_training_pipeline(
                train_audio, train_labels, train_audio[:10], train_labels[:10],  # Dummy test set
                self.config
            )
            
            print("✓ Baseline model ready for adversarial generation\n")
            return baseline_model
            
        except Exception as e:
            logger.error(f"Failed to train baseline model: {e}")
            print(f"⚠️  Could not train baseline model: {e}")
            print("   Adversarial experiments will be skipped")
            return None
    
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
    
        # Check if 'status' column exists, if not, assume all experiments succeeded
        if 'status' in results_df.columns:
            successful_count = len(results_df[results_df['status'] != 'failed'])
            failed_count = len(results_df[results_df['status'] == 'failed'])
        else:
            # No status column means all experiments succeeded
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
