"""
Main experiment runner for systematic class imbalance augmentation study.
"""
import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List
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
        
        # Initialize components
        self.dataset_manager = GSCDatasetManager(
            root_dir=config.dataset_root,
            version=config.dataset_version
        )
        
        self.augmentation_manager = AugmentationManager(
            fgsm_epsilon=config.fgsm_epsilon,
            synthetic_dataset_path=config.synthetic_dataset_path
        )
        
        self.model_trainer = ModelTrainer()
        
        # Storage for results
        self.results = []
        self.experiment_metadata = {}
        
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
    
    def set_random_seed(self, seed: int):
        """Set random seed for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def run_single_experiment(self, dataset_size: str, imbalance_ratio: float,
                            augmentation_method: str, trial: int) -> Dict:
        """Execute single experimental configuration"""
        
        experiment_id = f"{dataset_size}_{imbalance_ratio}_{augmentation_method}_trial{trial}"
        logger.info(f"Starting experiment: {experiment_id}")
        
        # Set reproducible seed
        seed = 42 + trial * 1000 + hash(f"{dataset_size}_{imbalance_ratio}_{augmentation_method}") % 1000
        self.set_random_seed(seed)
        
        try:
            # Load base dataset
            audio_files, labels = self.dataset_manager.load_dataset(dataset_size)
            logger.info(f"Loaded {len(audio_files)} samples")
            
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
            
            # Store model for adversarial generation in subsequent experiments
            if augmentation_method in ['adversarial', 'combined']:
                self.augmentation_manager.set_adversarial_model(trained_model)
            
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
        
        # Nested loops for systematic grid search
        for dataset_size in self.config.dataset_sizes:
            for imbalance_ratio in self.config.imbalance_ratios:
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
        
        summary_stats = {
            'total_experiments': len(results_df),
            'successful_experiments': len(results_df[results_df.get('status') != 'failed']),
            'failed_experiments': len(results_df[results_df.get('status') == 'failed']),
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
