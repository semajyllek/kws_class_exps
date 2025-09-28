"""
Main execution script for class imbalance augmentation experiments.
"""
import argparse
import logging
import sys
from pathlib import Path

from config import ExperimentConfig, create_quick_config, create_full_config
from experiment_runner import ExperimentRunner
from results_analysis import ExperimentAnalyzer

def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration"""
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('experiment_main.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized")
    return logger

def run_experiments(config: ExperimentConfig, logger):
    """Run the experimental grid"""
    
    logger.info("Starting experimental evaluation...")
    logger.info(f"Configuration: {config}")
    
    # Estimate runtime
    total_experiments = (len(config.dataset_sizes) * len(config.imbalance_ratios) * 
                        len(config.augmentation_methods) * config.n_trials)
    estimated_hours = total_experiments * 0.1
    
    logger.info(f"Estimated runtime: {estimated_hours:.1f} hours for {total_experiments} experiments")
    
    # Run experiments
    runner = ExperimentRunner(config)
    runner.run_experimental_grid()
    
    logger.info(f"Experiments completed. Results saved to: {runner.save_dir}")
    return runner.save_dir

def analyze_results(results_dir: Path, logger):
    """Analyze experimental results"""
    
    results_file = results_dir / 'experiment_results.csv'
    
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        return
    
    logger.info("Starting results analysis...")
    
    # Create analyzer
    analyzer = ExperimentAnalyzer(str(results_file))
    
    # Generate all analyses
    analysis_dir = analyzer.save_all_analyses(results_dir / 'analysis')
    
    logger.info(f"Analysis completed. Plots and summaries saved to: {analysis_dir}")
    
    # Print key findings
    summary = analyzer.generate_publication_summary()
    print("\n" + "="*60)
    print("KEY EXPERIMENTAL FINDINGS")
    print("="*60)
    print(summary)
    print("="*60)

def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description='Run class imbalance augmentation experiments')
    parser.add_argument('--config', choices=['quick', 'full'],
                       default='quick', help='Experiment configuration type')
    parser.add_argument('--analyze-only', type=str, metavar='RESULTS_DIR',
                       help='Only analyze existing results from specified directory')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    try:
        # Analysis-only mode
        if args.analyze_only:
            results_dir = Path(args.analyze_only)
            if not results_dir.exists():
                logger.error(f"Results directory not found: {results_dir}")
                return 1
            
            analyze_results(results_dir, logger)
            return 0
        
        # Create configuration
        logger.info(f"Creating {args.config} configuration...")
        
        if args.config == 'quick':
            config = create_quick_config()
        elif args.config == 'full':
            config = create_full_config()
        else:
            raise ValueError(f"Unknown config type: {args.config}")
        
        # Run experiments
        results_dir = run_experiments(config, logger)
        
        if results_dir is None:
            logger.info("Experiment cancelled")
            return 1
        
        # Analyze results
        analyze_results(results_dir, logger)
        
        logger.info("Complete experimental pipeline finished successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
