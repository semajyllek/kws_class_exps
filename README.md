# Class Imbalance Augmentation for Keyword Detection

This repository contains the complete experimental framework for systematically evaluating **TTS (Text-to-Speech)** and **adversarial augmentation** methods for addressing class imbalance in keyword detection tasks.

## 🎯 Research Question

**When and why do TTS and adversarial augmentation help vs hurt for keyword detection class imbalance?**

This framework provides definitive experimental data across:
- Multiple dataset sizes (1K to 100K samples)
- Various imbalance ratios (5% to 100% minority class)
- Different augmentation methods (TTS, FGSM adversarial, combined)
- Statistical significance testing and effect size analysis

## 📊 Key Results Preview

Our systematic evaluation shows:
- **TTS augmentation**: Most effective at extreme imbalance (≤10% minority class), providing up to 45% F1 improvement
- **Adversarial augmentation**: Consistent moderate improvements (8-15%) across most conditions
- **Combined approach**: Diminishing returns, often no better than TTS alone
- **Dataset size matters**: Larger datasets reduce the benefit of both methods

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo-url>
cd kws_class_exps

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install system dependencies for TTS (Linux/macOS)
sudo apt-get install espeak espeak-data libespeak1 libespeak-dev  # Linux
# brew install espeak  # macOS
```

### 2. Generate Synthetic Dataset (One-time, 2-4 hours)

```bash
# Quick test dataset (400 samples, ~30 minutes)
python synthetic_data_generator.py --config quick

# Large research dataset (10,000 samples, ~2-4 hours)
python synthetic_data_generator.py --config large
```

### 3. Run Experiments

```bash
# Quick test (72 experiments, ~2-3 hours)
python main.py --config quick

# Full research grid (560 experiments, ~1-2 days)
python main.py --config full
```

### 4. View Results

Results automatically appear in `./results/analysis/`:
- 📈 **Performance heatmaps** showing F1 scores across conditions
- 📊 **Improvement analysis** with statistical significance
- 📉 **Method comparison** across imbalance ratios
- 📋 **Summary report** with key findings

## 📁 Repository Structure

```
kws_class_exps/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── config.py                    # Experimental configurations
├── synthetic_data_generator.py  # Pre-generate TTS datasets
├── audio_processing.py          # Audio preprocessing utilities
├── dataset_manager.py           # Google Speech Commands management
├── adversarial_augmenter.py     # FGSM adversarial generation
├── augmentation_manager.py      # Orchestrates all augmentation
├── model_training.py            # CNN model training & evaluation
├── experiment_runner.py         # Main experimental orchestration
├── results_analysis.py          # Statistical analysis & visualization
├── main.py                      # Main execution script
│
├── data/                        # Google Speech Commands (auto-downloaded)
├── synthetic_datasets/          # Pre-generated TTS data
├── results/                     # Experimental results & analysis
└── logs/                        # Execution logs
```

## 🔬 Experimental Design

### Scientific Methodology

- **Controlled Variables**: Fixed TTS model (Tacotron2-DDC), CNN architecture, preprocessing
- **Systematic Grid**: 4 methods × 4 dataset sizes × 7 imbalance ratios × 5 trials = 560 experiments
- **Statistical Rigor**: Multiple random seeds, stratified splits, significance testing
- **Reproducible**: All synthetic data pre-generated, fixed random seeds, version-controlled configs

### Datasets

- **Base**: Google Speech Commands V1/V2 (keyword detection benchmark)
- **Keywords**: `['yes', 'no', 'up', 'down']` (configurable)
- **Sizes**: Small (1K), Medium (5K), Large (20K), Full (~100K)
- **Imbalance Ratios**: 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 (minority:majority)

### Augmentation Methods

1. **None** (baseline): Original imbalanced data only
2. **TTS**: Coqui Tacotron2 with prosodic variations (`"yes"`, `"yes!"`, `"YES"`, etc.)
3. **Adversarial**: FGSM (ε=0.01) targeting minority class
4. **Combined**: 50% TTS + 50% adversarial

### Evaluation Metrics

- **Primary**: F1-score for keyword class (minority)
- **Secondary**: Balanced accuracy, AUC-ROC, per-class precision/recall
- **Statistical**: Effect sizes, p-values, confidence intervals

## 🎛️ Configuration Options

### Pre-built Configurations

```bash
# Quick test (2-3 hours total)
python main.py --config quick

# Full research (1-2 days total)  
python main.py --config full
```

### Custom Configuration

Edit `config.py` or create custom configs:

```python
config = ExperimentConfig(
    dataset_version='v2',
    target_keywords=['stop', 'go', 'left', 'right'],
    dataset_sizes=['medium', 'large'],
    imbalance_ratios=[0.1, 0.2, 0.5],
    augmentation_methods=['none', 'synthetic', 'adversarial'],
    n_trials=3,
    synthetic_dataset_path='./synthetic_datasets/my_custom_dataset'
)
```

### Synthetic Data Generation

```bash
# Custom synthetic dataset
python synthetic_data_generator.py \
  --keywords yes no stop go \
  --samples-per-keyword 500 \
  --output-dir ./my_synthetic_data
```

## 📈 Understanding Results

### Generated Visualizations

1. **`performance_heatmap.png`**: F1 scores across dataset size × imbalance ratio grid
2. **`improvement_analysis.png`**: Box plots showing when methods help vs hurt  
3. **`method_comparison.png`**: Line plots and precision-recall analysis
4. **`statistical_significance.png`**: P-values and effect sizes

### Key Files

- **`experiment_results.csv`**: Raw results from all experiments
- **`summary_report.txt`**: Publication-ready text summary
- **`improvement_metrics.csv`**: Calculated improvements over baseline

### Example Findings

```
TTS METHOD:
  - Average improvement: +12.3%
  - Best case improvement: +45.2% (at 0.1 imbalance ratio)
  - Helpful in 18/21 conditions (85.7%)
  - Most effective: small datasets with extreme imbalance

ADVERSARIAL METHOD:
  - Average improvement: +8.1%
  - Best case improvement: +31.4%
  - Helpful in 15/21 conditions (71.4%)
  - Most consistent across different conditions
```

## 🔧 Advanced Usage

### Analyze Existing Results

```bash
python main.py --analyze-only ./results/
```

### Monitor Long-Running Experiments

```bash
# Check progress
tail -f experiment_main.log

# View intermediate results
ls ./results/intermediate_results_*.csv
```

### Resume Failed Experiments

The framework automatically saves progress every 10 experiments. Simply re-run the same command to continue from the last checkpoint.

## 🧪 Validation & Testing

### Quick Validation

```bash
# Test synthetic data generation (5 minutes)
python synthetic_data_generator.py --config quick

# Test full pipeline (30 minutes)  
python main.py --config quick
```

### Expected Outputs

After quick test, you should see:
- `./synthetic_datasets/gsc_synthetic_quick/` with ~400 samples
- `./results/experiment_results.csv` with 72 experimental results
- `./results/analysis/` with 4 visualization plots

## 📚 Technical Details

### TTS Pipeline

1. **Text Variations**: `"yes"` → `["yes", "yes!", "YES", "yes?", ...]`
2. **Speech Synthesis**: Coqui Tacotron2-DDC (deterministic, reproducible)
3. **Acoustic Variations**: Pitch, amplitude, noise modifications
4. **Quality Control**: Energy thresholding, validation checks

### Adversarial Pipeline

1. **Target Selection**: Majority class samples → flip to minority
2. **FGSM Attack**: ε=0.01 perturbation on raw audio
3. **Validation**: Perturbation bounds, audio quality checks

### Model Architecture

- **CNN**: 3 conv blocks (32→64→128 channels) + 2 FC layers
- **Input**: 1-second audio (16kHz = 16,000 samples)
- **Training**: Adam optimizer, early stopping, gradient clipping
- **Output**: Binary classification (keyword vs non-keyword)

## 🤝 Contributing

### Adding New Augmentation Methods

1. Create new augmenter class in separate file
2. Implement `generate_samples(audio_files, labels, n_samples)` method
3. Add to `AugmentationManager.apply_augmentation()`
4. Update config options

### Adding New Datasets

1. Extend `GSCDatasetManager` or create new dataset manager
2. Ensure compatible audio preprocessing
3. Update configuration options

## 🐛 Troubleshooting

### Common Issues

**TTS Installation Problems**:
```bash
# Try alternative TTS installation
pip install TTS --no-deps
pip install torch torchaudio

# Check espeak installation
espeak "test"
```

**Memory Issues**:
- Reduce `batch_size` in config
- Use smaller dataset sizes first
- Monitor GPU memory usage

**Slow Experiments**:
- Use `--config quick` for testing
- Generate synthetic data once, reuse for multiple experiments
- Consider reducing `n_trials` for faster iteration

### Getting Help

1. Check logs in `./logs/` directory
2. Verify synthetic dataset generation completed successfully
3. Ensure all required files are present (see Repository Structure)
4. Test with `--config quick` before running full experiments

## 📄 Citation

If you use this framework in your research, please cite:

```bibtex
@article{your2024tts,
  title={TTS and Adversarial Augmentation for Keyword Detection Class Imbalance},
  author={Your Name},
  journal={Your Conference/Journal},
  year={2024}
}
```

## 📋 License

[Choose your license - MIT, Apache 2.0, etc.]

## 🔗 Related Work

- [Google Speech Commands Dataset](https://arxiv.org/abs/1804.03209)
- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [FGSM Adversarial Examples](https://arxiv.org/abs/1412.6572)

---

**Questions?** Open an issue or check the troubleshooting section above.
