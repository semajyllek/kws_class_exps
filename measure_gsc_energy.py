"""
Measure GSC energy statistics and save for synthetic data generation.
"""
import torch
import numpy as np
import json
from pathlib import Path
from dataset_manager import GSCDatasetManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def measure_gsc_energy(output_file: str = './synthetic_datasets/gsc_energy_profile.json',
                      dataset_size: str = 'medium') -> dict:
    """
    Measure energy statistics of GSC dataset and save to file.
    
    Args:
        output_file: Path to save energy profile JSON
        dataset_size: Size of GSC dataset to sample ('small', 'medium', 'large')
    
    Returns:
        Dictionary with energy statistics
    """
    manager = GSCDatasetManager(root_dir='./data', version='v2')
    
    # Load a sample of the dataset
    logger.info(f"Loading GSC {dataset_size} dataset to measure energy...")
    audio_files, labels = manager.load_dataset(dataset_size)
    
    # Calculate RMS values
    logger.info("Calculating energy statistics...")
    rms_values = []
    energies = []
    
    for audio in audio_files:
        rms = torch.sqrt(torch.mean(audio ** 2)).item()
        energy = torch.sum(audio ** 2).item()
        rms_values.append(rms)
        energies.append(energy)
    
    # Compile statistics
    energy_profile = {
        'dataset_version': 'v2',
        'dataset_size_measured': dataset_size,
        'n_samples_measured': len(audio_files),
        'rms': {
            'mean': float(np.mean(rms_values)),
            'median': float(np.median(rms_values)),
            'std': float(np.std(rms_values)),
            'min': float(np.min(rms_values)),
            'max': float(np.max(rms_values)),
            'p25': float(np.percentile(rms_values, 25)),
            'p75': float(np.percentile(rms_values, 75))
        },
        'energy': {
            'mean': float(np.mean(energies)),
            'median': float(np.median(energies)),
            'std': float(np.std(energies)),
            'min': float(np.min(energies)),
            'max': float(np.max(energies)),
            'p25': float(np.percentile(energies, 25)),
            'p75': float(np.percentile(energies, 75))
        },
        'recommended_target_rms': float(np.mean(rms_values))
    }
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(energy_profile, f, indent=2)
    
    logger.info(f"Energy profile saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("GSC AUDIO ENERGY PROFILE")
    print("="*60)
    print(f"Dataset: {dataset_size} ({len(audio_files)} samples)")
    print("\nRMS Statistics:")
    print(f"  Mean RMS: {energy_profile['rms']['mean']:.6f}")
    print(f"  Median RMS: {energy_profile['rms']['median']:.6f}")
    print(f"  Std RMS: {energy_profile['rms']['std']:.6f}")
    print(f"  Range: [{energy_profile['rms']['min']:.6f}, {energy_profile['rms']['max']:.6f}]")
    print("\nEnergy Statistics:")
    print(f"  Mean Energy: {energy_profile['energy']['mean']:.2f}")
    print(f"  Median Energy: {energy_profile['energy']['median']:.2f}")
    print(f"  Std Energy: {energy_profile['energy']['std']:.2f}")
    print("\n" + "="*60)
    print(f"✓ Recommended target_rms: {energy_profile['recommended_target_rms']:.6f}")
    print("="*60)
    
    return energy_profile


def load_gsc_energy_profile(profile_file: str = './synthetic_datasets/gsc_energy_profile.json') -> dict:
    """
    Load GSC energy profile from file.
    
    Args:
        profile_file: Path to energy profile JSON
    
    Returns:
        Dictionary with energy statistics
    
    Raises:
        FileNotFoundError: If profile doesn't exist
    """
    profile_path = Path(profile_file)
    
    if not profile_path.exists():
        raise FileNotFoundError(
            f"GSC energy profile not found at {profile_path}. "
            "Run measure_gsc_energy() first to generate it."
        )
    
    with open(profile_path, 'r') as f:
        profile = json.load(f)
    
    logger.info(f"Loaded GSC energy profile: target_rms={profile['recommended_target_rms']:.6f}")
    
    return profile


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Measure GSC energy statistics')
    parser.add_argument('--dataset-size', choices=['small', 'medium', 'large', 'full'],
                       default='medium', help='GSC dataset size to measure')
    parser.add_argument('--output', type=str, 
                       default='./synthetic_datasets/gsc_energy_profile.json',
                       help='Output file for energy profile')
    
    args = parser.parse_args()
    
    measure_gsc_energy(output_file=args.output, dataset_size=args.dataset_size)
