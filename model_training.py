"""
Model training and evaluation for keyword detection experiments.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                           balanced_accuracy_score, roc_auc_score)
import logging

logger = logging.getLogger(__name__)

class KeywordDataset(Dataset):
    """PyTorch Dataset for keyword detection"""
    
    def __init__(self, audio_files: List[torch.Tensor], labels: List[str]):
        self.audio_files = audio_files
        self.label_to_idx = {'non_keyword': 0, 'keyword': 1}
        self.labels = [self.label_to_idx[label] for label in labels]
        
        assert len(self.audio_files) == len(self.labels), "Mismatch in audio and labels"
        logger.info(f"Created dataset with {len(self.audio_files)} samples")
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio = self.audio_files[idx].squeeze()
        label = self.labels[idx]
        return audio, label

class KeywordClassifier(nn.Module):
    """CNN-based keyword classifier optimized for 1-second audio"""
    
    def __init__(self, input_length: int = 16000, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(1, 32, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),
            
            # Second conv block
            nn.Conv1d(32, 64, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),
            
            # Third conv block
            nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class ModelTrainer:
    """Handles model training, evaluation, and metrics calculation"""
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initialized trainer on device: {self.device}")
    
    def full_training_pipeline(self, train_audio: List[torch.Tensor], train_labels: List[str],
                             test_audio: List[torch.Tensor], test_labels: List[str],
                             config) -> Tuple[Dict[str, float], nn.Module]:
        """Complete training and evaluation pipeline"""
        
        # Create datasets
        train_dataset = KeywordDataset(train_audio, train_labels)
        test_dataset = KeywordDataset(test_audio, test_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        # Initialize model
        model = KeywordClassifier(input_length=config.max_audio_length, n_classes=2).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Train model
        for epoch in range(config.n_epochs):
            model.train()
            total_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 5 == 0:
                avg_loss = total_loss / len(train_loader)
                logger.print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")
        
        # Evaluate model
        test_metrics = self.evaluate_model(model, test_loader)
        
        return test_metrics, model
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model and return comprehensive metrics"""
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = model(data)
                probabilities = torch.softmax(output, dim=1)
                predictions = torch.argmax(output, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        return self._calculate_metrics(all_targets, all_predictions, all_probabilities)
    
    def _calculate_metrics(self, targets: List[int], predictions: List[int], 
                          probabilities: List[float]) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        
        # Basic accuracy metrics
        accuracy = accuracy_score(targets, predictions)
        balanced_accuracy = balanced_accuracy_score(targets, predictions)
        
        # Per-class precision, recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        # Handle case where only one class present
        if len(precision) == 1:
            precision = np.pad(precision, (0, 1), 'constant')
            recall = np.pad(recall, (0, 1), 'constant')
            f1 = np.pad(f1, (0, 1), 'constant')
        
        # AUC-ROC
        try:
            auc_roc = roc_auc_score(targets, probabilities)
        except ValueError:
            auc_roc = 0.5
            logger.warning("Only one class in test set, setting AUC-ROC to 0.5")
        
        metrics = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'precision_non_keyword': precision[0],
            'recall_non_keyword': recall[0],
            'f1_non_keyword': f1[0],
            'precision_keyword': precision[1] if len(precision) > 1 else 0.0,
            'recall_keyword': recall[1] if len(recall) > 1 else 0.0,
            'f1_keyword': f1[1] if len(f1) > 1 else 0.0,
            'auc_roc': auc_roc,
            'n_test_samples': len(targets),
            'n_positive_test': sum(targets),
            'n_negative_test': len(targets) - sum(targets)
        }
        
        return metrics
