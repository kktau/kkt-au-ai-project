#!/usr/bin/env python3
import torch
from torch import nn
from torchvision import models
from typing import Dict, Tuple, Optional, Union
import logging
from pathlib import Path
from collections import OrderedDict

class ModelArchitecture:
    """Class to handle model architecture configurations"""
    
    # Available architectures and their configurations
    ARCHITECTURES = {
        'vgg16': {
            'model': models.vgg16,
            'features': 25088,
            'classifier_attr': 'classifier'
        },
        'resnet18': {
            'model': models.resnet18,
            'features': 512,
            'classifier_attr': 'fc'
        },
        'densenet121': {
            'model': models.densenet121,
            'features': 1024,
            'classifier_attr': 'classifier'
        }
    }

class FlowerClassifier:
    """Main class for flower classification model operations"""
    
    def __init__(self, arch: str = 'vgg16', hidden_units: int = 4096,
                 dropout: float = 0.5, num_classes: int = 102):
        """
        Initialize the classifier
        
        Args:
            arch: Model architecture
            hidden_units: Number of hidden units
            dropout: Dropout rate
            num_classes: Number of output classes
        """
        if arch not in ModelArchitecture.ARCHITECTURES:
            raise ValueError(f"Unsupported architecture. Choose from: {list(ModelArchitecture.ARCHITECTURES.keys())}")
            
        self.arch = arch
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.num_classes = num_classes
        self.logger = self._setup_logger()
        
        # Build model
        self.model = self._build_model()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _build_model(self) -> nn.Module:
        """Build and configure the model"""
        # Get architecture configuration
        arch_config = ModelArchitecture.ARCHITECTURES[self.arch]
        
        # Load pretrained model
        model = arch_config['model'](pretrained=True)
        
        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Build classifier
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(arch_config['features'], self.hidden_units)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(self.dropout)),
            ('fc2', nn.Linear(self.hidden_units, self.num_classes)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        
        # Set classifier
        setattr(model, arch_config['classifier_attr'], classifier)
        
        self.logger.info(f"Built {self.arch} model with {self.hidden_units} hidden units")
        return model
    
    def save_checkpoint(self, save_path: Union[str, Path], class_to_idx: Dict,
                       optimizer: torch.optim.Optimizer, epochs: int,
                       accuracy: float) -> None:
        """
        Save model checkpoint
        
        Args:
            save_path: Path to save checkpoint
            class_to_idx: Class to index mapping
            optimizer: Optimizer state
            epochs: Number of training epochs
            accuracy: Model accuracy
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get classifier attribute name
        classifier_attr = ModelArchitecture.ARCHITECTURES[self.arch]['classifier_attr']
        
        checkpoint = {
            'arch': self.arch,
            'hidden_units': self.hidden_units,
            'state_dict': self.model.state_dict(),
            'class_to_idx': class_to_idx,
            'classifier': getattr(self.model, classifier_attr),
            'optimizer_state': optimizer.state_dict(),
            'epochs': epochs,
            'accuracy': accuracy
        }
        
        torch.save(checkpoint, save_path)
        self.logger.info(f"Checkpoint saved to {save_path}")
    
    @classmethod
    def load_checkpoint(cls, filepath: Union[str, Path], device: Optional[torch.device] = None) -> 'FlowerClassifier':
        """
        Load model from checkpoint
        
        Args:
            filepath: Path to checkpoint file
            device: Device to load model to
            
        Returns:
            FlowerClassifier: Loaded model instance
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=device)
        
        # Create model instance
        model = cls(
            arch=checkpoint['arch'],
            hidden_units=checkpoint['hidden_units']
        )
        
        # Load state dict
        model.model.load_state_dict(checkpoint['state_dict'])
        
        # Set class to idx mapping
        model.model.class_to_idx = checkpoint['class_to_idx']
        
        # Move to device if specified
        if device:
            model.model.to(device)
        
        model.logger.info(f"Model loaded from {filepath}")
        return model
    
    def to(self, device: torch.device) -> None:
        """Move model to specified device"""
        self.model.to(device)
    
    def train(self) -> None:
        """Set model to training mode"""
        self.model.train()
    
    def eval(self) -> None:
        """Set model to evaluation mode"""
        self.model.eval()
    
    def parameters(self):
        """Get model parameters"""
        return self.model.parameters()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.model(x)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Make model callable"""
        return self.forward(x)

# Example usage
if __name__ == '__main__':
    try:
        # Create model
        model = FlowerClassifier(arch='vgg16', hidden_units=4096)
        
        # Create dummy data
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Test forward pass
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        
        # Save checkpoint
        optimizer = torch.optim.Adam(model.parameters())
        model.save_checkpoint(
            'checkpoints/test.pth',
            class_to_idx={'1': 0},
            optimizer=optimizer,
            epochs=1,
            accuracy=0.85
        )
        
        # Load checkpoint
        loaded_model = FlowerClassifier.load_checkpoint('checkpoints/test.pth')
        print("Model loaded successfully!")
        
    except Exception as e:
        logging.error(f"Error in testing: {str(e)}")
