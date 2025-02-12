#!/usr/bin/env python3
import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, Tuple, List, Optional, Union

class DataLoader:
    """Class for handling data loading operations"""
    
    @staticmethod
    def get_transforms(train: bool = False) -> transforms.Compose:
        """
        Get data transformations for training or validation/testing
        
        Args:
            train: Whether to include training augmentations
            
        Returns:
            transforms.Compose: Composed transformations
        """
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    @classmethod
    def load_data(cls, data_dir: str, batch_size: int = 64) -> Tuple[Dict, Dict]:
        """
        Load and transform data for training and validation
        
        Args:
            data_dir: Root directory containing the dataset
            batch_size: Batch size for dataloaders
            
        Returns:
            tuple: (dataloaders, class_to_idx)
        """
        data_dir = Path(data_dir)
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {data_dir}")
        
        train_dir = data_dir / 'train'
        valid_dir = data_dir / 'valid'
        test_dir = data_dir / 'test'
        
        if not all(d.exists() for d in [train_dir, valid_dir, test_dir]):
            raise FileNotFoundError("Missing required train, valid, or test directory")
        
        # Load datasets with transforms
        image_datasets = {
            'train': datasets.ImageFolder(train_dir, transform=cls.get_transforms(train=True)),
            'valid': datasets.ImageFolder(valid_dir, transform=cls.get_transforms(train=False)),
            'test': datasets.ImageFolder(test_dir, transform=cls.get_transforms(train=False))
        }
        
        # Create dataloaders
        dataloaders = {
            x: torch.utils.data.DataLoader(
                image_datasets[x], 
                batch_size=batch_size,
                shuffle=True if x == 'train' else False,
                num_workers=4 if torch.cuda.is_available() else 0
            )
            for x in ['train', 'valid', 'test']
        }
        
        return dataloaders, image_datasets['train'].class_to_idx

class ImageProcessor:
    """Class for handling image processing operations"""
    
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def process_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Process an image for use in a PyTorch model
        
        Args:
            image_path: Path to image file
            
        Returns:
            torch.Tensor: Processed image tensor
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            return self.transforms(img)
            
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")
    
    @staticmethod
    def imshow(image: torch.Tensor, ax: Optional[plt.Axes] = None, 
               title: Optional[str] = None) -> plt.Axes:
        """
        Display a PyTorch tensor as an image
        
        Args:
            image: Image tensor
            ax: Optional matplotlib axes
            title: Optional title for the plot
            
        Returns:
            plt.Axes: Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        # Convert tensor to numpy array
        image = image.numpy().transpose((1, 2, 0))
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        
        # Display
        ax.imshow(image)
        if title:
            ax.set_title(title)
        ax.axis('off')
        
        return ax

class CategoryManager:
    """Class for handling category mappings"""
    
    @staticmethod
    def load_category_names(filepath: Union[str, Path]) -> Dict[str, str]:
        """
        Load category names from JSON file
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            dict: Mapping of category indices to names
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Category mapping file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Error loading category names: {str(e)}")

class Visualizer:
    """Class for visualization utilities"""
    
    @staticmethod
    def plot_predictions(image_path: Union[str, Path], probs: np.ndarray, 
                        classes: List[str], cat_to_name: Optional[Dict] = None) -> None:
        """
        Plot image and its predictions
        
        Args:
            image_path: Path to image file
            probs: Prediction probabilities
            classes: Predicted class indices
            cat_to_name: Optional mapping of indices to names
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(figsize=(12,4), nrows=1, ncols=2)
        
        # Display image
        img = Image.open(image_path)
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title('Input Image')
        
        # Convert class indices to names if mapping provided
        if cat_to_name:
            labels = [cat_to_name[cls] for cls in classes]
        else:
            labels = classes
        
        # Plot probabilities
        y_pos = np.arange(len(probs))
        ax2.barh(y_pos, probs)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels)
        ax2.invert_yaxis()
        ax2.set_xlabel('Probability')
        ax2.set_title('Top Predictions')
        
        plt.tight_layout()
        plt.show()

def setup_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Setup a logger with specified configuration
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# Example usage
if __name__ == '__main__':
    logger = setup_logger()
    
    try:
        # Test data loading
        dataloaders, class_to_idx = DataLoader.load_data('flowers')
        logger.info("Data loaded successfully")
        
        # Test image processing
        processor = ImageProcessor()
        img_tensor = processor.process_image('flowers/test/1/image_06743.jpg')
        logger.info("Image processed successfully")
        
        # Test category loading
        categories = CategoryManager.load_category_names('cat_to_name.json')
        logger.info("Categories loaded successfully")
        
    except Exception as e:
        logger.error(f"Error in testing: {str(e)}")
