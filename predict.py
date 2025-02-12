#!/usr/bin/env python3
import argparse
import torch
from torchvision import transforms, models
from PIL import Image
import json
import logging
import os
from collections import OrderedDict

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_input_args():
    """Get command line arguments"""
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    
    # Required arguments
    parser.add_argument('image_path', type=str, help='Path to image')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    
    # Optional arguments
    parser.add_argument('--top_k', type=int, default=5,
                        help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='Path to category names mapping')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for inference')
    
    return parser.parse_args()

def load_checkpoint(filepath, device):
    """
    Load a trained model checkpoint
    
    Args:
        filepath: Path to checkpoint file
        device: Device to load the model to
        
    Returns:
        model: Loaded PyTorch model
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    try:
        checkpoint = torch.load(filepath, map_location=device)
        
        # Load the appropriate model architecture
        if checkpoint['arch'] == 'vgg16':
            model = models.vgg16(pretrained=True)
            model.classifier = checkpoint['classifier']
        elif checkpoint['arch'] == 'resnet18':
            model = models.resnet18(pretrained=True)
            model.fc = checkpoint['classifier']
        elif checkpoint['arch'] == 'densenet121':
            model = models.densenet121(pretrained=True)
            model.classifier = checkpoint['classifier']
        else:
            raise ValueError(f"Unsupported architecture: {checkpoint['arch']}")
        
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        model.to(device)
        model.eval()
        
        return model
    
    except Exception as e:
        raise Exception(f"Error loading checkpoint: {str(e)}")

def process_image(image_path):
    """
    Process image for model input
    
    Args:
        image_path: Path to input image
        
    Returns:
        torch.Tensor: Processed image tensor
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        # Open and convert image to RGB
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Define transforms
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Process image
        img_tensor = preprocess(img)
        
        return img_tensor
    
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def predict(image_path, model, device, topk=5):
    """
    Predict the class of an image
    
    Args:
        image_path: Path to input image
        model: Trained PyTorch model
        device: Device to run inference on
        topk: Number of top predictions to return
        
    Returns:
        tuple: (probabilities, classes)
    """
    try:
        # Process image
        img = process_image(image_path)
        img = img.unsqueeze_(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(img)
            ps = torch.exp(output)
            top_p, top_class = ps.topk(topk, dim=1)
        
        # Convert indices to classes
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        classes = [idx_to_class[i] for i in top_class.cpu().numpy()[0]]
        probs = top_p.cpu().numpy()[0]
        
        return probs, classes
    
    except Exception as e:
        raise Exception(f"Error making prediction: {str(e)}")

def load_category_names(filepath):
    """
    Load category names from JSON file
    
    Args:
        filepath: Path to category names JSON file
        
    Returns:
        dict: Category names mapping
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Category names file not found: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise Exception(f"Error loading category names: {str(e)}")

def display_predictions(probs, classes, category_names=None):
    """
    Display predictions in a formatted way
    
    Args:
        probs: Prediction probabilities
        classes: Predicted class indices
        category_names: Optional mapping of classes to names
    """
    print("\nTop predictions:")
    print("-" * 50)
    
    if category_names:
        for prob, cls in zip(probs, classes):
            name = category_names.get(cls, cls)
            print(f"{name:<30} {prob*100:>6.2f}%")
    else:
        for prob, cls in zip(probs, classes):
            print(f"Class {cls:<25} {prob*100:>6.2f}%")
    
    print("-" * 50)

def main():
    """Main function"""
    # Get command line arguments
    args = get_input_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting prediction with arguments: {vars(args)}")
    
    try:
        # Set device
        device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        if args.gpu and not torch.cuda.is_available():
            logger.warning("GPU was requested but is not available. Using CPU instead.")
        
        # Load model
        model = load_checkpoint(args.checkpoint, device)
        logger.info("Model loaded successfully")
        
        # Load category names if provided
        category_names = None
        if args.category_names:
            category_names = load_category_names(args.category_names)
            logger.info("Category names loaded successfully")
        
        # Make prediction
        probs, classes = predict(args.image_path, model, device, args.top_k)
        logger.info("Prediction completed successfully")
        
        # Display results
        display_predictions(probs, classes, category_names)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == '__main__':
    main()
