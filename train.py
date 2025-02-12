#!/usr/bin/env python3
import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import os
import json
from tqdm import tqdm
import logging

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
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset')
    
    # Required argument
    parser.add_argument('data_dir', type=str, help='Path to dataset directory')
    
    # Optional arguments
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16',
                        choices=['vgg16', 'resnet18', 'densenet121'],
                        help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=4096,
                        help='Hidden units in classifier')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for training')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    
    return parser.parse_args()

def get_model(arch, hidden_units):
    """Build model with specified architecture"""
    # Architecture configurations
    architectures = {
        'vgg16': (models.vgg16, 25088),
        'resnet18': (models.resnet18, 512),
        'densenet121': (models.densenet121, 1024)
    }
    
    if arch not in architectures:
        raise ValueError(f"Architecture {arch} not supported")
    
    # Get model and input features
    model_func, input_features = architectures[arch]
    model = model_func(pretrained=True)
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Build classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    # Set classifier
    if arch == 'vgg16':
        model.classifier = classifier
    else:
        model.fc = classifier
        
    return model

def load_data(data_dir, batch_size=64):
    """Load and transform data"""
    # Validate data directory
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    
    # Define transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # Load datasets
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    }
    
    # Create dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(
            image_datasets['train'], 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4 if torch.cuda.is_available() else 0
        ),
        'valid': torch.utils.data.DataLoader(
            image_datasets['valid'], 
            batch_size=batch_size,
            num_workers=4 if torch.cuda.is_available() else 0
        )
    }
    
    return dataloaders, image_datasets['train'].class_to_idx

def train_model(model, dataloaders, criterion, optimizer, device, epochs, logger):
    """Train the model"""
    steps = 0
    print_every = 40
    best_accuracy = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    logger.info("Starting training...")
    for epoch in range(epochs):
        running_loss = 0
        model.train()
        
        # Training loop with progress bar
        train_bar = tqdm(dataloaders['train'], 
                        desc=f'Epoch {epoch+1}/{epochs}',
                        leave=False)
        
        for inputs, labels in train_bar:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_bar.set_postfix({'train_loss': loss.item()})
            
            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                
                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        valid_loss += criterion(outputs, labels).item()
                        
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                valid_loss = valid_loss/len(dataloaders['valid'])
                accuracy = accuracy/len(dataloaders['valid'])
                
                scheduler.step(valid_loss)
                
                logger.info(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {valid_loss:.3f}.. "
                          f"Accuracy: {accuracy:.3f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                
                running_loss = 0
                model.train()
    
    return best_accuracy

def save_checkpoint(model, save_dir, arch, class_to_idx, optimizer, epochs, accuracy):
    """Save model checkpoint"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    checkpoint = {
        'arch': arch,
        'classifier': model.classifier if arch == 'vgg16' else model.fc,
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'optimizer_state': optimizer.state_dict(),
        'epochs': epochs,
        'accuracy': accuracy
    }
    
    save_path = os.path.join(save_dir, f'{arch}_checkpoint.pth')
    torch.save(checkpoint, save_path)
    return save_path

def main():
    """Main function"""
    # Get command line arguments
    args = get_input_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting with arguments: {vars(args)}")
    
    try:
        # Set device
        device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        if args.gpu and not torch.cuda.is_available():
            logger.warning("GPU was requested but is not available. Using CPU instead.")
        
        # Load data
        dataloaders, class_to_idx = load_data(args.data_dir, args.batch_size)
        logger.info("Data loaded successfully")
        
        # Build model
        model = get_model(args.arch, args.hidden_units)
        model.to(device)
        logger.info(f"Built {args.arch} model with {args.hidden_units} hidden units")
        
        # Define loss and optimizer
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(
            model.classifier.parameters() if args.arch == 'vgg16' else model.fc.parameters(),
            lr=args.learning_rate
        )
        
        # Train model
        accuracy = train_model(
            model, dataloaders, criterion, optimizer,
            device, args.epochs, logger
        )
        
        # Save checkpoint
        save_path = save_checkpoint(
            model, args.save_dir, args.arch,
            class_to_idx, optimizer, args.epochs,
            accuracy
        )
        logger.info(f"Model saved to {save_path}")
        logger.info(f"Final accuracy: {accuracy:.3f}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == '__main__':
    main()
