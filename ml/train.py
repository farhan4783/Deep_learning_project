import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import json

from app.models.detector import EfficientNetDetector, XceptionDetector, CustomCNNDetector


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=50,
    learning_rate=0.0001,
    device='cuda',
    save_dir='models'
):
    """
    Train a deepfake detection model
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        save_dir: Directory to save checkpoints
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{train_loss/len(pbar):.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{val_loss/len(pbar):.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }
            torch.save(checkpoint, save_path / 'best_model.pth')
            print(f'✓ Saved best model (Val Acc: {val_acc:.2f}%)')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }
            torch.save(checkpoint, save_path / f'checkpoint_epoch_{epoch+1}.pth')
    
    # Save training history
    with open(save_path / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f'\n✓ Training complete! Best Val Acc: {best_val_acc:.2f}%')
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    parser.add_argument('--model', type=str, default='efficientnet', 
                        choices=['efficientnet', 'xception', 'custom_cnn'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save-dir', type=str, default='models')
    
    args = parser.parse_args()
    
    # Create model
    if args.model == 'efficientnet':
        model = EfficientNetDetector()
    elif args.model == 'xception':
        model = XceptionDetector()
    else:
        model = CustomCNNDetector()
    
    print(f'Training {args.model} model...')
    print(f'Device: {args.device}')
    print(f'Epochs: {args.epochs}')
    print(f'Learning Rate: {args.lr}')
    
    # Note: You need to implement data loaders for your dataset
    # This is a template - replace with actual data loading code
    print('\n⚠️  Note: You need to implement data loaders for your dataset')
    print('See the data/ directory for dataset preparation utilities')
