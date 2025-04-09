import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import json
from Dataset import TrajectoryDataset
from ModelSocial import ImprovedSocialLSTM

class ConfidenceLoss(nn.Module):
    """
    Custom loss function that uses predicted confidence
    """
    def __init__(self):
        super(ConfidenceLoss, self).__init__()
        
    def forward(self, outputs, targets):
        # Extract predicted position and confidence
        pred_position = outputs[:, 0]
        pred_confidence = outputs[:, 1]
        
        # Position error (MSE)
        position_error = (pred_position - targets.squeeze()) ** 2
        
        # Scale error by confidence and add confidence regularization
        # Higher confidence → higher penalty for being wrong
        # Lower confidence → lower penalty but penalize low confidence
        confidence_penalty = -torch.log(pred_confidence)
        loss = (position_error * pred_confidence + confidence_penalty).mean()
        
        # Return loss components for logging
        return {
            'total_loss': loss,
            'position_error': position_error.mean(),
            'confidence': pred_confidence.mean()
        }

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, path='checkpoint.pt', min_delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.min_delta = min_delta
        self.path = path
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
    
    def __call__(self, val_loss, model, epoch, training_curve):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, training_curve)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, training_curve)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, epoch, training_curve):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        # Save model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_val_loss': val_loss,
            'training_curve': training_curve
        }, os.path.join(self.path, 'best_model.pth'))
        self.val_loss_min = val_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, early_stopping):
    training_curve = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch in train_bar:
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict['total_loss']
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            post_fix = {key: f'{value.item():.4f}' for key, value in loss_dict.items()}
            train_bar.set_postfix(post_fix)
        
        avg_train_loss = train_loss / len(train_loader)
        training_curve['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch in val_bar:
                inputs = batch['inputs'].to(device)
                targets = batch['targets'].to(device)

                outputs = model(inputs)
                loss_dict = criterion(outputs, targets)
                loss = loss_dict['total_loss']
                val_loss += loss.item()
                
                post_fix = {key: f'{value.item():.4f}' for key, value in loss_dict.items()}
                val_bar.set_postfix(post_fix)
        
        avg_val_loss = val_loss / len(val_loader)
        training_curve['val_loss'].append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Early stopping check
        early_stopping(avg_val_loss, model, epoch, training_curve)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

if __name__ == '__main__':
    # Training parameters
    patience = 8 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    lane_unit = 200  # each lane unit is 0.5 meters
    time_span = 10
    hidden_size = 64
    social_size = 32
    num_layers = 2
    learning_rate = 1e-4
    weight_decay = 1e-5
    dropout = 0.3
    num_epochs = 100
    
    # Model initialization
    model = ImprovedSocialLSTM(
        hidden_size=hidden_size,
        social_size=social_size,
        num_layers=num_layers,
        input_frames=time_span,
        output_size=2,  # Position and confidence
        dropout=dropout
    ).to(device)
    
    # Loss and optimizer
    criterion = ConfidenceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Create model save directory
    model_save_path = "D:\TimeSpaceDiagramDataset\SocialLSTMDataset\models\social_lstm"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    # Find next train number
    history_train_list = os.listdir(model_save_path)
    history_train_list = [x for x in history_train_list if os.path.isdir(os.path.join(model_save_path, x))]
    history_train_list.sort()
    
    if len(history_train_list) == 0:
        train_num = 0
    else:
        train_num = int(history_train_list[-1].split('_')[-1]) + 1
    
    model_save_path = os.path.join(model_save_path, f'train_{train_num}')
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_save_path, min_delta=0.01)
    os.makedirs(model_save_path)
    train_dir = r'D:\TimeSpaceDiagramDataset\SocialLSTMDataset\train\train_dataset.h5'
    val_dir = r'D:\TimeSpaceDiagramDataset\SocialLSTMDataset\val\val_dataset.h5'
    # Save training parameters
    with open(os.path.join(model_save_path, 'training_parameters.json'), 'w') as f:
        json.dump({
            'patience': patience,
            'device': device.type,
            'batch_size': batch_size,
            'lane_unit': lane_unit,
            'time_span': time_span,
            'hidden_size': hidden_size,
            'social_size': social_size,
            'num_layers': num_layers,
            'input_frames': time_span,
            'output_size': 2,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'num_epochs': num_epochs,
            'loss_func': criterion.__class__.__name__,
            'optimizer': optimizer.__class__.__name__,
            'model': model.__class__.__name__,
            'train_dir': train_dir,
            'val_dir': val_dir
        }, f)
    
    # Load datasets
    train_dataset = TrajectoryDataset(data_path = train_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    val_dataset = TrajectoryDataset(data_path= val_dir)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    # Train model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, early_stopping)

    print("Training complete")