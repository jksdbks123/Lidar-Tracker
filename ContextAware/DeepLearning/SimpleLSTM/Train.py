import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import json
from Dataset import TrajDataset,PicklableH5Dataset
from Models import TrajectoryLSTM

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
    
    def __call__(self, val_loss, model, epoch, history):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, history)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, history)
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

def train_model(model, 
                train_loader, 
                val_loader, 
                criterion, 
                optimizer, 
                num_epochs, 
                device, 
                early_stopping, 
                scheduler):
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, batch in enumerate(train_bar):
            sequence_tensor, mask_tensor, target_tensor = batch 
            # Move tensors to device
            sequence_tensor = sequence_tensor.to(device)
            mask_tensor = mask_tensor.to(device)
            target_tensor = target_tensor.to(device)
            optimizer.zero_grad()
            # Forward pass
            pred = model(sequence_tensor, mask_tensor)
            # Compute loss
            loss = criterion(pred, target_tensor)

            # Backward pass
            loss.backward()
            # Gradient clipping
            optimizer.step()
            # Record loss
            train_losses.append(loss.item())
            
            post_fix = {'Train Loss': loss.item()}
            # Record average confidence
            train_bar.set_postfix(post_fix)
        
        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch_idx, batch in enumerate(val_bar):
                # Move tensors to device
                sequence_tensor = sequence_tensor.to(device)
                mask_tensor = mask_tensor.to(device)
                target_tensor = target_tensor.to(device)
                if torch.all(~mask_tensor):
                    continue

                # Forward pass
                pred = model(sequence_tensor, mask_tensor)
                
                # Compute loss
                loss = criterion(pred, target_tensor)
                # Record metrics
                val_losses.append(loss.item())
                

                post_fix = {'Val Loss': loss.item()}
                val_bar.set_postfix(post_fix)
        
        # Calculate average metrics
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"LR: {current_lr:.6f}")

        # Early stopping check
        early_stopping(avg_val_loss, model, epoch, history)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

if __name__ == '__main__':
    # Training parameters

    patience = 8 
    hidden_size=64
    num_layers=2
    input_frames=20
    output_frames=1
    lane_cells=200
    time_span = 100 # 100 frames in reading dataset
    dropout=0.2
    num_epochs = 100
    batch_size = 32
    learning_rate = 0.001
    weight_decay = 0.0001
    occlusion_rate = 0.2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Model initialization
    model = TrajectoryLSTM(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout = dropout,
    )
    model.to(device)
    # Loss and optimizer
    # criterion = combined_distribution_loss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    
    # Create model save directory
    model_save_path = "D:\TimeSpaceDiagramDataset\SimpleLSTM\Models"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)
    
    # Find next train number
    history_train_list = os.listdir(model_save_path)
    history_train_list = [x for x in history_train_list if os.path.isdir(os.path.join(model_save_path, x))]
    history_train_list.sort()
    
    if len(history_train_list) == 0:
        train_num = 0
    else:
        train_num = int(history_train_list[-1].split('_')[-1]) + 1
    
    model_save_path = os.path.join(model_save_path, f'train_{train_num}')
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_save_path, min_delta=0.0001)
    os.makedirs(model_save_path, exist_ok=True)
    train_dir = r'D:\TimeSpaceDiagramDataset\SimpleLSTM\Dataset\train\train_data.h5'
    train_dataset = PicklableH5Dataset(train_dir,
                          occlusion_rate=occlusion_rate)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    val_dir = r'D:\TimeSpaceDiagramDataset\SimpleLSTM\Dataset\val\val_data.h5'
    val_dataset = PicklableH5Dataset(val_dir,
                          occlusion_rate=occlusion_rate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
    # Move model to device
    # Save training parameters
    with open(os.path.join(model_save_path, 'training_parameters.json'), 'w') as f:
        json.dump({
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'input_frames': input_frames,
            'output_frames': output_frames,
            'lane_cells': lane_cells,
            'time_span': time_span,
            'dropout': dropout,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'occlusion_rate': occlusion_rate,
            'patience': patience,
            'train_num': train_num
        }, f, indent=4)
    
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer=optimizer,
    max_lr=0.003,
    steps_per_epoch=len(train_loader),
    epochs=50,
    pct_start=0.3)

    # Train model
    train_model(model, 
                train_loader, 
                val_loader,
                criterion, 
                optimizer, 
                num_epochs, 
                device, 
                early_stopping,
                scheduler)

    print("Training complete")