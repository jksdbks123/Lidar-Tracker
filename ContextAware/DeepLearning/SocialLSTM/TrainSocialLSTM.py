import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import json
from Dataset import SocialLSTMDataset,MemoryMappedSocialLSTMDataset
from ModelSocial import LaneSocialLSTM
from CriterionSocial import combined_distribution_loss,focal_loss

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
        
        for input_positions, input_context, target_positions, target_distributions, output_masks in train_bar:
            # Move tensors to device
            input_positions = input_positions.to(device)
            input_context = input_context.to(device)
            target_positions = target_positions.to(device)
            target_distributions = target_distributions.to(device)
            output_masks = output_masks.to(device)

            optimizer.zero_grad()
            # Forward pass
            pred_distributions = model(input_positions, input_context)
            

            # Compute loss
            loss = criterion(
                pred_distributions,target_positions, output_masks
            )
            
            # Backward pass
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            # Record loss
            train_losses.append(loss.item())
            # train_js_losses.append(js_loss.item())
            # train_pos_losses.append(pos_loss.item())
            
            post_fix = {'Train Loss': loss.item()}
            # Record average confidence
            train_bar.set_postfix(post_fix)
        
        # Validation phase
        model.eval()
        val_losses = []
        # val_js_losses = []
        # val_pos_losses = []

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for input_positions, input_context, target_positions, target_distributions, output_masks in val_bar:
                
                # Move tensors to device
                input_positions = input_positions.to(device)
                input_context = input_context.to(device)
                target_positions = target_positions.to(device)
                target_distributions = target_distributions.to(device)
                output_masks = output_masks.to(device)

                # Forward pass
                pred_distributions = model(input_positions, input_context)
                
                # Compute loss
                loss = criterion(
                pred_distributions,target_positions, output_masks
            )
                
                # Record metrics
                val_losses.append(loss.item())
                # val_js_losses.append(js_loss.item())
                # val_pos_losses.append(pos_loss.item())
                

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
    hidden_size=128
    social_size=32
    neighborhood_size=16
    num_layers=1
    input_frames=10
    output_frames=1
    lane_cells=200
    dropout=0.2
    num_epochs = 100
    batch_size = 32
    learning_rate = 0.001
    weight_decay = 0.0001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Model initialization
    model = LaneSocialLSTM(
        hidden_size=hidden_size,
        social_size=social_size,
        neighborhood_size=neighborhood_size,
        num_layers=num_layers,
        input_frames=input_frames,
        output_frames=output_frames,
        lane_cells=lane_cells,
        dropout = dropout,
        device=device
    )
    
    # Loss and optimizer
    # criterion = combined_distribution_loss
    criterion = focal_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    
    # Create model save directory
    model_save_path = "D:\TimeSpaceDiagramDataset\SocialLSTMDataset\models\social_lstm"
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
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_save_path, min_delta=0.001)
    os.makedirs(model_save_path, exist_ok=True)
    train_h5_dir = r'D:\TimeSpaceDiagramDataset\SocialLSTMDataset\dataset\train\social_lstm_data.h5'
    train_dataset = MemoryMappedSocialLSTMDataset(
        h5_path=train_h5_dir,
        input_frames=10,
        output_frames=1
    )
    train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=8,
        )
    val_h5_path = r'D:\TimeSpaceDiagramDataset\SocialLSTMDataset\dataset\val\social_lstm_data.h5'
    val_dataset = MemoryMappedSocialLSTMDataset(
        h5_path=val_h5_path,
        input_frames=10,
        output_frames=1
    )
    val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=8,

        )
    # Save training parameters
    with open(os.path.join(model_save_path, 'training_parameters.json'), 'w') as f:
        json.dump({
            'hidden_size': hidden_size,
            'social_size': social_size,
            'neighborhood_size': neighborhood_size,
            'num_layers': num_layers,
            'input_frames': input_frames,
            'output_frames': output_frames,
            'lane_cells': lane_cells,
            'dropout': dropout,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay
        }, f)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(max_lr=0.003,
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