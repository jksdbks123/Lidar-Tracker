import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
# Custom Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from tqdm import tqdm
from Models import *

def train_model(model_save_folder,model_name, model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=5):

    early_stopping = EarlyStopping(patience=patience, verbose=True, path=os.path.join(model_save_folder, model_name))
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        # train_activity_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in train_bar:
            targets = batch['target']  
            post_occlusion = batch['post_occ_X']

            targets, post_occlusion = targets.to(device), post_occlusion.to(device)
            optimizer.zero_grad()
            # outputs, _ = model(post_occlusion)
            outputs = model(post_occlusion)

            loss_dict = criterion(outputs, targets)
            loss = loss_dict['total_loss']
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # train_activity_loss += loss_dict['activity_loss'].item()
            # Update progress bar with all losses
            post_fix = {key: f'{value.item():.4f}' for key, value in loss_dict.items()}
            train_bar.set_postfix(post_fix)
        
        avg_train_loss = train_loss / len(train_loader)
        # avg_activity_loss = train_activity_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}')
        # Validation phase
        model.eval()
        val_loss = 0.0
        # val_activity_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch in val_bar:
                targets = batch['target']  
                post_occlusion = batch['post_occ_X']
                targets, post_occlusion = targets.to(device), post_occlusion.to(device)
                
                # outputs, _ = model(post_occlusion)
                outputs = model(post_occlusion)
                loss_dict = criterion(outputs, targets)
                loss = loss_dict['total_loss']
                val_loss += loss.item()
                post_fix = {key: f'{value.item():.4f}' for key, value in loss_dict.items()}
                val_bar.set_postfix(post_fix)
        
        avg_val_loss = val_loss / len(val_loader)
        # avg_val_activity_loss = val_activity_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        # Early stopping check
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
0
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta # Minimum change in the monitored quantity to qualify as an improvement
        self.verbose = verbose
        self.path = path
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            
            self.save_checkpoint(val_loss, model)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.best_score:.6f} --> {-val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)

if __name__ == '__main__':

    patience = 8 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    lane_unit = 200
    time_span = 10
    hidden_size = 64
    num_layers = 2
    input_size = lane_unit
    learning_rate = 0.0001
    num_epochs = 100
    model = BidirectionalLSTMLaneReconstructor(input_size, hidden_size, num_layers).to(device)
    # criterion = RangeWeightedBCELoss().to(device)
    criterion = EnhancedSpatialLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Create datasets
    mocel_save_path = r"D:\TimeSpaceDiagramDataset\EncoderDecoder_EvenlySampled_FreeflowAug_0826\models"
    if not os.path.exists(mocel_save_path):
        os.makedirs(mocel_save_path)
    model_name = f'lstm_encoder_decoder_64_bi_t_10_hidden_128.pth'
    train_dataset = TrajDataset(data_dir=r"D:\TimeSpaceDiagramDataset\EncoderDecoder_EvenlySampled_FreeflowAug_0826\train", time_span=time_span)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_dataset = TrajDataset(data_dir=r"D:\TimeSpaceDiagramDataset\EncoderDecoder_EvenlySampled_FreeflowAug_0826\val", time_span=time_span)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=8)

    # Train the model
    train_model(mocel_save_path,model_name, model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    print("Training complete")