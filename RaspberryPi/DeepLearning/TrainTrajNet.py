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
import json
# import focal loss
from torchvision.ops import sigmoid_focal_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, early_stopping):

    training_curve = {'train_loss': [], 'val_loss': []}

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
        training_curve['train_loss'].append(avg_train_loss)
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
        training_curve['val_loss'].append(avg_val_loss)
        # avg_val_activity_loss = val_activity_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        # Early stopping check
        early_stopping(avg_val_loss, model, epoch,training_curve)
        if early_stopping.early_stop:
            # save the training curve
            print("Early stopping triggered")
            break

if __name__ == '__main__':

    patience = 8 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    lane_unit = 200 # in default, the model is trained in 100 meters, so each lane unit is 0.5 meters
    time_span = 100
    hidden_size = 128
    num_layers = 2
    input_size = lane_unit
    learning_rate = 1e-4
    weight_decay = 1e-5
    dropout = 0.5
    num_epochs = 100
    alpha= 0.97
    gamma= 4
    model = BidirectionalLSTMLaneReconstructor(input_size, hidden_size, num_layers,droupout=dropout).to(device)
    # model = UnidirectionalLSTMLaneReconstructor(input_size, hidden_size, num_layers).to(device)
    criterion = FocalLoss(alpha,gamma).to(device)
    # criterion = SpeedFocalLoss(alpha,gamma).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Create datasets
    model_save_path = r"D:\TimeSpaceDiagramDataset\EncoderDecoder_EvenlySampled_FreeflowAug_0911_5res_lanechange_freeflow&signal\models"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    # new training folder will be named as "train_num"
    # read the history training folder and get the latest training folder
    history_train_list = os.listdir(model_save_path)
    # only keep folder
    history_train_list = [x for x in history_train_list if os.path.isdir(os.path.join(model_save_path, x))]
    history_train_list.sort()
    if len(history_train_list) == 0:
        train_num = 0
    else:
        train_num = int(history_train_list[-1].split('_')[-1]) + 1
    model_save_path = os.path.join(model_save_path, f'train_{train_num}')
    early_stopping = EarlyStopping(patience=patience, verbose=True, path= model_save_path, min_delta=5)
    os.makedirs(model_save_path)
    # save the training parameters as .json
    with open(os.path.join(model_save_path, 'training_parameters.json'), 'w') as f:
        json.dump({
            'patience': patience,
            'device': device.type,
            'batch_size': batch_size,
            'lane_unit': lane_unit,
            'time_span': time_span,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'focal_alpha' : alpha,
            'focal_gamma' : gamma,
            'input_size': input_size,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'num_epochs': num_epochs,
            'loss_func': criterion.__class__.__name__,
            'optimizer': optimizer.__class__.__name__,
            'model': model.__class__.__name__
        }, f)
    
    train_dataset = TrajDataset(data_dir=r"D:\TimeSpaceDiagramDataset\EncoderDecoder_EvenlySampled_FreeflowAug_0911_5res_lanechange_freeflow&signal\100_frame\train", time_span=time_span)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16)
    val_dataset = TrajDataset(data_dir=r"D:\TimeSpaceDiagramDataset\EncoderDecoder_EvenlySampled_FreeflowAug_0911_5res_lanechange_freeflow&signal\100_frame\val", time_span=time_span)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=8)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, early_stopping)

    print("Training complete")