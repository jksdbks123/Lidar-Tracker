import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
# Custom Dataset
# from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from tqdm import tqdm
import json


#PyTorch
class FocalLoss(nn.Module):
    """
    alpha: a float value between 0 and 1 representing a weighting factor used to deal with class imbalance. Positive classes and negative classes have alpha and (1 - alpha) as their weighting factors respectively. Defaults to 0.25.
    gamma: a positive float value representing the tunable focusing parameter, defaults to 2.

    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='sum'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure inputs are probabilities
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Here, pt is directly the predicted probability
        pt = inputs * targets + (1 - inputs) * (1 - targets)
        
        # Calculate focal loss
        focal_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            focal_loss =  torch.sum(focal_loss)
        loss_dict = {
            'total_loss': focal_loss,
        }
        return loss_dict

class UnidirectionalLSTMLaneReconstructor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,droupout=0.2):
        super(UnidirectionalLSTMLaneReconstructor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Unidirectional Encoder LSTM
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)

        # Decoder LSTM
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

        self.dropout = nn.Dropout(droupout)
        # Output layer
        self.output_layer = nn.Linear(hidden_size, input_size)

        # Activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, time_span, num_lane_unit)
        batch_size, num_lane_unit, time_span = x.size()
        x = x.view(batch_size, time_span, num_lane_unit)

        # Encode the input sequence
        encoder_outputs, (hidden, cell) = self.encoder(x)

        # Decoder
        decoder_outputs, _ = self.decoder(encoder_outputs, (hidden, cell))
        decoder_outputs = self.dropout(decoder_outputs)

        # Apply output layer
        outputs = self.output_layer(decoder_outputs)

        # Apply sigmoid to get values between 0 and 1
        reconstructed = self.sigmoid(outputs)
        reconstructed = reconstructed.reshape(-1, num_lane_unit, time_span)
        return reconstructed

class BidirectionalLSTMLaneReconstructor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, droupout=0.2):
        super(BidirectionalLSTMLaneReconstructor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional Encoder LSTM
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Decoder LSTM
        self.decoder = nn.LSTM(hidden_size * 2, hidden_size * 2, num_layers, batch_first=True)

        self.dropout = nn.Dropout(droupout)
        # Output layer
        self.output_layer = nn.Linear(hidden_size * 2, input_size)


        # Activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, time_span, num_lane_unit)
        batch_size, num_lane_unit, time_span = x.size()
        x = x.view(batch_size, time_span, num_lane_unit)
        # Encode the input sequence
        encoder_outputs, (hidden, cell) = self.encoder(x)

        # Prepare hidden and cell states for the decoder
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
        cell = cell.view(self.num_layers, 2, batch_size, self.hidden_size)
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)

        # Decode
        decoder_outputs, _ = self.decoder(encoder_outputs, (hidden, cell))
        decoder_outputs = self.dropout(decoder_outputs)
        # Apply output layer
        outputs = self.output_layer(decoder_outputs)

        # Apply sigmoid to get values between 0 and 1
        reconstructed = self.sigmoid(outputs)
        reconstructed = reconstructed.reshape(-1,num_lane_unit, time_span)
        return reconstructed



class TrajDataset(Dataset):
    def __init__(self, data_dir, time_span):
        self.data_dir = data_dir
        self.time_span = time_span

        # Determine which folder to use based on whether it's training or validation
        self.folder_path = data_dir

        # Get list of file names (assuming they're numbered consistently across subfolders)
        self.file_names = [f for f in os.listdir(os.path.join(self.folder_path, 'target')) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        # Load target (shape: 200, future_length)
        target_path = os.path.join(self.folder_path, 'target', file_name)
        target = np.load(target_path)
        target = target[:,:self.time_span]

        # Load post-occlusion label (shape: 200, historical_length)
        post_occ_path = os.path.join(self.folder_path, 'post_occ_label', file_name)
        post_occ_label = np.load(post_occ_path)
        post_occ_label = post_occ_label[:,:self.time_span]
        
        speed_label_path = os.path.join(self.folder_path, 'speed_label', file_name)
        speed_label = np.load(speed_label_path)
        speed_label = speed_label[:,:self.time_span]

        traj_id_label_path = os.path.join(self.folder_path, 'traj_id_label', file_name)
        traj_id_label = np.load(traj_id_label_path)
        traj_id_label = traj_id_label[:,:self.time_span]

        # Convert to PyTorch tensors
        target_tensor = torch.FloatTensor(target)
        post_occ_tensor = torch.FloatTensor(post_occ_label)
        speed_tensor = torch.FloatTensor(speed_label)
        traj_id_tensor = torch.FloatTensor(traj_id_label)


        return {
            'post_occ_X': post_occ_tensor,
            'speed_target': speed_tensor,
            'target': target_tensor,
            'traj_id': traj_id_tensor
        }

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, verbose=False, path='./'):
        self.patience = patience
        self.min_delta = min_delta # Minimum change in the monitored quantity to qualify as an improvement
        self.verbose = verbose
        self.path = path
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model, epoch, training_curve):
    

        with open(os.path.join(self.path, 'training_curve.json'), 'w') as f:
                json.dump(training_curve, f)

        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, epoch)
            
            
            
        elif self.best_loss - val_loss <= self.min_delta:
            if val_loss < self.best_loss:
                self.save_checkpoint(val_loss, model, epoch)
                self.best_loss = val_loss
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(val_loss, model, epoch)
            self.best_loss = val_loss
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), os.path.join(self.path, f'checkpoint_{epoch}.pth'))

