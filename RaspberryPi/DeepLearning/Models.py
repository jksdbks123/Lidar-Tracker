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
from torch.nn.functional import binary_cross_entropy_with_logits

class EnhancedSpatialLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.2, gamma=0.2):
        super(EnhancedSpatialLoss, self).__init__()
        self.alpha = alpha  # Weight for element-wise BCE loss
        self.beta = beta    # Weight for temporal consistency loss
        self.gamma = gamma  # Weight for spatial consistency loss

    def forward(self, predictions, targets):
        # Element-wise BCE loss
        bce_loss = F.binary_cross_entropy(predictions, targets, reduction='mean')
        
        # Temporal consistency loss (horizontal differential)
        temporal_loss = F.mse_loss(predictions[:, 1:] - predictions[:, :-1],
                                   targets[:, 1:] - targets[:, :-1],
                                   reduction='mean')
        
        # Spatial consistency loss (vertical differential)
        spatial_loss = F.mse_loss(predictions[:, :, 1:] - predictions[:, :, :-1],
                                  targets[:, :, 1:] - targets[:, :, :-1],
                                  reduction='mean')
        
        # Combine losses
        total_loss = self.alpha * bce_loss + self.beta * temporal_loss + self.gamma * spatial_loss
        loss_dict = {
            'total_loss': total_loss,
            'bce_loss': bce_loss,
            'temporal_loss': temporal_loss,
            'spatial_loss': spatial_loss
        }
        
        return loss_dict

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        loss_dict = {
            'total_loss': torch.mean(F_loss),
        }
        return loss_dict

class BidirectionalLSTMLaneReconstructor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BidirectionalLSTMLaneReconstructor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional Encoder LSTM
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Decoder LSTM
        self.decoder = nn.LSTM(hidden_size * 2, hidden_size * 2, num_layers, batch_first=True)

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

        # Apply output layer
        outputs = self.output_layer(decoder_outputs)

        # Apply sigmoid to get values between 0 and 1
        reconstructed = self.sigmoid(outputs)
        reconstructed = reconstructed.reshape(-1,num_lane_unit, time_span)
        return reconstructed


class AttentionModule(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionModule, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch_size, hidden_size)
        # encoder_outputs: (batch_size, time_span, hidden_size)
        
        time_span = encoder_outputs.size(1)
        
        # Repeat hidden state time_span times
        hidden = hidden.unsqueeze(1).repeat(1, time_span, 1)
        
        # Calculate attention scores
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        
        # Apply softmax to get attention weights
        return F.softmax(attention, dim=1)

class LSTMTrajectoryReconstructorWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMTrajectoryReconstructorWithAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder LSTM
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Attention module
        self.attention = AttentionModule(hidden_size * 2)  # *2 for bidirectional

        # Decoder LSTM
        self.decoder = nn.LSTM(hidden_size * 2, hidden_size, num_layers, batch_first=True)

        # Output layer
        self.output_layer = nn.Linear(hidden_size, input_size)

        # Activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, num_lane_unit, time_span)
        batch_size, num_lane_unit, time_span = x.size()

        # Reshape input to (batch_size * num_lane_unit, time_span, 1)
        x_reshaped = x.view(batch_size * num_lane_unit, time_span, 1)

        # Encode the input sequence
        encoder_outputs, (hidden, cell) = self.encoder(x_reshaped)

        # Prepare hidden state for attention (combine forward and backward)
        hidden_for_attn = torch.cat([hidden[-2], hidden[-1]], dim=1)

        # Calculate attention weights
        attn_weights = self.attention(hidden_for_attn, encoder_outputs)

        # Apply attention weights to encoder outputs
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        # Repeat context for each time step
        context_repeated = context.repeat(1, time_span, 1)

        # Decode all steps at once
        decoder_output, _ = self.decoder(context_repeated, (hidden[-self.num_layers:], cell[-self.num_layers:]))

        # Apply output layer to all time steps
        outputs = self.output_layer(decoder_output)

        # Reshape outputs to (batch_size, num_lane_unit, time_span)
        outputs = outputs.view(batch_size, num_lane_unit, time_span)

        # Apply sigmoid to get values between 0 and 1
        reconstructed = self.sigmoid(outputs)

        return reconstructed, attn_weights

class LSTMTrajectoryReconstructor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMTrajectoryReconstructor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder LSTM
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Decoder LSTM
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Output layer
        self.output_layer = nn.Linear(hidden_size, input_size)

        # Activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, num_lane_unit, time_span)
        batch_size, num_lane_unit, time_span = x.size()

        # Reshape input to (batch_size * num_lane_unit, time_span, 1)
        x_reshaped = x.view(batch_size * num_lane_unit, time_span, 1)

        # Encode the input sequence
        _, (hidden, cell) = self.encoder(x_reshaped)

        # Create decoder input: all zeros, shape (batch_size * num_lane_unit, time_span, hidden_size)
        decoder_input = torch.zeros(batch_size * num_lane_unit, time_span, self.hidden_size, device=x.device)

        # Decode all steps at once
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))



        # Apply output layer to all time steps
        outputs = self.output_layer(decoder_output)

        # Reshape outputs to (batch_size, num_lane_unit, time_span)
        outputs = outputs.view(batch_size, num_lane_unit, time_span)

        # Apply sigmoid to get values between 0 and 1
        reconstructed = self.sigmoid(outputs)

        return reconstructed

class BidirectionalLSTMTrajectoryReconstructor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BidirectionalLSTMTrajectoryReconstructor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional Encoder LSTM
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Decoder LSTM (unidirectional for simplicity)
        self.decoder = nn.LSTM(hidden_size * 2, hidden_size * 2, num_layers, batch_first=True)

        # Output layer
        self.output_layer = nn.Linear(hidden_size * 2, input_size)

        # Activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, num_lane_unit, time_span)
        batch_size, num_lane_unit, time_span = x.size()

        # Reshape input to (batch_size * num_lane_unit, time_span, 1)
        x_reshaped = x.view(batch_size * num_lane_unit, time_span, self.input_size)

        # Encode the input sequence
        encoder_outputs, (hidden, cell) = self.encoder(x_reshaped)

        # Prepare hidden and cell states for the decoder
        # Concatenate forward and backward states
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
        cell = cell.view(self.num_layers, 2, -1, self.hidden_size)
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)

        # Decode
        decoder_outputs, _ = self.decoder(encoder_outputs, (hidden, cell))

        # Apply output layer
        outputs = self.output_layer(decoder_outputs)

        # Reshape outputs
        outputs = outputs.view(batch_size, num_lane_unit, time_span)

        # Apply sigmoid to get values between 0 and 1
        reconstructed = self.sigmoid(outputs)
        return reconstructed

    
class ConsistentTrajectoryLoss(nn.Module):
    def __init__(self,num_lane_units = 200):
        super(ConsistentTrajectoryLoss, self).__init__()
        # further range should have higher weights, say linearly increasing from 0 to 1 accoridng to the dimension of the lane_unit 200
        weights = torch.linspace(0, 1, num_lane_units)
        weights = weights / weights.sum()
        weights = weights.view(1, -1, 1)
        self.bce_loss = nn.BCELoss(weight=weights) # Binary Cross-Entropy Loss
        

    def forward(self, predictions, targets):
        # predictions: (batch, 200, future_len)
        # targets: (batch, 200, future_len)
        # input_series: (batch, 200, his_len)
        # 1. Binary Cross-Entropy Loss
        bce_loss = self.bce_loss(predictions, targets)
        # Combine losses
        total_loss = bce_loss
        loss_dict = {
            'total_loss': total_loss,
        }
        return loss_dict


class RangeWeightedBCELoss(nn.Module):
    def __init__(self, num_lane_units = 200, max_weight=1):
        super(RangeWeightedBCELoss, self).__init__()
        self.num_lane_units = num_lane_units
        self.max_weight = max_weight
        
        # Create weights that linearly increase from 1 to max_weight
        self.weights = torch.linspace(1, max_weight, num_lane_units)
        self.weights = self.weights.view(-1, 1)

        self.bce_loss = nn.BCELoss(weight=self.weights) # Binary Cross-Entropy Loss

        
    def forward(self, predictions, targets):
        total_loss = self.bce_loss(predictions, targets)
        loss_dict = {
            'total_loss': total_loss,
        }
        return loss_dict


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

        # Convert to PyTorch tensors
        target_tensor = torch.FloatTensor(target)
        post_occ_tensor = torch.FloatTensor(post_occ_label)


        return {
            'post_occ_X': post_occ_tensor,
            # 'occ_mask_X': occ_mask_tensor,
            'target': target_tensor
        }