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

class BidirectionalRNNLaneReconstructor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, droupout=0.2):
        super(BidirectionalRNNLaneReconstructor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional Encoder RNN (replacing LSTM)
        self.encoder = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Decoder RNN (replacing LSTM)
        self.decoder = nn.RNN(hidden_size * 2, hidden_size * 2, num_layers, batch_first=True)

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
        encoder_outputs, hidden = self.encoder(x)
        
        # Prepare hidden state for the decoder (RNN has no cell state)
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
        
        # Decode - RNN only needs hidden state (no cell state)
        decoder_outputs, _ = self.decoder(encoder_outputs, hidden)
        decoder_outputs = self.dropout(decoder_outputs)
        
        # Apply output layer
        outputs = self.output_layer(decoder_outputs)
        
        # Apply sigmoid to get values between 0 and 1
        reconstructed = self.sigmoid(outputs)
        reconstructed = reconstructed.reshape(-1, num_lane_unit, time_span)
        return reconstructed
    
class AttentionModule(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionModule, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)
        
    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden shape: [batch_size, hidden_size*2]
        # encoder_outputs shape: [batch_size, time_span, hidden_size*2]
        
        batch_size, time_span, hidden_dim = encoder_outputs.size()
        
        # Repeat decoder hidden state for each time step
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).repeat(1, time_span, 1)
        
        # Calculate attention scores
        attn_scores = self.attention(encoder_outputs).squeeze(-1)  # [batch_size, time_span]
        attn_weights = F.softmax(attn_scores, dim=1)  # [batch_size, time_span]
        
        # Apply attention weights to encoder outputs
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [batch_size, 1, hidden_dim]
        attn_applied = attn_applied.squeeze(1)  # [batch_size, hidden_dim]
        
        # Return context vector and attention weights
        return attn_applied, attn_weights

class BidirectionalRNNLaneReconstructorWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(BidirectionalRNNLaneReconstructorWithAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional Encoder RNN
        self.encoder = nn.RNN(input_size, hidden_size, num_layers, 
                             batch_first=True, bidirectional=True)
        
        # Attention module
        self.attention = AttentionModule(hidden_size)
        
        # Decoder RNN
        self.decoder = nn.RNN(hidden_size * 2, hidden_size * 2, 
                             num_layers, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size * 2, input_size)
        
        # Activation function
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch_size, num_lane_unit, time_span)
        batch_size, num_lane_unit, time_span = x.size()
        x = x.view(batch_size, time_span, num_lane_unit)
        
        # Encode the input sequence
        encoder_outputs, hidden = self.encoder(x)
        
        # Prepare hidden state for the decoder
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
        hidden_forward = hidden[:, 0, :, :]
        hidden_backward = hidden[:, 1, :, :]
        hidden = torch.cat([hidden_forward, hidden_backward], dim=2)
        
        # Create container for decoder outputs
        decoder_outputs = torch.zeros(batch_size, time_span, 
                                     self.hidden_size * 2, 
                                     device=x.device)
        
        # Initial decoder hidden state
        decoder_hidden = hidden
        
        # Decode step by step with attention
        for t in range(time_span):
            # Apply attention mechanism
            context, _ = self.attention(decoder_hidden[-1], encoder_outputs)
            
            # Feed context vector to decoder RNN
            context_input = context.unsqueeze(1)  # [batch_size, 1, hidden_size*2]
            output, decoder_hidden = self.decoder(context_input, decoder_hidden)
            
            # Store decoder output
            decoder_outputs[:, t:t+1, :] = output
        
        # Apply dropout
        decoder_outputs = self.dropout(decoder_outputs)
        
        # Apply output layer
        outputs = self.output_layer(decoder_outputs)
        
        # Apply sigmoid to get values between 0 and 1
        reconstructed = self.sigmoid(outputs)
        reconstructed = reconstructed.reshape(batch_size, num_lane_unit, time_span)
        
        return reconstructed


class AttentionModule(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionModule, self).__init__()
        # The hidden size here will be hidden_size * 2 because of bidirectional LSTM
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, encoder_outputs):
        # encoder_outputs shape: (batch_size, time_span, hidden_size)
        
        # Calculate attention scores
        attention_scores = self.attention(encoder_outputs)  # (batch_size, time_span, 1)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, time_span, 1)
        
        # Apply attention weights to encoder outputs
        context_vector = torch.sum(attention_weights * encoder_outputs, dim=1)  # (batch_size, hidden_size)
        
        return context_vector, attention_weights

class BidirectionalLSTMLaneReconstructorWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(BidirectionalLSTMLaneReconstructorWithAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional Encoder LSTM
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # Attention mechanism
        self.attention = AttentionModule(hidden_size * 2)
        
        # Decoder LSTM
        self.decoder = nn.LSTM(hidden_size * 4, hidden_size * 2, num_layers, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        
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
        
        # Apply attention at each timestep of decoding
        decoder_inputs = encoder_outputs
        decoder_outputs = []
        
        # Store attention weights for visualization if needed
        attention_weights_list = []
        
        # Initialize decoder hidden state
        decoder_hidden = (hidden, cell)
        
        # Process each timestep
        for t in range(time_span):
            # Get attention context for current timestep
            context_vector, attention_weights = self.attention(encoder_outputs)
            attention_weights_list.append(attention_weights)
            
            # Expand context vector to match timesteps for concatenation
            context_vector_expanded = context_vector.unsqueeze(1).expand(-1, 1, -1)
            
            # Concatenate context vector with current input
            current_input = decoder_inputs[:, t:t+1, :]
            augmented_input = torch.cat([current_input, context_vector_expanded], dim=2)
            
            # Run through decoder for this timestep
            output, decoder_hidden = self.decoder(augmented_input, decoder_hidden)
            decoder_outputs.append(output)
        
        # Concatenate outputs from all timesteps
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = self.dropout(decoder_outputs)
        
        # Apply output layer
        outputs = self.output_layer(decoder_outputs)
        
        # Apply sigmoid to get values between 0 and 1
        reconstructed = self.sigmoid(outputs)
        reconstructed = reconstructed.reshape(-1, num_lane_unit, time_span)
        
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

