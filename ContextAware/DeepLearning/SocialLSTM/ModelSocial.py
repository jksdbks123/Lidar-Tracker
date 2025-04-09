import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimplifiedLaneSocialLSTM(nn.Module):
    def __init__(self, 
                 hidden_size=64,
                 social_size=32,
                 neighborhood_size=16,
                 num_layers=1,
                 input_frames=10,
                 output_frames=12,
                 lane_cells=200,
                 dropout=0.2,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(SimplifiedLaneSocialLSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.social_size = social_size
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.lane_cells = lane_cells
        self.neighborhood_size = neighborhood_size
        
        # Position embedding (for vehicle's front position)
        self.position_embedding = nn.Linear(1, hidden_size)
        
        # Social pooling embedding
        self.social_pooling_embedding = nn.Linear(neighborhood_size, social_size)
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=hidden_size + social_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output decoder
        self.decoder_rnn = nn.LSTMCell(
            input_size=hidden_size,
            hidden_size=hidden_size
        )
        
        # Output layers - position and confidence
        self.output_position = nn.Linear(hidden_size, 1)
        self.output_confidence = nn.Linear(hidden_size, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.to(device)
    
    def get_neighborhood(self, traffic_context, current_pos):
        """
        Extract traffic context neighborhood around current position
        
        Args:
            traffic_context: Binary lane occupancy [batch_size, lane_cells]
            current_pos: Current front position indices [batch_size]
            
        Returns:
            neighborhood: Extracted neighborhood [batch_size, neighborhood_size]
        """
        batch_size = traffic_context.shape[0]
        neighborhood = torch.zeros(batch_size, self.neighborhood_size, device=self.device)
        
        for b in range(batch_size):
            # Center position for this batch
            pos = current_pos[b].long()
            
            # Extract window around position
            start_idx = max(0, pos - self.neighborhood_size // 2)
            end_idx = min(self.lane_cells, pos + self.neighborhood_size // 2 + self.neighborhood_size % 2)
            width = end_idx - start_idx
            
            if width > 0:
                # Calculate offset to center the neighborhood window
                offset = max(0, self.neighborhood_size // 2 - pos)
                neighborhood[b, offset:offset+width] = traffic_context[b, start_idx:end_idx]
                
        return neighborhood
    
    def forward(self, vehicle_positions, traffic_context):
        """
        Forward pass with social pooling
        
        Args:
            vehicle_positions: Vehicle front positions [batch_size, input_frames]
            traffic_context: Binary traffic context [batch_size, input_frames, lane_cells]
            
        Returns:
            predicted_positions: Predicted positions [batch_size, output_frames]
            confidences: Prediction confidences [batch_size, output_frames]
        """
        batch_size = vehicle_positions.shape[0]
        
        # Prepare LSTM inputs
        lstm_inputs = torch.zeros(batch_size, self.input_frames, self.hidden_size + self.social_size, device=self.device)
        
        # Process each input time step
        for t in range(self.input_frames):
            # Current position
            pos = vehicle_positions[:, t].unsqueeze(1).float()  # [batch, 1]
            
            # Embed position
            pos_embedded = self.dropout(self.relu(self.position_embedding(pos)))
            
            # Get neighborhood for social pooling
            neighborhood = self.get_neighborhood(traffic_context[:, t], vehicle_positions[:, t])
            
            # Embed social context
            social_context = self.dropout(self.relu(self.social_pooling_embedding(neighborhood)))
            
            # Combine for LSTM input
            lstm_inputs[:, t] = torch.cat((pos_embedded, social_context), dim=1)
        
        # Process through LSTM
        lstm_out, (h_n, c_n) = self.lstm(lstm_inputs)
        
        # Initialize decoder state
        decoder_h = h_n[-1]  # [batch_size, hidden_size]
        decoder_c = c_n[-1]  # [batch_size, hidden_size]
        
        # Storage for predictions
        predicted_positions = []
        confidences = []
        
        # Current position starts as the last observed position
        current_pos = vehicle_positions[:, -1]
        
        # Generate predictions for future frames
        for t in range(self.output_frames):
            # Update LSTM state
            decoder_h, decoder_c = self.decoder_rnn(
                decoder_h,  # Previous hidden state used as input
                (decoder_h, decoder_c)
            )
            
            # Predict position and confidence
            pos_pred = self.output_position(decoder_h).squeeze(1)
            conf_pred = self.sigmoid(self.output_confidence(decoder_h)).squeeze(1)
            
            # Store predictions
            predicted_positions.append(pos_pred)
            confidences.append(conf_pred)
            
            # Update current position for next iteration (detached to prevent backprop through predictions)
            current_pos = pos_pred.detach()
        
        # Stack predictions
        predicted_positions = torch.stack(predicted_positions, dim=1)  # [batch_size, output_frames]
        confidences = torch.stack(confidences, dim=1)  # [batch_size, output_frames]
        
        return predicted_positions, confidences