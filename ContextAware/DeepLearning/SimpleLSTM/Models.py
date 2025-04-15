import torch
import torch.nn as nn

class TrajectoryLSTM(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.2):
        """
        LSTM model for trajectory prediction
        
        Args:
            hidden_size: Number of features in hidden state
            num_layers: Number of recurrent layers
            dropout: Dropout rate
        """
        super(TrajectoryLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=1,          # One feature (position)
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x, mask):
        """
        Forward pass with handling of occlusions
        
        Args:
            x: Input tensor [batch_size, sequence_length]
            mask: Boolean mask indicating non-occluded positions [batch_size, sequence_length]
        """
        batch_size = x.size(0)
        
        # Create valid indices tensor from mask
        valid_indices = mask.float() * torch.arange(x.size(1), device=x.device).float().unsqueeze(0)
        valid_indices = valid_indices.masked_fill(~mask, float('inf')).sort()[0]
        valid_indices = valid_indices.masked_fill(valid_indices == float('inf'), 0).long()
        
        # Replace occluded positions with interpolated values
        x_interp = x.clone()
        
        for i in range(batch_size):
            # Find positions of non-occluded frames
            valid_pos = torch.nonzero(mask[i]).squeeze(-1)
            
            if len(valid_pos) <= 1:
                # Not enough valid points for interpolation, use forward filling
                if len(valid_pos) == 1:
                    x_interp[i] = x[i, valid_pos[0]]
                continue
            
            # For each occluded position, interpolate from nearest non-occluded positions
            for j in range(x.size(1)):
                if not mask[i, j]:
                    # Find nearest valid positions before and after
                    prev_valid = valid_pos[valid_pos < j].max() if len(valid_pos[valid_pos < j]) > 0 else valid_pos[0]
                    next_valid = valid_pos[valid_pos > j].min() if len(valid_pos[valid_pos > j]) > 0 else valid_pos[-1]
                    
                    # Calculate interpolation weights
                    if prev_valid == next_valid:
                        # Edge case - use the value directly
                        x_interp[i, j] = x[i, prev_valid]
                    else:
                        # Linear interpolation
                        t = (j - prev_valid) / (next_valid - prev_valid)
                        x_interp[i, j] = (1 - t) * x[i, prev_valid] + t * x[i, next_valid]
        
        # Reshape for LSTM input [batch_size, sequence_length, input_size]
        x_interp = x_interp.unsqueeze(-1)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Process through LSTM
        output, (h_n, _) = self.lstm(x_interp, (h0, c0))
        
        # Use last hidden state for prediction
        pred = self.fc(h_n[-1])
        
        return pred