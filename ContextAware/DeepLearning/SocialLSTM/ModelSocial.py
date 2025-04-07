import torch
import torch.nn as nn

class ImprovedSocialLSTM(nn.Module):
    def __init__(self, 
                 hidden_size=64,         # LSTM hidden state size
                 social_size=32,         # Social context embedding size
                 input_frames=10,        # Number of input frames 
                 output_size=2,          # Number of outputs: [position, confidence]
                 dropout=0.2,            # Dropout probability
                 use_cuda=True):         # GPU acceleration
        """
        Improved Social LSTM with confidence output
        """
        super(ImprovedSocialLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.social_size = social_size
        self.input_frames = input_frames
        self.output_size = output_size
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        # Position embedding
        self.position_embedding = nn.Linear(1, hidden_size)
        
        # Social context embeddings with explicit presence flags
        self.front_vehicle_embedding = nn.Linear(2, social_size)  # [distance, presence_flag]
        self.back_vehicle_embedding = nn.Linear(2, social_size)   # [distance, presence_flag]
        
        # Combine social embeddings into context
        self.social_context_combine = nn.Linear(2 * social_size, hidden_size)
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=2 * hidden_size,  # Position embedding + social context
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout if input_frames > 1 else 0
        )
        
        # Output layer - now outputs position and confidence
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # For confidence scoring
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        """
        Forward pass with confidence output
        
        Args:
            inputs: Input tensor structure [batch_size, input_frames, 5]
                inputs[:,:,0] = tracked vehicle position
                inputs[:,:,1] = distance to front vehicle (0 if none)
                inputs[:,:,2] = front vehicle presence flag (0 or 1)
                inputs[:,:,3] = distance to back vehicle (0 if none)
                inputs[:,:,4] = back vehicle presence flag (0 or 1)
            
        Returns:
            outputs: Predicted [position, confidence] [batch_size, output_size]
        """
        batch_size = inputs.size(0)
        seq_length = inputs.size(1)
        
        # Prepare output tensor
        lstm_inputs = torch.zeros(batch_size, seq_length, 2 * self.hidden_size)
        if self.use_cuda:
            lstm_inputs = lstm_inputs.cuda()
        
        # Process each time step to create embeddings
        for t in range(seq_length):
            # Extract data for current time step
            pos = inputs[:, t, 0].unsqueeze(1)  # [batch, 1]
            
            front_data = inputs[:, t, 1:3]      # [batch, 2] - [distance, presence]
            back_data = inputs[:, t, 3:5]       # [batch, 2] - [distance, presence]
            
            # Embed position
            pos_embedded = self.dropout(self.relu(self.position_embedding(pos)))
            
            # Embed social context with presence flags
            front_embedded = self.dropout(self.relu(self.front_vehicle_embedding(front_data)))
            back_embedded = self.dropout(self.relu(self.back_vehicle_embedding(back_data)))
            
            # Combine social context
            social_context = torch.cat((front_embedded, back_embedded), dim=1)
            social_embedded = self.dropout(self.relu(self.social_context_combine(social_context)))
            
            # Combine all features
            lstm_inputs[:, t] = torch.cat((pos_embedded, social_embedded), dim=1)
        
        # Process through LSTM
        lstm_out, (h_n, _) = self.lstm(lstm_inputs)
        
        # Get raw outputs
        raw_output = self.output_layer(h_n[-1])
        
        # Split into position and confidence
        position = raw_output[:, 0]
        confidence = self.sigmoid(raw_output[:, 1])  # Sigmoid to get 0-1 confidence
        
        # Combine into final output
        output = torch.cat((position.unsqueeze(1), confidence.unsqueeze(1)), dim=1)
        
        return output

    def prepare_input_data(self, positions, front_distances, back_distances):
        """
        Prepare input data with proper presence flags
        
        Args:
            positions: Positions of tracked vehicles [batch, seq_len]
            front_distances: Distances to front vehicles [batch, seq_len] (or None)
            back_distances: Distances to back vehicles [batch, seq_len] (or None)
            
        Returns:
            inputs: Formatted input tensor [batch, seq_len, 5]
        """
        batch_size = positions.size(0)
        seq_len = positions.size(1)
        
        # Create input tensor with proper structure
        inputs = torch.zeros(batch_size, seq_len, 5)
        if self.use_cuda:
            inputs = inputs.cuda()
        
        # Set positions
        inputs[:, :, 0] = positions
        
        # Set front vehicle info with presence flags
        if front_distances is not None:
            # Where distances are valid (not None), set presence flag to 1
            front_present = (front_distances != 0).float()
            inputs[:, :, 1] = front_distances
            inputs[:, :, 2] = front_present
        
        # Set back vehicle info with presence flags
        if back_distances is not None:
            # Where distances are valid (not None), set presence flag to 1
            back_present = (back_distances != 0).float()
            inputs[:, :, 3] = back_distances
            inputs[:, :, 4] = back_present
        
        return inputs
        
    def train_step(self, positions, front_distances, back_distances, targets, optimizer, criterion=None):
        """
        Training step with confidence-aware loss
        
        Args:
            positions: Positions of tracked vehicles [batch, seq_len]
            front_distances: Distances to front vehicles [batch, seq_len] (or None)
            back_distances: Distances to back vehicles [batch, seq_len] (or None)
            targets: Target positions [batch, 1]
            optimizer: PyTorch optimizer
            criterion: Loss function (if None, use custom confidence-aware loss)
            
        Returns:
            loss: Training loss
        """
        # Prepare input data
        inputs = self.prepare_input_data(positions, front_distances, back_distances)
        
        # Forward pass
        outputs = self.forward(inputs)
        
        # Extract predicted position and confidence
        pred_position = outputs[:, 0]
        pred_confidence = outputs[:, 1]
        
        # If no custom criterion is provided, use confidence-aware MSE loss
        if criterion is None:
            # Position error
            position_error = (pred_position - targets.squeeze(1)) ** 2
            
            # Scale error by confidence and add confidence regularization
            # Higher confidence → higher penalty for being wrong
            # Lower confidence → lower penalty but penalize low confidence
            confidence_penalty = -torch.log(pred_confidence)
            loss = (position_error * pred_confidence + confidence_penalty).mean()
        else:
            # Use provided criterion
            loss = criterion(pred_position, targets.squeeze(1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()