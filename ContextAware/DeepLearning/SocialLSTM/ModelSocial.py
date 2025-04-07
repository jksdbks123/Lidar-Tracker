import torch
import torch.nn as nn
import numpy as np

class LaneOccupancySocialLSTM(nn.Module):
    def __init__(self, 
                 embedding_size=64,        # Embedding dimension
                 rnn_size=128,             # LSTM hidden state size
                 lane_cell_num=200,        # Number of lane cells
                 input_frames=10,          # Number of input frames
                 pred_frames=1,            # Number of frames to predict
                 output_size=2,            # Parameters for output distribution (mean, std)
                 dropout=0.2,              # Dropout probability
                 use_cuda=True,            # GPU acceleration
                 gru=False):               # Use GRU instead of LSTM
        """
        Lane Occupancy Social LSTM for traffic trajectory prediction
        
        Works with trajectory data extracted from scenarios
        """
        super(LaneOccupancySocialLSTM, self).__init__()

        # Store parameters
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.lane_cell_num = lane_cell_num
        self.input_frames = input_frames
        self.pred_frames = pred_frames
        self.output_size = output_size
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.gru = gru

        # Embedding layers
        self.position_embedding_layer = nn.Linear(1, self.embedding_size)
        self.social_embedding_layer = nn.Linear(2*self.embedding_size, self.embedding_size)
        
        # RNN Cell (LSTM or GRU)
        if self.gru:
            self.cell = nn.GRUCell(2*self.embedding_size, self.rnn_size)
        else:
            self.cell = nn.LSTMCell(2*self.embedding_size, self.rnn_size)
            
        # Output layer
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def find_neighbors(self, scenario_data, target_vehicle_id, target_pos, time_step):
        """
        Find front and back vehicles at a specific time step
        
        Args:
            scenario_data: Lane occupancy with trajectory IDs [lane_cell_num, seq_len]
            target_vehicle_id: ID of the vehicle we're finding neighbors for
            target_pos: Position of the target vehicle
            time_step: Current time step
            
        Returns:
            front_distance: Distance to front vehicle (in cells), -1 if none
            back_distance: Distance to back vehicle (in cells), -1 if none
        """
        # Get occupancy at current time step
        current_occupancy = scenario_data[:, time_step]
        
        # Find front vehicle
        front_distance = -1
        for pos in range(target_pos + 1, self.lane_cell_num):
            if current_occupancy[pos] > 0 and current_occupancy[pos] != target_vehicle_id:
                front_distance = pos - target_pos
                break
        
        # Find back vehicle
        back_distance = -1
        for pos in range(target_pos - 1, -1, -1):
            if current_occupancy[pos] > 0 and current_occupancy[pos] != target_vehicle_id:
                back_distance = pos - target_pos  # Will be negative
                break
                
        return front_distance, back_distance

    def forward(self, input_window, scenario_data, vehicle_id, target_position):
        """
        Forward pass for a single vehicle trajectory
        
        Args:
            input_window: Input window [input_frames, lane_cell_num]
            scenario_data: Full scenario data [lane_cell_num, seq_len]
            vehicle_id: ID of the vehicle we're predicting
            target_position: Target position(s) for evaluation [pred_frames]
            
        Returns:
            predicted_distribution: Predicted distribution parameters [pred_frames, output_size]
            position_predictions: Predicted positions [pred_frames]
        """
        # Extract vehicle positions from input window
        positions = []
        for t in range(self.input_frames):
            pos = torch.nonzero(input_window[t] > 0, as_tuple=True)[0]
            if len(pos) > 0:
                positions.append(pos[0].item())
            else:
                positions.append(-1)
        
        # Initialize hidden and cell states
        hidden_state = torch.zeros(1, self.rnn_size)
        if self.use_cuda:
            hidden_state = hidden_state.cuda()
            
        if not self.gru:
            cell_state = torch.zeros(1, self.rnn_size)
            if self.use_cuda:
                cell_state = cell_state.cuda()
        
        # Process input sequence
        for t in range(self.input_frames):
            current_pos = positions[t]
            
            # Skip if position is invalid
            if current_pos == -1:
                continue
            
            # Find front and back neighbors
            # Assuming input_window starts at time 0 in scenario_data for simplicity
            # Adjust the time index as needed for your specific data
            front_dist, back_dist = self.find_neighbors(scenario_data, vehicle_id, current_pos, t)
            
            # Convert distances to meters (0.5m per cell)
            front_dist_m = front_dist * 0.5 if front_dist != -1 else -1
            back_dist_m = back_dist * 0.5 if back_dist != -1 else -1
            
            # Embed position
            pos_input = torch.tensor([[current_pos * 0.5]])  # Convert to meters
            if self.use_cuda:
                pos_input = pos_input.cuda()
            position_embedded = self.dropout(self.relu(self.position_embedding_layer(pos_input)))
            
            # Embed front and back vehicle info
            front_input = torch.tensor([[front_dist_m if front_dist_m != -1 else 0]]).float()
            back_input = torch.tensor([[back_dist_m if back_dist_m != -1 else 0]]).float()
            if self.use_cuda:
                front_input = front_input.cuda()
                back_input = back_input.cuda()
            
            front_embedded = self.position_embedding_layer(front_input)
            back_embedded = self.position_embedding_layer(back_input)
            
            # If no vehicle, zero out the embedding
            if front_dist == -1:
                front_embedded.zero_()
            if back_dist == -1:
                back_embedded.zero_()
            
            # Combine front and back embeddings
            neighbor_context = torch.cat((front_embedded, back_embedded), dim=1)
            social_embedded = self.dropout(self.relu(self.social_embedding_layer(neighbor_context)))
            
            # Concatenate with position embedding
            concat_embedded = torch.cat((position_embedded, social_embedded), dim=1)
            
            # LSTM/GRU step
            if self.gru:
                hidden_state = self.cell(concat_embedded, hidden_state)
            else:
                hidden_state, cell_state = self.cell(concat_embedded, (hidden_state, cell_state))
        
        # Predict future trajectory
        predicted_distribution = []
        position_predictions = []
        current_pos = positions[-1]  # Last position from input sequence
        
        for t in range(self.pred_frames):
            # Predict output distribution
            output_dist = self.output_layer(hidden_state)
            predicted_distribution.append(output_dist.squeeze(0))
            
            # Extract mean prediction (assuming first output is mean)
            position_change = output_dist[0, 0].item()
            
            # Update position
            new_pos = int(round(current_pos + position_change))
            new_pos = max(0, min(new_pos, self.lane_cell_num - 1))  # Clamp to lane bounds
            position_predictions.append(new_pos)
            current_pos = new_pos
            
            # If predicting multiple frames, prepare for next time step
            if t < self.pred_frames - 1:
                # Find front and back neighbors at new position
                front_dist, back_dist = self.find_neighbors(
                    scenario_data, vehicle_id, current_pos, self.input_frames + t
                )
                
                # Convert distances to meters
                front_dist_m = front_dist * 0.5 if front_dist != -1 else -1
                back_dist_m = back_dist * 0.5 if back_dist != -1 else -1
                
                # Embed position
                pos_input = torch.tensor([[current_pos * 0.5]])  # Convert to meters
                if self.use_cuda:
                    pos_input = pos_input.cuda()
                position_embedded = self.dropout(self.relu(self.position_embedding_layer(pos_input)))
                
                # Embed front and back vehicle info
                front_input = torch.tensor([[front_dist_m if front_dist_m != -1 else 0]]).float()
                back_input = torch.tensor([[back_dist_m if back_dist_m != -1 else 0]]).float()
                if self.use_cuda:
                    front_input = front_input.cuda()
                    back_input = back_input.cuda()
                print(front_input.dtype,back_input.dtype)
                
                front_embedded = self.position_embedding_layer(front_input)
                back_embedded = self.position_embedding_layer(back_input)
                
                # If no vehicle, zero out the embedding
                if front_dist == -1:
                    front_embedded.zero_()
                if back_dist == -1:
                    back_embedded.zero_()
                
                # Combine front and back embeddings
                neighbor_context = torch.cat((front_embedded, back_embedded), dim=1)
                social_embedded = self.dropout(self.relu(self.social_embedding_layer(neighbor_context)))
                
                # Concatenate with position embedding
                concat_embedded = torch.cat((position_embedded, social_embedded), dim=1)
                
                # LSTM/GRU step
                if self.gru:
                    hidden_state = self.cell(concat_embedded, hidden_state)
                else:
                    hidden_state, cell_state = self.cell(concat_embedded, (hidden_state, cell_state))
        
        return torch.stack(predicted_distribution), torch.tensor(position_predictions)