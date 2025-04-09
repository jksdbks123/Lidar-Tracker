import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LaneSocialLSTM(nn.Module):
    def __init__(self, 
                 input_frames=10,
                 output_frames=12,
                 lane_cells=200,
                 hidden_size=64,
                 social_size=32,
                 neighborhood_size=20,  # Size of neighborhood to consider in lane cells
                 num_layers=1,
                 dropout=0.2,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Social LSTM model adapted for lane-based traffic context diagrams
        
        Args:
            input_frames: Number of input frames to observe
            output_frames: Number of frames to predict
            lane_cells: Number of cells in the lane (200 in your case)
            hidden_size: Size of LSTM hidden state
            social_size: Size of social context embedding
            neighborhood_size: Size of neighborhood to consider for social context
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            device: Device to run the model on
        """
        super(LaneSocialLSTM, self).__init__()
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.lane_cells = lane_cells
        self.hidden_size = hidden_size
        self.social_size = social_size
        self.neighborhood_size = neighborhood_size
        self.device = device
        
        # Position and vehicle embedding
        self.position_embedding = nn.Linear(1, hidden_size)
        
        # Social context embedding
        self.social_pooling = nn.Linear(neighborhood_size, social_size)
        
        # Combined embedding
        self.combined_embedding = nn.Linear(hidden_size + social_size, hidden_size)
        
        # LSTM cell
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer for position prediction
        self.output_layer = nn.Linear(hidden_size, 2)  # position and confidence
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(dropout)
        
        self.to(device)
    
    def extract_vehicle_trajectories(self, traffic_context, start_frame, num_frames):
        """
        Extract individual vehicle trajectories from traffic context diagram
        
        Args:
            traffic_context: Traffic context diagram [batch_size, lane_cells, time_span]
            start_frame: Starting frame index
            num_frames: Number of frames to extract
            
        Returns:
            vehicle_positions: Dict mapping vehicle_id to trajectory positions
            vehicle_frames: Dict mapping vehicle_id to frame indices
        """
        batch_size = traffic_context.shape[0]
        vehicle_positions = {}
        vehicle_frames = {}
        
        for b in range(batch_size):
            vehicle_positions[b] = {}
            vehicle_frames[b] = {}
            
            # Extract frames of interest
            context_window = traffic_context[b, :, start_frame:start_frame+num_frames]
            
            # Find unique vehicle IDs (excluding 0 which is empty)
            vehicle_ids = torch.unique(context_window)
            vehicle_ids = vehicle_ids[vehicle_ids > 0]
            
            for vid in vehicle_ids:
                vid_int = vid.item()
                positions = []
                frames = []
                
                # For each frame, find the vehicle position (front-most cell)
                for t in range(num_frames):
                    # Get positions where this vehicle exists in this frame
                    vehicle_mask = (context_window[:, t] == vid)
                    if torch.any(vehicle_mask):
                        # Get front-most position (smallest index)
                        pos_indices = torch.nonzero(vehicle_mask)
                        front_pos = torch.min(pos_indices)
                        positions.append(front_pos.item())
                        frames.append(t)
                
                if positions:  # Only add if vehicle was visible
                    vehicle_positions[b][vid_int] = positions
                    vehicle_frames[b][vid_int] = frames
        
        return vehicle_positions, vehicle_frames
    
    def get_social_context(self, traffic_context, frame_idx, vehicle_id, position):
        """
        Extract social context (neighborhood) for a vehicle
        
        Args:
            traffic_context: Traffic context diagram [lane_cells, time_span]
            frame_idx: Current frame index
            vehicle_id: ID of the vehicle
            position: Current position of the vehicle
            
        Returns:
            social_context: Binary occupancy vector of neighborhood [neighborhood_size]
        """
        # Create binary context (1 for occupied, 0 for empty)
        social_context = torch.zeros(self.neighborhood_size, device=self.device)
        
        # Calculate neighborhood boundaries
        start_idx = max(0, position - self.neighborhood_size // 2)
        end_idx = min(self.lane_cells, start_idx + self.neighborhood_size)
        
        # Extract neighborhood occupancy
        neighborhood = traffic_context[start_idx:end_idx, frame_idx]
        
        # Mark cells that are occupied by other vehicles
        mask = (neighborhood > 0) & (neighborhood != vehicle_id)
        
        # Map to social context tensor
        offset = 0
        width = end_idx - start_idx
        social_context[:width] = mask.float()
        
        return social_context
            
    def forward(self, traffic_context, target_vehicle_id=None):
        """
        Forward pass through the Social LSTM model
        
        Args:
            traffic_context: Traffic context diagram [batch_size, lane_cells, time_span]
            target_vehicle_id: Optional ID of target vehicle to predict (if None, predict all)
            
        Returns:
            predictions: Dict mapping vehicle_id to predicted trajectories
            confidences: Dict mapping vehicle_id to prediction confidences
        """
        batch_size = traffic_context.shape[0]
        
        # Extract vehicle trajectories from input frames
        vehicle_positions, vehicle_frames = self.extract_vehicle_trajectories(
            traffic_context, 
            0,  # Start from first frame
            self.input_frames
        )
        
        # Initialize prediction containers
        predictions = {b: {} for b in range(batch_size)}
        confidences = {b: {} for b in range(batch_size)}
        
        # Process each batch and each vehicle
        for b in range(batch_size):
            # Filter vehicles if target_vehicle_id is specified
            if target_vehicle_id is not None:
                vehicle_ids = [target_vehicle_id] if target_vehicle_id in vehicle_positions[b] else []
            else:
                vehicle_ids = list(vehicle_positions[b].keys())
            
            for vid in vehicle_ids:
                # Skip vehicles with insufficient data
                if len(vehicle_positions[b][vid]) < self.input_frames // 2:
                    continue
                
                # Initialize LSTM hidden state
                h = torch.zeros(1, 1, self.hidden_size, device=self.device)
                c = torch.zeros(1, 1, self.hidden_size, device=self.device)
                
                # Process input frames
                for t in range(self.input_frames):
                    # Skip if vehicle not visible in this frame
                    if t not in vehicle_frames[b][vid]:
                        continue
                    
                    # Get index in vehicle's trajectory
                    idx = vehicle_frames[b][vid].index(t)
                    pos = vehicle_positions[b][vid][idx]
                    
                    # Embed position
                    pos_tensor = torch.tensor([[pos]], dtype=torch.float, device=self.device)
                    pos_embedded = self.position_embedding(pos_tensor)
                    
                    # Get social context
                    social_context = self.get_social_context(
                        traffic_context[b], 
                        t, 
                        vid, 
                        pos
                    )
                    social_embedded = self.social_pooling(social_context.unsqueeze(0))
                    
                    # Combine embeddings
                    combined = torch.cat((pos_embedded, social_embedded), dim=2)
                    inputs = self.dropout_layer(self.relu(self.combined_embedding(combined)))
                    
                    # LSTM step
                    _, (h, c) = self.lstm(inputs, (h, c))
                
                # Generate predictions for output frames
                pred_positions = []
                pred_confidences = []
                
                last_pos = vehicle_positions[b][vid][-1]
                
                for t in range(self.output_frames):
                    # Predict next position and confidence
                    output = self.output_layer(h.squeeze(0))
                    pos_delta = output[0, 0]  # Position change prediction
                    confidence = self.sigmoid(output[0, 1])  # Prediction confidence
                    
                    # Calculate absolute position
                    if t == 0:
                        next_pos = last_pos + pos_delta
                    else:
                        next_pos = pred_positions[-1] + pos_delta
                    
                    # Store predictions
                    pred_positions.append(next_pos.item())
                    pred_confidences.append(confidence.item())
                    
                    # Update social context for next prediction
                    # Create temporary context with predicted position
                    temp_context = traffic_context[b].clone()
                    cell_idx = int(next_pos)
                    if 0 <= cell_idx < self.lane_cells:
                        temp_context[cell_idx, self.input_frames + t] = vid
                    
                    # Get updated social context
                    social_context = self.get_social_context(
                        temp_context, 
                        self.input_frames + t, 
                        vid, 
                        int(next_pos)
                    )
                    social_embedded = self.social_pooling(social_context.unsqueeze(0))
                    
                    # Embed predicted position
                    pos_tensor = torch.tensor([[next_pos]], dtype=torch.float, device=self.device)
                    pos_embedded = self.position_embedding(pos_tensor)
                    
                    # Combine embeddings
                    combined = torch.cat((pos_embedded, social_embedded), dim=2)
                    inputs = self.dropout_layer(self.relu(self.combined_embedding(combined)))
                    
                    # LSTM step
                    _, (h, c) = self.lstm(inputs, (h, c))
                
                # Store vehicle predictions
                predictions[b][vid] = pred_positions
                confidences[b][vid] = pred_confidences
        
        return predictions, confidences