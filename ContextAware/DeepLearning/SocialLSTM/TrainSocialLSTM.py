import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
# Custom Dataset
import torch.nn.functional as F
from tqdm import tqdm
import json
# import focal loss

def extract_trajectories(scenario_data):
        """
        Extract individual vehicle trajectories from scenario data
        
        Args:
            scenario_data: Lane occupancy with trajectory IDs [lane_cell_num, seq_len]
            
        Returns:
            vehicle_trajectories: List of (vehicle_id, positions) tuples
        """
        # Find all unique vehicle IDs (excluding 0 which means unoccupied)
        unique_ids = torch.unique(scenario_data)
        unique_ids = unique_ids[unique_ids > 0]
        
        vehicle_trajectories = []
        
        # Extract trajectory for each vehicle ID
        for vehicle_id in unique_ids:
            # Find positions where this vehicle is present at each time step
            positions = []
            for t in range(scenario_data.size(1)):  # Iterate through time steps
                # Find lane cells occupied by this vehicle at time t
                occupied_cells = torch.nonzero(scenario_data[:, t] == vehicle_id, as_tuple=True)[0]
                
                # If vehicle is present at this time step, record its position
                # (taking the first cell if multiple are occupied)
                if len(occupied_cells) > 0:
                    positions.append(occupied_cells[0].item())
                else:
                    positions.append(-1)  # -1 indicates vehicle not present
            
            # Filter out trajectories with too many missing values
            if positions.count(-1) < scenario_data.size(1) // 2:  # At least half of positions are valid
                vehicle_trajectories.append((vehicle_id.item(), positions))
                
        return vehicle_trajectories

def prepare_training_data(scenario_data, window_size=11,input_frames = 10, pred_frames = 1, lane_cell_num = 200):
    """
    Prepare training data from scenario data using sliding window
    
    Args:
        scenario_data: Lane occupancy with trajectory IDs [lane_cell_num, seq_len]
        window_size: Total window size (input_frames + pred_frames)
        
    Returns:
        input_windows: List of input windows [input_frames, lane_cell_num]
        target_positions: List of target positions for each input window
        vehicle_ids: List of vehicle IDs for each input window
    """
    # Extract trajectories
    vehicle_trajectories = extract_trajectories(scenario_data)
    
    input_windows = []
    target_positions = []
    vehicle_ids = []
    
    # For each vehicle trajectory
    for vehicle_id, positions in vehicle_trajectories:
        # Create sliding windows
        for i in range(len(positions) - window_size + 1):
            # Skip if any position in this window is missing
            if -1 in positions[i:i+window_size]:
                continue
            
            # Extract window
            window = positions[i:i+window_size]
            
            # Input frames and target
            input_pos = window[:input_frames]
            target_pos = window[input_frames:input_frames+pred_frames]
            
            # Create window representation
            window_data = torch.zeros(input_frames, lane_cell_num)
            for t, pos in enumerate(input_pos):
                # Set the vehicle's position to 1 at each time step
                window_data[t, pos] = 1
            
            input_windows.append(window_data)
            target_positions.append(target_pos)
            vehicle_ids.append(vehicle_id)
    
    return input_windows, target_positions, vehicle_ids


def train_model(scenarios, optimizer, criterion=nn.MSELoss()):
        """
        Training function for the Lane Occupancy Social LSTM
        
        Args:
            scenarios: List of scenario data tensors [lane_cell_num, seq_len]
            optimizer: PyTorch optimizer
            criterion: Loss function
            
        Returns:
            loss: Average training loss
        """
        total_loss = 0
        total_samples = 0
        
        # Process each scenario
        for scenario_data in scenarios:
            # Prepare training data
            input_windows, target_positions, vehicle_ids = prepare_training_data(scenario_data)
            
            # Skip if no valid training samples
            if len(input_windows) == 0:
                continue
                
            # Process each sample
            for input_window, target_pos, vehicle_id in zip(input_windows, target_positions, vehicle_ids):
                # Forward pass
                pred_dist, pred_pos = self.forward(input_window, scenario_data, vehicle_id, target_pos)
                
                # Calculate loss
                target_tensor = torch.tensor(target_pos).float()
                if self.use_cuda:
                    target_tensor = target_tensor.cuda()
                
                # Loss based on predicted mean (first output dimension)
                pred_mean = pred_dist[:, 0]
                loss = criterion(pred_mean, target_tensor)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_samples += 1
        
        # Return average loss
        return total_loss / total_samples if total_samples > 0 else 0.0